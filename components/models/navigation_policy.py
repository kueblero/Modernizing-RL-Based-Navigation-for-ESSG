import os

import torch
import torch.nn as nn

from components.models.feature_encoder import PositionalEncoding


class NavigationPolicy(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        action_space_config,  # {'num_directions': 24, 'num_lengths': 15} or {'num_actions': 15}
        use_transformer=False,
        value_head=False,
        num_layers=2,
        num_heads=4,
        dropout=0.1,
        max_len=256,
        device=None,
        use_legacy_actions=False,
        action_dim=None,
        rgb_dim=None,
        depth_dim=None,
        sg_dim=None,
        stagnation_dim=32,
        stop_head_hidden_dim=64,
        collision_head_hidden_dim=128,
    ):
        super().__init__()
        self.use_transformer = use_transformer
        self.use_legacy_actions = use_legacy_actions
        self.hidden_dim = hidden_dim
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Detect action space mode
        if "num_actions" in action_space_config:
            num_actions = action_space_config["num_actions"]
            if num_actions > 16:
                self.action_space_mode = "single_head_large"
                self.use_legacy_actions = False
            else:
                self.action_space_mode = "legacy"
                self.use_legacy_actions = True
        else:
            self.action_space_mode = "multi_head"
            self.use_legacy_actions = False

        if use_transformer:
            self.input_proj = nn.Linear(input_dim, hidden_dim)
            self.pos_encoder = PositionalEncoding(hidden_dim, max_len=max_len)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim, dropout=dropout, batch_first=True
            )
            self.core = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.core_output_dim = hidden_dim
        else:
            self.core = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
            self.core_output_dim = hidden_dim

        self.shared = nn.Sequential(nn.Linear(self.core_output_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU())

        self.depth_slice = None
        self.stagnation_slice = None  # unused now; kept for compatibility
        if all(v is not None for v in (action_dim, rgb_dim, depth_dim, sg_dim, stagnation_dim)):
            expected_dim = action_dim + rgb_dim + depth_dim + (2 * sg_dim) + stagnation_dim
            if expected_dim != input_dim:
                raise ValueError(
                    f"Feature dim mismatch: input_dim={input_dim} but expected {expected_dim} from components."
                )
            # Only set depth_slice if depth is actually used (depth_dim > 0)
            if depth_dim > 0:
                depth_start = action_dim + rgb_dim
                self.depth_slice = slice(depth_start, depth_start + depth_dim)

        # Action Space: Single-head (legacy 16 or large 504) vs Multi-Head (24 directions × 21 lengths)
        if self.action_space_mode in ["legacy", "single_head_large"]:
            # Single-head: One action head with num_actions outputs (16 for legacy, 504 for single_head_large)
            num_actions = action_space_config.get("num_actions", 16)
            self.action_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, num_actions)
            )
            # Auxiliary collision prediction head (only if depth is used)
            if depth_dim is not None and depth_dim > 0 and self.depth_slice is not None:
                self.collision_head = nn.Sequential(
                    nn.Linear(depth_dim, collision_head_hidden_dim),
                    nn.ReLU(),
                    nn.Linear(collision_head_hidden_dim, 1),
                )
            else:
                self.collision_head = None
            self.dir_head = None
            self.len_head = None
            self.stop_head = None
        else:
            # Multi-Head: Separate heads for direction and length
            self.dir_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, action_space_config["num_directions"])
            )
            self.len_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, action_space_config["num_lengths"])
            )
            # Separate STOP head (binary: continue vs stop), uses full shared feature
            self.stop_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 2),
            )
            # Auxiliary collision prediction head (only if depth is used)
            if depth_dim is not None and depth_dim > 0 and self.depth_slice is not None:
                self.collision_head = nn.Sequential(
                    nn.Linear(depth_dim, collision_head_hidden_dim),
                    nn.ReLU(),
                    nn.Linear(collision_head_hidden_dim, 1),
                )
            else:
                self.collision_head = None
            self.action_head = None

        self.value_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)) if value_head else None

    def forward(self, seq, hidden=None, pad_mask=None):
        """
        Forward pass through policy network.

        Args:
            seq: Input feature sequence [B, T, feature_dim]
            hidden: LSTM hidden state (tuple of h, c) for non-transformer models
            pad_mask: Padding mask for transformer [B, T]

        Returns:
            logits_dict: Dictionary with action logits (format depends on action_space_mode)
                - "action": [B, T, num_actions] for legacy/single_head_large
                - "direction", "length", "stop": [B, T, ...] for multi_head
                - "collision": [B, T, 1] if depth is used (auxiliary head)
            value: Value estimates [B, T] or None if no value head
            hidden: Updated LSTM hidden state (None for transformer)
        """
        if seq.dim() == 2:
            seq = seq.unsqueeze(1)

        feature_seq = seq
        if self.use_transformer:
            hidden = None
            seq = self.input_proj(seq)
            seq = self.pos_encoder(seq)
            out = self.core(seq, src_key_padding_mask=pad_mask)
        else:
            out, hidden = self.core(seq, hidden)

        out = self.shared(out)

        stop_in = out

        # Return format depends on action space type
        if self.action_space_mode in ["legacy", "single_head_large"]:
            # Single-head: Single action logits tensor + collision prediction
            logits = self.action_head(out)
            result_dict = {"action": logits}
            # Only add collision if head exists (depth is used)
            if self.collision_head is not None and self.depth_slice is not None:
                collision_in = feature_seq[..., self.depth_slice]
                result_dict["collision"] = self.collision_head(collision_in)
            value = self.value_head(out).squeeze(-1) if self.value_head is not None else None
            return result_dict, value, hidden
        else:
            # Multi-head: Dictionary with direction and length logits
            logits_dir = self.dir_head(out)
            logits_len = self.len_head(out)
            logits_stop = self.stop_head(stop_in)
            result_dict = {"direction": logits_dir, "length": logits_len, "stop": logits_stop}
            # Only add collision if head exists (depth is used)
            if self.collision_head is not None and self.depth_slice is not None:
                collision_in = feature_seq[..., self.depth_slice]
                result_dict["collision"] = self.collision_head(collision_in)
            value = self.value_head(out).squeeze(-1) if self.value_head is not None else None
            return result_dict, value, hidden

    def save_model(self, path):
        # Adjusted save method for multi-head structure if needed, or rely on state_dict
        filename = f"navigation_policy_multihead_{self.use_transformer}.pth"
        torch.save(self.state_dict(), os.path.join(path, filename))

    def load_weights(self, model_path, device="cpu"):
        state_dict = torch.load(model_path, map_location=device)
        self.load_state_dict(state_dict)
        self.to(device)
