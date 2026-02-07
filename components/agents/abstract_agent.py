import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.distributions import Categorical

from components.models.feature_encoder import FeatureEncoder
from components.models.navigation_policy import NavigationPolicy
from components.utils.rollout_buffer import RolloutBuffer


class AbstractAgent(nn.Module):
    """
    Shared base class for policy-gradient agents with multi-head action space.
    """

    def __init__(self, env, navigation_config, agent_config, device=None, mapping_path=None):
        super().__init__()
        self.env = env
        self.navigation_config = navigation_config
        self.agent_config = agent_config

        self.alpha = agent_config.get("alpha", 1e-4)
        self.gamma = agent_config.get("gamma", 0.99)
        self.collision_loss_coef = agent_config.get("collision_loss_coef", 0.2)
        self.use_trajectory_info = agent_config.get("use_trajectory_info", True)

        self.use_transformer = navigation_config["use_transformer"]
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cache_state_features = agent_config.get("cache_state_features", False)

        # Action Space Config
        self.action_space_config = env.get_action_space_dims()
        self.use_legacy_actions = hasattr(env, "use_legacy_actions") and env.use_legacy_actions
        self.action_space_mode = getattr(env, "action_space_mode", "multi_head")

        # Dimensions
        rgb_dim = navigation_config["rgb_dim"]
        depth_dim = navigation_config.get("depth_dim", 128)
        action_dim = navigation_config["action_dim"]
        sg_dim = navigation_config["sg_dim"]

        # Depth usage flag (for Li et al. baseline comparison)
        use_depth = navigation_config.get("use_depth", True)
        self.use_depth = use_depth
        effective_depth_dim = depth_dim if use_depth else 0

        # Calculate input_dim first for auto policy_hidden
        # +32 for stagnation detection embedding
        stagnation_dim = 32
        input_dim_calculated = int(rgb_dim + effective_depth_dim + action_dim + 2 * sg_dim + stagnation_dim)

        # Policy hidden dim: defaults to input_dim if not specified
        policy_hidden_dim = navigation_config.get("policy_hidden", input_dim_calculated)

        obj_embedding_dim = navigation_config.get("obj_embedding_dim", 128)
        rel_embedding_dim = navigation_config.get("rel_embedding_dim", 64)
        max_object_types = navigation_config.get("max_object_types", 1000)
        max_relation_types = navigation_config.get("max_relation_types", 50)

        lssg_layers = navigation_config.get("lssg_layers", 2)
        graph_hidden = navigation_config.get("graph_hidden", 128)
        graph_heads = navigation_config.get("graph_heads", 4)
        graph_layers = navigation_config.get("graph_layers", 2)

        if mapping_path is None:
            mapping_path = os.path.join(os.path.dirname(__file__), "..", "data", "scene_graph_mappings", "default")

        self.encoder = FeatureEncoder(
            action_space_config=self.action_space_config,
            rgb_dim=rgb_dim,
            depth_dim=depth_dim,
            action_dim=action_dim,
            sg_dim=sg_dim,
            obj_embedding_dim=obj_embedding_dim,
            max_object_types=max_object_types,
            rel_embedding_dim=rel_embedding_dim,
            max_relation_types=max_relation_types,
            mapping_path=mapping_path,
            use_transformer=self.use_transformer,
            backbone=navigation_config.get("backbone", "resnet18"),
            lssg_layers=lssg_layers,
            graph_hidden=graph_hidden,
            graph_heads=graph_heads,
            graph_layers=graph_layers,
            freeze_backbone=True,  # Keep RGB frozen by default
            freeze_rgb_backbone=True,
            freeze_depth_backbone=False,
            use_cuda_streams=True,  # Use CUDA streams for parallel processing
            use_pinned_memory=True,  # Use pinned memory for non-blocking GPU transfers
            use_depth=use_depth,  # Disable depth for Li et al. baseline
        ).to(self.device)

        # Policy input dim = RGB + Depth + Action + 2*SG
        self.input_dim = input_dim_calculated

        stop_head_hidden_dim = navigation_config.get("stop_head_hidden_dim", 64)
        collision_head_hidden_dim = navigation_config.get("collision_head_hidden_dim", 128)

        self.policy = NavigationPolicy(
            input_dim=self.input_dim,
            hidden_dim=policy_hidden_dim,
            action_space_config=self.action_space_config,
            use_transformer=self.use_transformer,
            value_head=True if agent_config["name"] in ["ppo"] else False,
            device=self.device,
            use_legacy_actions=self.use_legacy_actions,
            action_dim=action_dim,
            rgb_dim=rgb_dim,
            depth_dim=effective_depth_dim,  # 0 when depth disabled, skips collision head
            sg_dim=sg_dim,
            stagnation_dim=stagnation_dim,
            stop_head_hidden_dim=stop_head_hidden_dim,
            collision_head_hidden_dim=collision_head_hidden_dim,
        ).to(self.device)

        self.optimizer = optim.Adam(self.parameters(), lr=self.alpha)
        self.rollout_buffers = RolloutBuffer(agent_config["num_steps"])

        # Last action initialized as START tuple [-1, -1]
        self.last_action = torch.tensor([-1, -1], dtype=torch.long, device=self.device)

        self.lssg_hidden = None
        self.gssg_hidden = None
        self.policy_hidden = None
        self.stagnation_state = None  # Tracks GSSG change for stagnation detection
        self.obs_buffer = []
        self.action_buffer = []

        self.to(self.device)

    def forward(self, obs):
        if self.use_transformer:
            self.obs_buffer.append(obs)
            if len(self.action_buffer) == 0:
                self.action_buffer.append(torch.tensor([-1, -1], dtype=torch.long, device=self.device))
            else:
                self.action_buffer.append(self.last_action)

            # Stack action buffer for batch processing
            action_seq = torch.stack(self.action_buffer).unsqueeze(0)  # [1, T, 2]
            state_vector, _, _, self.stagnation_state = self.encoder(self.obs_buffer, action_seq)
            logits_dict, value, _ = self.policy(state_vector)
        else:
            state_vector, self.lssg_hidden, self.gssg_hidden, self.stagnation_state = self.encoder(
                obs, self.last_action, lssg_hidden=self.lssg_hidden, gssg_hidden=self.gssg_hidden, stagnation_state=self.stagnation_state
            )
            logits_dict, value, self.policy_hidden = self.policy(state_vector, hidden=self.policy_hidden)

        return logits_dict, value.squeeze(-1) if value is not None else None

    def get_action(self, obs, deterministic: bool = False):
        with torch.no_grad():
            logits_or_dict, value = self.forward(obs)

            if self.use_legacy_actions:
                # Legacy: extract action logits from dict (policy returns dict with collision head)
                logits = logits_or_dict["action"]

                if self.use_transformer:
                    logits = logits[:, -1]
                else:
                    logits = logits.squeeze(0)

                probs = F.softmax(logits, dim=-1)
                dist = Categorical(probs)
                action_idx = torch.argmax(probs) if deterministic else dist.sample()

                prev_action = self.last_action

                # For legacy: action is just the index, but we store as [idx, -1] for compatibility
                current_action = torch.tensor([action_idx.item(), -1], dtype=torch.long, device=self.device)
                self.last_action = current_action

                val_scalar = value.item() if value is not None else None

                # Return single action index for environment
                return (
                    action_idx.item(),
                    self.lssg_hidden,
                    self.gssg_hidden,
                    self.policy_hidden,
                    prev_action,
                    val_scalar,
                    self.stagnation_state,
                )
            elif self.action_space_mode == "single_head_large":
                # Single-Head-Large: 504 flat actions with curriculum masking
                logits = logits_or_dict["action"]

                if self.use_transformer:
                    logits = logits[:, -1]
                else:
                    logits = logits.squeeze(0)

                # Apply curriculum mask (if available)
                if hasattr(self.env, "valid_flat_action_indices"):
                    mask = torch.zeros(logits.shape[-1], dtype=torch.bool, device=logits.device)
                    for idx in self.env.valid_flat_action_indices:
                        mask[idx] = True
                    logits = logits.masked_fill(~mask, float("-inf"))

                # Sample action
                probs = F.softmax(logits, dim=-1)
                dist = Categorical(probs)
                action_idx = torch.argmax(probs) if deterministic else dist.sample()

                prev_action = self.last_action

                # Decode flat index to (dir_idx, len_idx) for embedding compatibility
                dir_idx = action_idx.item() // 21
                len_idx = action_idx.item() % 21
                current_action = torch.tensor([dir_idx, len_idx], dtype=torch.long, device=self.device)
                self.last_action = current_action

                val_scalar = value.item() if value is not None else None

                # Return flat action index for environment
                return (
                    action_idx.item(),
                    self.lssg_hidden,
                    self.gssg_hidden,
                    self.policy_hidden,
                    prev_action,
                    val_scalar,
                    self.stagnation_state,
                )
            else:
                # Multi-head: dictionary with direction and length logits
                logits_dict = logits_or_dict
                logits_dir = logits_dict["direction"]
                logits_len = logits_dict["length"]
                logits_stop = logits_dict.get("stop")

                if self.use_transformer:
                    logits_dir = logits_dir[:, -1]
                    logits_len = logits_len[:, -1]
                    if logits_stop is not None:
                        logits_stop = logits_stop[:, -1]
                else:
                    logits_dir = logits_dir.squeeze(0)
                    logits_len = logits_len.squeeze(0)
                    if logits_stop is not None:
                        logits_stop = logits_stop.squeeze(0)

                # Apply curriculum masks if available
                if hasattr(self.env, "valid_direction_indices") and hasattr(self.env, "valid_length_indices"):
                    # Create boolean masks for valid actions
                    dir_valid_mask = torch.zeros(logits_dir.shape[-1], dtype=torch.bool, device=logits_dir.device)
                    for idx in self.env.valid_direction_indices:
                        dir_valid_mask[idx] = True

                    len_valid_mask = torch.zeros(logits_len.shape[-1], dtype=torch.bool, device=logits_len.device)
                    for idx in self.env.valid_length_indices:
                        len_valid_mask[idx] = True

                    # Apply masks: set invalid actions to -inf
                    logits_dir = logits_dir.masked_fill(~dir_valid_mask, float("-inf"))
                    logits_len = logits_len.masked_fill(~len_valid_mask, float("-inf"))

                # Use Categorical with logits directly (better numerical stability)
                dist_dir = Categorical(logits=logits_dir)
                dist_len = Categorical(logits=logits_len)
                dist_stop = Categorical(logits=logits_stop) if logits_stop is not None else None

                if deterministic:
                    action_dir = torch.argmax(logits_dir)
                    action_len = torch.argmax(logits_len)
                    action_stop = torch.argmax(logits_stop) if dist_stop is not None else torch.tensor(0, device=self.device)
                else:
                    # Stochastic sampling from policy distribution
                    action_dir = dist_dir.sample()
                    action_len = dist_len.sample()
                    action_stop = dist_stop.sample() if dist_stop is not None else torch.tensor(0, device=self.device)

                prev_action = self.last_action

                # Current action as tensor [2]
                current_action = torch.tensor([action_dir.item(), action_len.item()], dtype=torch.long, device=self.device)
                self.last_action = current_action

                val_scalar = value.item() if value is not None else None

                # Return tuple (dir, len, stop_flag) as 'action'
                return (
                    (action_dir.item(), action_len.item(), action_stop.item()),
                    self.lssg_hidden,
                    self.gssg_hidden,
                    self.policy_hidden,
                    prev_action,
                    val_scalar,
                    self.stagnation_state,
                )

    def reset(self):
        self.last_action = torch.tensor([-1, -1], dtype=torch.long, device=self.device)
        self.rollout_buffers.clear()
        self.stagnation_state = None
        if self.use_transformer:
            self.obs_buffer.clear()
            self.action_buffer.clear()
        else:
            self.lssg_hidden = None
            self.gssg_hidden = None
            self.policy_hidden = None

    def _prepare_batch_for_update(self, batch):
        """
        Normalize rollout batches (single or vectorized) for training.
        """
        keys = ["rgb", "depth", "lssg", "gssg", "actions", "returns", "rewards", "dones", "last_actions", "agent_positions", "move_success"]
        if self.use_trajectory_info:
            keys.append("trajectory_info")
        processed = {k: batch[k] for k in keys if k in batch}

        tensor_keys = ["actions", "returns", "rewards", "dones", "last_actions", "move_success"]
        for k in tensor_keys:
            if k not in processed:
                continue

            value = processed[k]
            if not isinstance(value, torch.Tensor):
                if (
                    k in ["actions", "last_actions"]
                    and isinstance(value, list)
                    and value
                    and isinstance(value[0], (tuple, list, torch.Tensor))
                ):
                    processed[k] = torch.tensor(value, dtype=torch.long, device=self.device)
                elif k == "dones":
                    processed[k] = torch.tensor(value, dtype=torch.bool, device=self.device)
                else:
                    processed[k] = torch.tensor(value, device=self.device)
            else:
                processed[k] = value.to(self.device)

            # Adjust dimensions: [T, 2] -> [1, T, 2] for actions
            if k in ["actions", "last_actions"] and processed[k].dim() == 2:
                processed[k] = processed[k].unsqueeze(0)
            elif k in ["returns", "rewards", "dones"] and processed[k].dim() == 1:
                processed[k] = processed[k].unsqueeze(0)

        for k in ["rgb", "depth", "lssg", "gssg", "agent_positions"]:
            if k not in processed:
                continue

            value = processed[k]

            # When batches arrive as stacked arrays/tensors (e.g., VecEnv rollouts),
            # wrap them as a single sequence so forward_seq sees B=1, T=len(value).
            if isinstance(value, (torch.Tensor, np.ndarray)):
                processed[k] = [value]
                continue

            if isinstance(value, list) and (not value or not isinstance(value[0], list)):
                processed[k] = [value]

        return processed

    def _get_update_values(self):
        raw_batch = self.rollout_buffers.get(self.gamma)
        return self._prepare_batch_for_update(raw_batch)

    def forward_update(self, batch, use_cached_features=False):
        """
        Forward pass for PPO update with proper trajectory handling.

        For VecEnv training, trajectories from different environments must be
        processed separately to maintain correct LSTM hidden state propagation.

        Args:
            batch: Batch dict with trajectory data (keys: rgb, depth, lssg, gssg, actions, last_actions, etc.)
            use_cached_features: If True, use cached encoder features from first epoch (memory optimization)

        Returns:
            logits_dict: Policy outputs (dict with action/direction/length/stop logits and optional collision)
            value: Value function estimates, or None if no value head
        """
        # Ensure batch follows expected shape conventions (B, T, ...)
        if not (batch.get("rgb") and isinstance(batch["rgb"][0], list)):
            batch = self._prepare_batch_for_update(batch)

        # Check if we have trajectory info (from VecEnvRolloutBuffer)
        trajectory_info = batch.get("trajectory_info", None) if self.use_trajectory_info else None

        if trajectory_info:
            # VecEnv mode: Process each trajectory separately with fresh hidden states
            all_logits_dir = []
            all_logits_len = []
            all_logits_stop = []
            all_logits_collision = []
            all_values = []

            if not self.cache_state_features and hasattr(self, "_cached_state_features"):
                del self._cached_state_features
            if self.cache_state_features and not use_cached_features:
                self._cached_state_features = []

            stateless = None
            if not (use_cached_features and self.cache_state_features and hasattr(self, "_cached_state_features")):
                stateless = self.encoder.precompute_stateless(batch, batch.get("last_actions"))
                if stateless["B"] != 1:
                    raise ValueError("Trajectory-aware update expects batch size B=1 after flattening.")

            for traj_idx, traj_info in enumerate(trajectory_info):
                start_idx = traj_info["start_idx"]
                length = traj_info["length"]
                end_idx = start_idx + length

                # Restore per-trajectory hidden states (if provided)
                init_hidden = traj_info.get("initial_hidden", None)
                if init_hidden is not None:
                    if len(init_hidden) == 4:
                        lssg_init, gssg_init, policy_init, stag_init = init_hidden
                    else:
                        lssg_init, gssg_init, policy_init = init_hidden
                        stag_init = None
                else:
                    lssg_init = gssg_init = policy_init = stag_init = None

                # MEMORY OPTIMIZATION: Reuse cached encoder features if available (PPO epochs 2-4)
                if use_cached_features and self.cache_state_features and hasattr(self, "_cached_state_features"):
                    state_seq = self._cached_state_features[traj_idx]
                    stop_target_seq = None
                else:
                    traj_stateless = {
                        "rgb_feat": stateless["rgb_feat"][:, start_idx:end_idx, :],
                        "act_feat": stateless["act_feat"][:, start_idx:end_idx, :],
                        "lssg_feat": stateless["lssg_feat"][:, start_idx:end_idx, :],
                        "gssg_feat": stateless["gssg_feat"][:, start_idx:end_idx, :],
                        "B": 1,
                        "T": end_idx - start_idx,
                    }
                    # Only include depth_feat if depth is used
                    if "depth_feat" in stateless:
                        traj_stateless["depth_feat"] = stateless["depth_feat"][:, start_idx:end_idx, :]

                    # Forward with FRESH hidden state for each trajectory
                    stop_target_seq = None
                    if self.use_legacy_actions:
                        state_seq, _, _, _ = self.encoder.forward_seq_from_stateless(
                            traj_stateless, lssg_hidden=lssg_init, gssg_hidden=gssg_init, stagnation_state=stag_init
                        )
                    else:
                        state_seq, _, _, _ = self.encoder.forward_seq_from_stateless(
                            traj_stateless, lssg_hidden=lssg_init, gssg_hidden=gssg_init, stagnation_state=stag_init
                        )

                    # Cache features for next epochs (detach to prevent memory leak)
                    if self.cache_state_features and not use_cached_features:
                        self._cached_state_features.append(state_seq.detach())

                logits_or_dict, value, _ = self.policy(state_seq, hidden=policy_init)

                if self.use_legacy_actions or self.action_space_mode == "single_head_large":
                    # Legacy mode also returns dict with auxiliary collision head
                    all_logits_dir.append(logits_or_dict["action"])
                    if "collision" in logits_or_dict:
                        all_logits_collision.append(logits_or_dict["collision"])
                else:
                    all_logits_dir.append(logits_or_dict["direction"])
                    all_logits_len.append(logits_or_dict["length"])
                    all_logits_stop.append(logits_or_dict["stop"])
                    if "collision" in logits_or_dict:
                        all_logits_collision.append(logits_or_dict["collision"])

                if value is not None:
                    all_values.append(value.squeeze(0))

            # Concatenate all trajectories
            if self.use_legacy_actions or self.action_space_mode == "single_head_large":
                logits = torch.cat(all_logits_dir, dim=1)
                value = torch.cat(all_values, dim=0) if all_values else None
                result_dict = {"action": logits}
                if all_logits_collision:
                    result_dict["collision"] = torch.cat(all_logits_collision, dim=1)
                return result_dict, value
            else:
                logits_dir = torch.cat(all_logits_dir, dim=1)
                logits_len = torch.cat(all_logits_len, dim=1)
                logits_stop = torch.cat(all_logits_stop, dim=1)
                value = torch.cat(all_values, dim=0) if all_values else None
                logits_dict = {"direction": logits_dir, "length": logits_len, "stop": logits_stop}
                if all_logits_collision:
                    logits_dict["collision"] = torch.cat(all_logits_collision, dim=1)
                return logits_dict, value
        else:
            # Single trajectory or legacy mode: Process as before
            state_seq, _, _, _ = self.encoder.forward_seq(
                batch, batch["last_actions"], lssg_hidden=None, gssg_hidden=None, stagnation_state=None
            )
            logits_dict, value, _ = self.policy(state_seq, hidden=None)

            if value is None:
                return logits_dict
            else:
                value = value.squeeze(0)
                return logits_dict, value

    def load_weights(self, encoder_path=None, model_path=None, device="cpu"):
        if encoder_path is not None:
            self.encoder.load_weights(encoder_path, device=device)
        elif model_path is not None:
            state_dict = torch.load(model_path, map_location=device)

            # Handle torch.compile checkpoints (remove _orig_mod prefix)
            if any(k.startswith("_orig_mod.") or "._orig_mod." in k for k in state_dict.keys()):
                state_dict = {k.replace("._orig_mod.", "."): v for k, v in state_dict.items()}

            self.load_state_dict(state_dict)
            self.to(device)
        else:
            raise Exception("No encoder or model specified")

    def _compute_collision_loss(self, logits_or_dict, actions_len, move_success, actions=None):
        """
        Compute auxiliary collision prediction loss from move_action_success labels.

        Shared method for all agents (PPO, REINFORCE, etc.) to learn collision prediction
        from depth maps.

        Args:
            logits_or_dict: Policy output (dict with 'collision' key)
            actions_len: Length indices for filtering movement actions (multi-head only, None for single_head_large)
            move_success: Binary labels (1=success, 0=collision)
            actions: Raw action indices (optional, used for single_head_large decoding)

        Returns:
            collision_loss: BCE loss for collision prediction (tensor, 0.0 if no valid data)
            collision_acc: Accuracy for logging, or None if no valid movement actions
        """
        if move_success is None:
            return torch.tensor(0.0, device=self.device), None

        # Extract collision logits (works for both legacy and multi-head)
        if not isinstance(move_success, torch.Tensor):
            move_success = torch.tensor(move_success, device=self.device, dtype=torch.float32)
        else:
            move_success = move_success.to(self.device).float()

        if not isinstance(logits_or_dict, dict) or "collision" not in logits_or_dict:
            return torch.tensor(0.0, device=self.device), None

        collision_logits = logits_or_dict["collision"].view(-1)
        move_success = move_success.view(-1)

        # Determine movement mask based on action space mode
        if self.use_legacy_actions:
            # Legacy: decode action index to extract len_idx
            # Legacy format: idx = angle_idx * 2 + len_idx, where len_idx 0=0.0m, 1=0.3m
            if actions is not None:
                if not isinstance(actions, torch.Tensor):
                    actions = torch.tensor(actions, device=self.device)
                legacy_actions = actions.view(-1)
                len_idx = legacy_actions % 2  # Extract len_idx (0 or 1)
                movement_mask = (len_idx > 0).float()  # Only supervise on len_idx=1 (0.3m movement)
            else:
                # Fallback: no masking if actions not provided
                movement_mask = torch.ones_like(move_success)
        elif actions_len is None and self.action_space_mode == "single_head_large":
            # Single-head-large: decode flat index to extract len_idx
            # flat_idx = dir_idx * 21 + len_idx
            if actions is not None:
                if not isinstance(actions, torch.Tensor):
                    actions = torch.tensor(actions, device=self.device)
                flat_actions = actions.view(-1)
                len_idx = flat_actions % 21  # Extract len_idx
                movement_mask = (len_idx > 0).float()
            else:
                # Fallback: no masking if actions not provided
                movement_mask = torch.ones_like(move_success)
        elif actions_len is None:
            # Other cases where actions_len is None: no masking
            movement_mask = torch.ones_like(move_success)
        else:
            # Multi-head: only supervise on movement actions (len > 0)
            movement_mask = (actions_len.view(-1) > 0).float()

        if movement_mask.sum() <= 0:
            return torch.tensor(0.0, device=self.device), None

        # Collision targets: 1 = collision, 0 = success
        collision_targets = 1.0 - move_success
        per_loss = torch.nn.functional.binary_cross_entropy_with_logits(collision_logits, collision_targets, reduction="none")
        collision_loss = (per_loss * movement_mask).sum() / movement_mask.sum()

        # Compute accuracy for logging
        with torch.no_grad():
            preds = (torch.sigmoid(collision_logits) > 0.5).float()
            collision_acc = ((preds == collision_targets).float() * movement_mask).sum() / movement_mask.sum()

        return collision_loss, collision_acc

    def save_model(self, path, file_name: str | None = None, save_encoder: bool = False):
        base_path = Path(path)
        base_path.mkdir(parents=True, exist_ok=True)

        rgb_dim = self.encoder.rgb_encoder.output_dim
        action_dim = self.encoder.action_emb.emb_dim
        if self.use_transformer:
            sg_dim = self.encoder.lssg_encoder.output_dim
        else:
            sg_dim = self.encoder.lssg_encoder.lstm.hidden_size

        policy_hidden_dim = self.navigation_config["policy_hidden"]

        agent_name = self.get_agent_info().get("Agent Name", "Agent").replace(" ", "_")
        suffix = "_Transformer" if self.use_transformer else "_LSTM"
        model_dir = base_path / f"{agent_name}{suffix}"
        model_dir.mkdir(exist_ok=True)

        if save_encoder:
            self.encoder.save_model(model_dir)

        default_filename = f"{agent_name}{suffix}_{rgb_dim}_{action_dim}_{sg_dim}_{policy_hidden_dim}.pth"
        filename = default_filename if file_name is None else file_name
        full_path = model_dir / filename

        torch.save(self.state_dict(), str(full_path))
        print(f"Saved model to {full_path}")

    def get_agent_info(self):
        raise NotImplementedError("Subclasses must implement get_agent_info().")
