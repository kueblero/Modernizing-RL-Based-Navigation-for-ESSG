import hashlib
import json
import math
import os

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HGTConv
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights


class FeatureEncoder(nn.Module):
    """
    Encodes the multimodal agent state including:
    - RGB image via ResNet18 or similar backbone
    - Depth image via modified ResNet18
    - Last action via MultiHead embeddings (Direction + Length)
    - Occupancy map via CNN
    - Local and global scene graphs via LSTM or Transformer
    Combines all features into a single state vector.
    """

    def __init__(
        self,
        action_space_config,
        rgb_dim=512,
        depth_dim=128,
        action_dim=32,
        sg_dim=256,
        obj_embedding_dim=128,
        max_object_types=1000,
        rel_embedding_dim=64,
        max_relation_types=50,
        use_transformer=False,
        mapping_path=None,
        backbone="resnet18",
        lssg_layers=2,
        graph_hidden=128,
        graph_heads=4,
        graph_layers=2,
        freeze_backbone=True,
        freeze_rgb_backbone=None,
        freeze_depth_backbone=None,
        use_cuda_streams=True,
        use_pinned_memory=True,
        use_depth=True,
    ):
        super().__init__()
        self.action_space_config = action_space_config

        # Depth usage flag - when False, depth is not used and state is smaller
        self.use_depth = use_depth
        self.depth_dim = depth_dim if use_depth else 0

        self.rgb_encoder = build_rgb_encoder(backbone, rgb_dim)
        if self.use_depth:
            self.depth_encoder = SimpleDepthCNN(output_dim=depth_dim)
        else:
            self.depth_encoder = None

        # Optimization flags
        self.freeze_backbone = freeze_backbone
        self.freeze_rgb_backbone = freeze_backbone if freeze_rgb_backbone is None else freeze_rgb_backbone
        self.freeze_depth_backbone = freeze_backbone if freeze_depth_backbone is None else freeze_depth_backbone
        self.use_cuda_streams = use_cuda_streams
        self.use_pinned_memory = use_pinned_memory

        # Freeze backbones if requested
        if self.freeze_rgb_backbone or self.freeze_depth_backbone:
            self._freeze_backbone()

        # CUDA streams for parallel processing (lazy initialization to avoid OOM at init)
        self.rgb_stream = None
        self.depth_stream = None
        self._streams_initialized = False

        # Action Embedding: Support both legacy (single action) and multi-head (direction + length)
        if "num_actions" in action_space_config:
            # Legacy action space: Single embedding for discrete actions
            self.action_emb = LegacyActionEmbedding(num_actions=action_space_config["num_actions"], emb_dim=action_dim)
        else:
            # Multi-head action space: Separate embeddings for direction and length
            self.action_emb = MultiHeadActionEmbedding(
                num_directions=action_space_config["num_directions"], num_lengths=action_space_config["num_lengths"], emb_dim=action_dim
            )

        self.sg_dim = sg_dim
        self.use_transformer = use_transformer

        if use_transformer:
            SGEncoderClass = SceneGraphTransformerEncoder
            self.lssg_encoder = SGEncoderClass(input_dim=sg_dim, hidden_dim=sg_dim)
            self.gssg_encoder = SGEncoderClass(input_dim=sg_dim, hidden_dim=sg_dim)
        else:
            SGEncoderClass = SceneGraphLSTMEncoder
            self.lssg_encoder = SGEncoderClass(input_dim=sg_dim, hidden_dim=sg_dim, num_layers=lssg_layers)
            self.gssg_encoder = SGEncoderClass(input_dim=sg_dim, hidden_dim=sg_dim, num_layers=lssg_layers)

        self.node_att_vector = nn.Parameter(torch.randn(int(sg_dim / 2)))
        self.edge_att_vector = nn.Parameter(torch.randn(int(sg_dim / 2)))

        self.object_to_idx = {}
        self.relation_to_idx = {}

        self.max_object_types = max_object_types
        self.max_relation_types = max_relation_types
        self.obj_type_embedding = nn.Embedding(max_object_types, obj_embedding_dim)
        self.rel_type_embedding = nn.Embedding(max_relation_types, rel_embedding_dim)

        self.mapping_path = mapping_path
        if mapping_path and os.path.exists(os.path.join(mapping_path, "object_types.json")):
            self.load_mappings(mapping_path)

        relation_types = list(self.relation_to_idx.keys())
        graph_encoder_in_channels = 4 + obj_embedding_dim
        if relation_types:
            self.graph_feature_extractor = NodeEdgeHGTEncoder(
                in_channels=graph_encoder_in_channels,
                edge_in_channels=rel_embedding_dim,
                hidden_channels=graph_hidden,
                out_channels=int(sg_dim / 2),
                relation_types=relation_types,
                num_heads=graph_heads,
                num_layers=graph_layers,
            )
        else:
            self.graph_feature_extractor = None

        self.object_count = len(self.object_to_idx)
        self.relation_count = len(self.relation_to_idx)

        # Graph Data Cache: Cache PyG Data objects to speed up get_graph_features_batched
        # Key: hash of scene graph, Value: PyG HeteroData object
        self._graph_data_cache = {}
        self._cache_hits = 0
        self._cache_misses = 0
        self._max_cache_size = 5000  # Limit cache size to prevent memory issues

        # Stagnation Detection: Encodes GSSG change signal to help agent learn when to stop
        # Input: [current_change, smoothed_change, stagnation_signal] (3 values)
        # Output: 32-dim embedding
        self.stagnation_dim = 32
        self.stagnation_encoder = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, self.stagnation_dim))
        self.stagnation_smoothing_alpha = 0.7  # EMA smoothing factor

    def _freeze_backbone(self):
        """Freeze RGB and Depth encoder parameters to save memory and computation."""
        if self.freeze_rgb_backbone:
            for param in self.rgb_encoder.parameters():
                param.requires_grad = False
            self.rgb_encoder.eval()
        if self.freeze_depth_backbone:
            for param in self.depth_encoder.parameters():
                param.requires_grad = False
            self.depth_encoder.eval()

    def train(self, mode=True):
        """Override train() to keep frozen backbones in eval mode."""
        super().train(mode)
        if self.freeze_rgb_backbone:
            self.rgb_encoder.eval()
        if self.freeze_depth_backbone:
            self.depth_encoder.eval()
        return self

    @staticmethod
    def preprocess_rgb(rgb_list):
        """
        Accepts: list of np.ndarray, PIL.Image, or torch.Tensor
        Returns: FloatTensor [N, 3, H, W]
        """
        from PIL import Image

        if not rgb_list:
            return torch.empty((0, 3, 224, 224), dtype=torch.float32)

        valid_imgs = []
        valid_indices = []

        for idx, rgb in enumerate(rgb_list):
            if rgb is None or (isinstance(rgb, int) and rgb == 0):
                continue

            if isinstance(rgb, torch.Tensor):
                img = rgb.detach().cpu()
                if img.ndim == 2:
                    img = img.unsqueeze(0)
                elif img.ndim == 3 and img.shape[0] not in (1, 3) and img.shape[-1] in (1, 3):
                    img = img.permute(2, 0, 1)
                if img.ndim == 3 and img.shape[0] == 1:
                    img = img.repeat(3, 1, 1)
                if img.ndim == 3 and img.shape[0] == 3:
                    valid_imgs.append(img)
                    valid_indices.append(idx)
                continue

            if isinstance(rgb, np.ndarray):
                arr = np.ascontiguousarray(rgb)
            elif isinstance(rgb, Image.Image):
                arr = np.array(rgb)
            elif hasattr(rgb, "copy"):
                arr = np.array(rgb).copy()
            else:
                arr = np.array(rgb)

            if arr.ndim == 2:
                arr = np.stack([arr, arr, arr], axis=-1)
            elif arr.ndim == 3 and arr.shape[-1] == 1:
                arr = np.repeat(arr, 3, axis=-1)

            if arr.ndim != 3 or arr.shape[-1] != 3:
                continue

            img = torch.from_numpy(np.ascontiguousarray(arr)).permute(2, 0, 1)
            valid_imgs.append(img)
            valid_indices.append(idx)

        out = torch.zeros((len(rgb_list), 3, 224, 224), dtype=torch.float32)
        if not valid_imgs:
            return out

        shapes = {tuple(img.shape) for img in valid_imgs}
        if len(shapes) == 1:
            batch = torch.stack(valid_imgs, dim=0)
            batch = batch.float()
            if batch.max().item() > 1.0:
                batch = batch / 255.0
            batch = F.interpolate(batch, size=(224, 224), mode="bilinear", align_corners=False)
        else:
            resized = []
            for img in valid_imgs:
                img = img.float()
                if img.max().item() > 1.0:
                    img = img / 255.0
                img = F.interpolate(img.unsqueeze(0), size=(224, 224), mode="bilinear", align_corners=False).squeeze(0)
                resized.append(img)
            batch = torch.stack(resized, dim=0)

        mean = batch.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = batch.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        batch = (batch - mean) / std

        out[valid_indices] = batch
        return out

    @staticmethod
    def preprocess_depth(depth_list):
        """
        Normalizes depth maps.
        Expects list of [H, W] numpy arrays or tensors.
        Returns: FloatTensor [N, 1, 224, 224].
        """
        if not depth_list:
            return torch.empty((0, 1, 224, 224), dtype=torch.float32)

        valid_depth = []
        valid_indices = []

        for idx, d in enumerate(depth_list):
            if d is None or (isinstance(d, int) and d == 0):
                continue

            if isinstance(d, torch.Tensor):
                depth = d.detach().cpu()
            else:
                depth = torch.from_numpy(np.ascontiguousarray(d))

            if depth.ndim == 2:
                depth = depth.unsqueeze(0)
            elif depth.ndim == 3 and depth.shape[0] != 1 and depth.shape[-1] == 1:
                depth = depth.permute(2, 0, 1)

            if depth.ndim == 3 and depth.shape[0] != 1:
                depth = depth[:1]

            if depth.ndim == 3 and depth.shape[0] == 1:
                valid_depth.append(depth)
                valid_indices.append(idx)

        out = torch.zeros((len(depth_list), 1, 224, 224), dtype=torch.float32)
        if not valid_depth:
            return out

        shapes = {tuple(dep.shape) for dep in valid_depth}
        if len(shapes) == 1:
            batch = torch.stack(valid_depth, dim=0).float()
            batch = torch.clamp(batch, 0, 10.0) / 10.0
            batch = F.interpolate(batch, size=(224, 224), mode="bilinear", align_corners=False)
        else:
            resized = []
            for dep in valid_depth:
                dep = dep.float()
                dep = torch.clamp(dep, 0, 10.0) / 10.0
                dep = F.interpolate(dep.unsqueeze(0), size=(224, 224), mode="bilinear", align_corners=False).squeeze(0)
                resized.append(dep)
            batch = torch.stack(resized, dim=0)

        out[valid_indices] = batch
        return out

    def preprocess_rgb_optimized(self, rgb_list, device):
        """Optimized RGB preprocessing with pinned memory for non-blocking transfers."""
        rgb_tensor = self.preprocess_rgb(rgb_list)
        if self.use_pinned_memory and device.type == "cuda":
            rgb_tensor = rgb_tensor.pin_memory()
            return rgb_tensor.to(device, non_blocking=True)
        return rgb_tensor.to(device)

    def preprocess_depth_optimized(self, depth_list, device):
        """Optimized depth preprocessing with pinned memory for non-blocking transfers."""
        depth_tensor = self.preprocess_depth(depth_list)
        if self.use_pinned_memory and device.type == "cuda":
            depth_tensor = depth_tensor.pin_memory()
            return depth_tensor.to(device, non_blocking=True)
        return depth_tensor.to(device)

    def _compute_scene_graph_hash(self, sg):
        """
        Compute a hash for a scene graph based on its structure.
        Returns None if scene graph is invalid.
        """
        if sg is None or (isinstance(sg, int) and sg == 0) or not sg.nodes:
            return None

        # Create a deterministic string representation of the graph
        # Include: node IDs, node names, node positions, edges
        node_str = "|".join(
            [
                f"{node_id}:{node.name}:{node.position[0]:.2f},{node.position[1]:.2f},{node.position[2]:.2f}"
                for node_id, node in sorted(sg.nodes.items())
            ]
        )

        edge_str = "|".join(
            [f"{edge.source}-{edge.relation}-{edge.target}" for edge in sorted(sg.edges, key=lambda e: (e.source, e.target, e.relation))]
        )

        combined_str = f"nodes:{node_str}||edges:{edge_str}"
        return hashlib.md5(combined_str.encode()).hexdigest()

    def create_hgt_data(self, sg, device):
        if sg is None or (isinstance(sg, int) and sg == 0) or not sg.nodes or not self.relation_to_idx:
            return None

        data = HeteroData()
        node_id_map = {node_id: i for i, node_id in enumerate(sg.nodes)}

        node_positions = []
        object_type_indices = []
        visibilities = []
        for node in sg.nodes.values():
            node_positions.append(node.position)
            obj_type_idx = self.object_to_idx.setdefault(node.name, len(self.object_to_idx))
            object_type_indices.append(obj_type_idx)
            visibilities.append(getattr(node, "visibility", 1.0))

        # Debug: Check for out-of-bounds indices
        if object_type_indices:
            max_idx = max(object_type_indices)
            if max_idx >= self.max_object_types:
                print(f"[ERROR] Object index {max_idx} >= max_object_types {self.max_object_types}")
                print(f"[ERROR] Total unique objects: {len(self.object_to_idx)}")
                print(f"[ERROR] Object mappings sample: {list(self.object_to_idx.items())[:10]}")
                raise ValueError(f"Object type index {max_idx} exceeds embedding size {self.max_object_types}")

        obj_indices_tensor = torch.tensor(object_type_indices, dtype=torch.long, device=device)
        obj_embeddings = self.obj_type_embedding(obj_indices_tensor)
        pos_tensor = torch.tensor(node_positions, dtype=torch.float32, device=device)
        vis_tensor = torch.tensor(visibilities, dtype=torch.float32, device=device).unsqueeze(1)
        x = torch.cat([pos_tensor, vis_tensor, obj_embeddings], dim=1)
        data["object"].x = x

        for rel_type in self.relation_to_idx:
            sources, targets, edge_attr_idx = [], [], []
            for edge in sg.edges:
                if edge.relation == rel_type:
                    if edge.source in node_id_map and edge.target in node_id_map:
                        sources.append(node_id_map[edge.source])
                        targets.append(node_id_map[edge.target])
                        idx = self.relation_to_idx[rel_type]
                        edge_attr_idx.append(idx)
            edge_type = ("object", rel_type, "object")
            if sources:
                data[edge_type].edge_index = torch.tensor([sources, targets], dtype=torch.long, device=device)
                edge_attr_tensor = self.rel_type_embedding(torch.tensor(edge_attr_idx, dtype=torch.long, device=device))
                data[edge_type].edge_attr = edge_attr_tensor
            else:
                data[edge_type].edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
                data[edge_type].edge_attr = torch.empty((0, self.rel_type_embedding.embedding_dim), device=device)
        return data

    def attention_pooling(self, features, att_vector):
        if features.size(0) == 0:
            return torch.zeros_like(att_vector)
        scores = torch.matmul(features, att_vector)
        weights = torch.softmax(scores, dim=0)
        pooled = torch.sum(weights.unsqueeze(-1) * features, dim=0)
        return pooled

    def get_graph_features(self, sg_list: list):
        if self.graph_feature_extractor is None:
            device = next(self.parameters()).device
            return torch.zeros((0, self.sg_dim), device=device), torch.tensor([], dtype=torch.long, device=device)

        device = next(self.parameters()).device
        data_list, valid_indices = [], []
        for i, sg in enumerate(sg_list):
            data = self.create_hgt_data(sg, device)
            if data is not None:
                data_list.append(data)
                valid_indices.append(i)

        if not data_list:
            pooled_features = torch.zeros((0, self.sg_dim), device=device)
            return pooled_features, torch.tensor([], dtype=torch.long, device=device)

        graph_embeds = []
        for d in data_list:
            node_out, edge_out = self.graph_feature_extractor(d)
            node_pooled = (
                self.attention_pooling(node_out, self.node_att_vector) if node_out.shape[0] > 0 else torch.zeros_like(self.node_att_vector)
            )
            edge_pooled = (
                self.attention_pooling(edge_out, self.edge_att_vector) if edge_out.shape[0] > 0 else torch.zeros_like(self.edge_att_vector)
            )
            graph_embeds.append(torch.cat([node_pooled, edge_pooled], dim=-1))
        pooled_features = torch.stack(graph_embeds, dim=0)
        return pooled_features, torch.tensor(valid_indices, dtype=torch.long, device=device)

    def get_graph_features_batched(self, sg_list: list):
        """
        Batched version of get_graph_features that processes all graphs in one forward pass.
        Uses caching to avoid recomputing embeddings for identical scene graphs.
        More efficient than processing graphs individually.
        """
        if self.graph_feature_extractor is None:
            device = next(self.parameters()).device
            return torch.zeros((0, self.sg_dim), device=device), torch.tensor([], dtype=torch.long, device=device)

        device = next(self.parameters()).device
        data_list, valid_indices, cache_keys = [], [], []
        graph_embeds_map = {}  # Map from list index to embedding

        # Check cache for each scene graph
        for i, sg in enumerate(sg_list):
            sg_hash = self._compute_scene_graph_hash(sg)

            if sg_hash is None:
                continue

            # Check if this graph embedding is already cached
            if sg_hash in self._graph_data_cache:
                # Cache hit - reuse cached embedding
                graph_embeds_map[i] = self._graph_data_cache[sg_hash]
                valid_indices.append(i)
                self._cache_hits += 1
            else:
                # Cache miss - need to compute
                data = self.create_hgt_data(sg, device)
                if data is not None:
                    data_list.append(data)
                    cache_keys.append((i, sg_hash))  # Store index and hash for later caching
                    valid_indices.append(i)
                    self._cache_misses += 1

        # Compute embeddings for cache misses
        if data_list:
            for idx, data in enumerate(data_list):
                node_out, edge_out = self.graph_feature_extractor(data)
                node_pooled = (
                    self.attention_pooling(node_out, self.node_att_vector)
                    if node_out.shape[0] > 0
                    else torch.zeros_like(self.node_att_vector)
                )
                edge_pooled = (
                    self.attention_pooling(edge_out, self.edge_att_vector)
                    if edge_out.shape[0] > 0
                    else torch.zeros_like(self.edge_att_vector)
                )
                embedding = torch.cat([node_pooled, edge_pooled], dim=-1)

                # Store in map and cache
                list_idx, sg_hash = cache_keys[idx]
                graph_embeds_map[list_idx] = embedding

                # Add to cache (with LRU eviction if needed)
                if len(self._graph_data_cache) >= self._max_cache_size:
                    # Simple FIFO eviction - remove first item
                    self._graph_data_cache.pop(next(iter(self._graph_data_cache)))
                self._graph_data_cache[sg_hash] = embedding.detach()

        if not valid_indices:
            pooled_features = torch.zeros((0, self.sg_dim), device=device)
            return pooled_features, torch.tensor([], dtype=torch.long, device=device)

        # Stack embeddings in the correct order
        pooled_features = torch.stack([graph_embeds_map[i] for i in valid_indices], dim=0)

        return pooled_features, torch.tensor(valid_indices, dtype=torch.long, device=device)

    def obs_to_dict(self, obs):
        """
        Converts an Observation or a list of Observations to a dict for feature extraction.
        Handles both single observations and temporal sequences.
        """
        if isinstance(obs, list):
            rgb = [o.state[0] for o in obs]
            depth = [o.state[1] for o in obs]
            lssg = [o.state[2] for o in obs]
            gssg = [o.state[3] for o in obs]
            agent_pos = [o.info.get("agent_pos", None) for o in obs]
            return {"rgb": [rgb], "depth": [depth], "lssg": [lssg], "gssg": [gssg], "agent_pos": [agent_pos]}
        else:
            # Single Observation
            return {
                "rgb": [[obs.state[0]]],
                "depth": [[obs.state[1]]],
                "lssg": [[obs.state[2]]],
                "gssg": [[obs.state[3]]],
                "agent_pos": [[obs.info.get("agent_pos", None)]],
            }

    def forward(self, obs, last_action, lssg_hidden=None, gssg_hidden=None, stagnation_state=None):
        """
        Forward pass for a single observation or a sequence.

        Returns:
            out: Feature tensor
            lssg_hidden: Updated LSSG hidden state
            gssg_hidden: Updated GSSG hidden state
            stagnation_state: Updated stagnation tracking state
        """
        batch_dict = self.obs_to_dict(obs) if not isinstance(obs, dict) else obs
        device = next(self.parameters()).device

        # Normalize last_action to tensor of shape [B, T, 2]
        if isinstance(last_action, list):
            # If list of tuples/lists
            last_action = torch.tensor(last_action, dtype=torch.long, device=device)
        elif isinstance(last_action, torch.Tensor):
            last_action = last_action.to(device)

        # Ensure dimensions [B, T, 2]
        if last_action.ndim == 1:  # [2] -> [1, 1, 2]
            last_action = last_action.view(1, 1, -1)
        elif last_action.ndim == 2:  # [T, 2] -> [1, T, 2] or [B, 2]
            if batch_dict["rgb"] and len(batch_dict["rgb"]) == 1:
                last_action = last_action.unsqueeze(0)
            else:
                last_action = last_action.unsqueeze(1)

        return self.forward_seq(
            batch_dict, last_action, lssg_hidden=lssg_hidden, gssg_hidden=gssg_hidden, stagnation_state=stagnation_state
        )

    def forward_seq(
        self,
        batch_dict,
        last_actions,
        pad_mask=None,
        lssg_hidden=None,
        gssg_hidden=None,
        stagnation_state=None,
        return_stagnation_signal=False,
    ):
        """
        Forward pass for sequence of observations.

        Args:
            stagnation_state: Optional dict with keys:
                - 'prev_gssg': Previous GSSG embedding [B, sg_dim]
                - 'smoothed_change': EMA of GSSG change [B, 1]
                - 'steps_since_discovery': Steps since significant change [B, 1]

        Returns:
            out: Feature tensor [B, T, feature_dim]
            lssg_hidden: Updated LSSG LSTM hidden state
            gssg_hidden: Updated GSSG LSTM hidden state
            new_stagnation_state: Updated stagnation tracking state
        """
        device = next(self.parameters()).device

        # Normalize VecEnv batches: if rgb/depth arrive as stacked tensors/arrays [B, H, W, C]
        # instead of nested [B, T, ...], wrap once so B=1 and T=frames.
        if isinstance(batch_dict["rgb"], (torch.Tensor, np.ndarray)):
            batch_dict = {k: [v] for k, v in batch_dict.items()}
        elif batch_dict["rgb"] and isinstance(batch_dict["rgb"][0], (torch.Tensor, np.ndarray)) and batch_dict["rgb"][0].ndim == 4:
            batch_dict = {k: [v[0]] if isinstance(v, list) and v else [v] for k, v in batch_dict.items()}

        B, T = len(batch_dict["rgb"]), len(batch_dict["rgb"][0])
        total_steps = B * T

        # 1. Visual Features (RGB + Depth) with optimizations
        rgb_flat = [im for seq in batch_dict["rgb"] for im in seq]
        if self.use_depth:
            depth_flat = [im for seq in batch_dict["depth"] for im in seq]

        # Lazy initialization of CUDA streams (to avoid OOM at model init)
        if self.use_cuda_streams and not self._streams_initialized and device.type == "cuda":
            try:
                self.rgb_stream = torch.cuda.Stream()
                if self.use_depth:
                    self.depth_stream = torch.cuda.Stream()
                self._streams_initialized = True
            except RuntimeError:
                # If stream creation fails, disable CUDA streams
                self.use_cuda_streams = False
                self._streams_initialized = True

        # Use CUDA streams for parallel RGB/Depth processing if available
        if self.use_cuda_streams and self.rgb_stream is not None:
            # RGB processing in parallel stream
            with torch.cuda.stream(self.rgb_stream):
                rgb_tensor = self.preprocess_rgb_optimized(rgb_flat, device)
                # Use no_grad for frozen backbone
                if self.freeze_rgb_backbone:
                    with torch.no_grad():
                        rgb_feat = self.rgb_encoder(rgb_tensor)
                else:
                    rgb_feat = self.rgb_encoder(rgb_tensor)

            # Depth processing in parallel stream (only if depth is used)
            if self.use_depth:
                with torch.cuda.stream(self.depth_stream):
                    depth_tensor = self.preprocess_depth_optimized(depth_flat, device)
                    # Use no_grad for frozen backbone
                    if self.freeze_depth_backbone:
                        with torch.no_grad():
                            depth_feat = self.depth_encoder(depth_tensor)
                    else:
                        depth_feat = self.depth_encoder(depth_tensor)

            # Synchronize streams before concatenation
            torch.cuda.synchronize()
        else:
            # Sequential processing without CUDA streams
            rgb_tensor = self.preprocess_rgb_optimized(rgb_flat, device)

            if self.freeze_rgb_backbone:
                with torch.no_grad():
                    rgb_feat = self.rgb_encoder(rgb_tensor)
            else:
                rgb_feat = self.rgb_encoder(rgb_tensor)

            # Depth processing (only if depth is used)
            if self.use_depth:
                depth_tensor = self.preprocess_depth_optimized(depth_flat, device)
                if self.freeze_depth_backbone:
                    with torch.no_grad():
                        depth_feat = self.depth_encoder(depth_tensor)
                else:
                    depth_feat = self.depth_encoder(depth_tensor)

        # 2. Action Embeddings
        # last_actions is [B, T, 2], reshape to [B*T, 2]
        act_flat = last_actions.view(-1, 2).to(device)
        act_feat = self.action_emb(act_flat)

        # 3. Graph Features (batched for efficiency)
        lssg_flat = [sg for seq in batch_dict["lssg"] for sg in seq]
        gssg_flat = [sg for seq in batch_dict["gssg"] for sg in seq]

        lssg_embeds, lssg_valid_indices = self.get_graph_features_batched(lssg_flat)
        gssg_embeds, gssg_valid_indices = self.get_graph_features_batched(gssg_flat)

        lssg_feat_full = torch.zeros(total_steps, self.sg_dim, device=device)
        gssg_feat_full = torch.zeros(total_steps, self.sg_dim, device=device)

        lssg_feat_full[lssg_valid_indices] = lssg_embeds
        gssg_feat_full[gssg_valid_indices] = gssg_embeds

        # 4. Sequence Processing
        lssg_seq = lssg_feat_full.view(B, T, -1)
        gssg_seq = gssg_feat_full.view(B, T, -1)

        if self.use_transformer:
            if "lssg_mask" in batch_dict:
                lssg_mask = torch.tensor(batch_dict["lssg_mask"], dtype=torch.bool, device=device)
                gssg_mask = torch.tensor(batch_dict["gssg_mask"], dtype=torch.bool, device=device)
                lssg_feat = self.lssg_encoder(lssg_seq, pad_mask=~lssg_mask)
                gssg_feat = self.gssg_encoder(gssg_seq, pad_mask=~gssg_mask)
            else:
                lssg_feat = self.lssg_encoder(lssg_seq, pad_mask=pad_mask)
                gssg_feat = self.gssg_encoder(gssg_seq, pad_mask=pad_mask)
        else:
            lssg_feat, lssg_hidden = self.lssg_encoder(lssg_seq, lssg_hidden)
            gssg_feat, gssg_hidden = self.gssg_encoder(gssg_seq, gssg_hidden)

        lssg_feat = lssg_feat.reshape(total_steps, -1)
        gssg_feat = gssg_feat.reshape(total_steps, -1)

        # 5. Stagnation Detection
        # Compute GSSG change to help agent learn when exploration has stagnated
        gssg_feat_seq = gssg_feat.view(B, T, -1)  # [B, T, sg_dim]

        # Initialize or unpack stagnation state
        if stagnation_state is not None:
            prev_gssg = stagnation_state.get("prev_gssg")  # [B, sg_dim]
            smoothed_change = stagnation_state.get("smoothed_change")  # [B, 1]
            steps_since_discovery = stagnation_state.get("steps_since_discovery")  # [B, 1]
        else:
            prev_gssg = None
            smoothed_change = None
            steps_since_discovery = None

        # Compute stagnation features for each timestep
        stagnation_inputs = []
        alpha = self.stagnation_smoothing_alpha
        change_threshold = 0.1  # Threshold for "significant" discovery

        for t in range(T):
            current_gssg = gssg_feat_seq[:, t, :]  # [B, sg_dim]

            if prev_gssg is not None:
                # Compute L2 norm of change
                current_change = torch.norm(current_gssg - prev_gssg, dim=-1, keepdim=True)  # [B, 1]
                # Normalize by sg_dim for scale invariance
                current_change = current_change / (self.sg_dim**0.5)
            else:
                # First step: maximum change (everything is new)
                current_change = torch.ones(B, 1, device=device)

            # Update smoothed change (EMA)
            if smoothed_change is not None:
                smoothed_change = alpha * smoothed_change + (1 - alpha) * current_change
            else:
                smoothed_change = current_change.clone()

            # Update steps since discovery
            if steps_since_discovery is not None:
                # Reset if significant discovery, otherwise increment
                is_discovery = (current_change > change_threshold).float()
                steps_since_discovery = steps_since_discovery * (1 - is_discovery) + 1
            else:
                steps_since_discovery = torch.ones(B, 1, device=device)

            # Compute stagnation signal (normalized to ~[0, 1])
            # Increased divisor from 5.0 to 10.0 to delay signal and encourage longer exploration
            stagnation_signal = torch.tanh(steps_since_discovery / 10.0)

            # Stack inputs for stagnation encoder: [current_change, smoothed_change, stagnation_signal]
            stag_input = torch.cat([current_change, smoothed_change, stagnation_signal], dim=-1)  # [B, 3]
            stagnation_inputs.append(stag_input)

            # Update prev_gssg for next timestep
            prev_gssg = current_gssg.detach()

        # Stack all timesteps and encode
        stagnation_inputs = torch.stack(stagnation_inputs, dim=1)  # [B, T, 3]
        stagnation_signal_seq = stagnation_inputs[:, :, 2].detach()  # [B, T]
        stagnation_inputs_flat = stagnation_inputs.view(B * T, 3)  # [B*T, 3]
        stagnation_embedding = self.stagnation_encoder(stagnation_inputs_flat)  # [B*T, stagnation_dim]

        # Prepare new stagnation state for next call
        new_stagnation_state = {
            "prev_gssg": prev_gssg,  # [B, sg_dim]
            "smoothed_change": smoothed_change,  # [B, 1]
            "steps_since_discovery": steps_since_discovery,  # [B, 1]
        }

        # 6. Concatenate all features (exclude depth if not used)
        if self.use_depth:
            feats = [act_feat, rgb_feat, depth_feat, lssg_feat, gssg_feat, stagnation_embedding]
        else:
            feats = [act_feat, rgb_feat, lssg_feat, gssg_feat, stagnation_embedding]
        out = torch.cat(feats, dim=-1)
        if return_stagnation_signal:
            return out.view(B, T, -1), lssg_hidden, gssg_hidden, new_stagnation_state, stagnation_signal_seq
        return out.view(B, T, -1), lssg_hidden, gssg_hidden, new_stagnation_state

    def precompute_stateless(self, batch_dict, last_actions):
        """
        Precompute stateless per-step features (RGB, Depth, Action, Graph) for reuse.
        """
        device = next(self.parameters()).device

        if isinstance(batch_dict["rgb"], (torch.Tensor, np.ndarray)):
            batch_dict = {k: [v] for k, v in batch_dict.items()}
        elif batch_dict["rgb"] and isinstance(batch_dict["rgb"][0], (torch.Tensor, np.ndarray)) and batch_dict["rgb"][0].ndim == 4:
            batch_dict = {k: [v[0]] if isinstance(v, list) and v else [v] for k, v in batch_dict.items()}

        B, T = len(batch_dict["rgb"]), len(batch_dict["rgb"][0])
        total_steps = B * T

        rgb_flat = [im for seq in batch_dict["rgb"] for im in seq]
        if self.use_depth:
            depth_flat = [im for seq in batch_dict["depth"] for im in seq]

        if self.use_cuda_streams and not self._streams_initialized and device.type == "cuda":
            try:
                self.rgb_stream = torch.cuda.Stream()
                if self.use_depth:
                    self.depth_stream = torch.cuda.Stream()
                self._streams_initialized = True
            except RuntimeError:
                self.use_cuda_streams = False
                self._streams_initialized = True

        if self.use_cuda_streams and self.rgb_stream is not None:
            with torch.cuda.stream(self.rgb_stream):
                rgb_tensor = self.preprocess_rgb_optimized(rgb_flat, device)
                if self.freeze_rgb_backbone:
                    with torch.no_grad():
                        rgb_feat = self.rgb_encoder(rgb_tensor)
                else:
                    rgb_feat = self.rgb_encoder(rgb_tensor)

            if self.use_depth:
                with torch.cuda.stream(self.depth_stream):
                    depth_tensor = self.preprocess_depth_optimized(depth_flat, device)
                    if self.freeze_depth_backbone:
                        with torch.no_grad():
                            depth_feat = self.depth_encoder(depth_tensor)
                    else:
                        depth_feat = self.depth_encoder(depth_tensor)

            torch.cuda.synchronize()
        else:
            rgb_tensor = self.preprocess_rgb_optimized(rgb_flat, device)

            if self.freeze_rgb_backbone:
                with torch.no_grad():
                    rgb_feat = self.rgb_encoder(rgb_tensor)
            else:
                rgb_feat = self.rgb_encoder(rgb_tensor)

            if self.use_depth:
                depth_tensor = self.preprocess_depth_optimized(depth_flat, device)
                if self.freeze_depth_backbone:
                    with torch.no_grad():
                        depth_feat = self.depth_encoder(depth_tensor)
                else:
                    depth_feat = self.depth_encoder(depth_tensor)

        if last_actions is None:
            raise ValueError("last_actions required for stateless feature precompute.")
        act_flat = last_actions.view(-1, 2).to(device)
        act_feat = self.action_emb(act_flat)

        lssg_flat = [sg for seq in batch_dict["lssg"] for sg in seq]
        gssg_flat = [sg for seq in batch_dict["gssg"] for sg in seq]

        lssg_embeds, lssg_valid_indices = self.get_graph_features_batched(lssg_flat)
        gssg_embeds, gssg_valid_indices = self.get_graph_features_batched(gssg_flat)

        lssg_feat_full = torch.zeros(total_steps, self.sg_dim, device=device)
        gssg_feat_full = torch.zeros(total_steps, self.sg_dim, device=device)
        lssg_feat_full[lssg_valid_indices] = lssg_embeds
        gssg_feat_full[gssg_valid_indices] = gssg_embeds

        result = {
            "rgb_feat": rgb_feat.view(B, T, -1),
            "act_feat": act_feat.view(B, T, -1),
            "lssg_feat": lssg_feat_full.view(B, T, -1),
            "gssg_feat": gssg_feat_full.view(B, T, -1),
            "B": B,
            "T": T,
        }
        if self.use_depth:
            result["depth_feat"] = depth_feat.view(B, T, -1)
        return result

    def forward_seq_from_stateless(
        self, stateless, pad_mask=None, lssg_hidden=None, gssg_hidden=None, stagnation_state=None, return_stagnation_signal=False
    ):
        """
        Forward pass using precomputed stateless features.
        """
        device = next(self.parameters()).device
        B = stateless["B"]
        T = stateless["T"]

        rgb_feat = stateless["rgb_feat"].reshape(B, T, -1)
        if self.use_depth:
            depth_feat = stateless["depth_feat"].reshape(B, T, -1)
        act_feat = stateless["act_feat"].reshape(B, T, -1)
        lssg_seq = stateless["lssg_feat"].reshape(B, T, -1)
        gssg_seq = stateless["gssg_feat"].reshape(B, T, -1)

        if self.use_transformer:
            lssg_feat = self.lssg_encoder(lssg_seq, pad_mask=pad_mask)
            gssg_feat = self.gssg_encoder(gssg_seq, pad_mask=pad_mask)
        else:
            lssg_feat, lssg_hidden = self.lssg_encoder(lssg_seq, lssg_hidden)
            gssg_feat, gssg_hidden = self.gssg_encoder(gssg_seq, gssg_hidden)

        lssg_feat_flat = lssg_feat.reshape(B * T, -1)
        gssg_feat_flat = gssg_feat.reshape(B * T, -1)

        gssg_feat_seq = gssg_feat.reshape(B, T, -1)

        if stagnation_state is not None:
            prev_gssg = stagnation_state.get("prev_gssg")
            smoothed_change = stagnation_state.get("smoothed_change")
            steps_since_discovery = stagnation_state.get("steps_since_discovery")
        else:
            prev_gssg = None
            smoothed_change = None
            steps_since_discovery = None

        stagnation_inputs = []
        alpha = self.stagnation_smoothing_alpha
        change_threshold = 0.1

        for t in range(T):
            current_gssg = gssg_feat_seq[:, t, :]

            if prev_gssg is not None:
                current_change = torch.norm(current_gssg - prev_gssg, dim=-1, keepdim=True)
                current_change = current_change / (self.sg_dim**0.5)
            else:
                current_change = torch.ones(B, 1, device=device)

            if smoothed_change is not None:
                smoothed_change = alpha * smoothed_change + (1 - alpha) * current_change
            else:
                smoothed_change = current_change.clone()

            if steps_since_discovery is not None:
                is_discovery = (current_change > change_threshold).float()
                steps_since_discovery = steps_since_discovery * (1 - is_discovery) + 1
            else:
                steps_since_discovery = torch.ones(B, 1, device=device)

            # Increased divisor from 5.0 to 10.0 to delay signal and encourage longer exploration
            stagnation_signal = torch.tanh(steps_since_discovery / 10.0)
            stag_input = torch.cat([current_change, smoothed_change, stagnation_signal], dim=-1)
            stagnation_inputs.append(stag_input)

            prev_gssg = current_gssg.detach()

        stagnation_inputs = torch.stack(stagnation_inputs, dim=1)
        stagnation_signal_seq = stagnation_inputs[:, :, 2].detach()
        stagnation_inputs_flat = stagnation_inputs.reshape(B * T, 3)
        stagnation_embedding = self.stagnation_encoder(stagnation_inputs_flat)

        new_stagnation_state = {"prev_gssg": prev_gssg, "smoothed_change": smoothed_change, "steps_since_discovery": steps_since_discovery}

        if self.use_depth:
            feats = [
                act_feat.reshape(B * T, -1),
                rgb_feat.reshape(B * T, -1),
                depth_feat.reshape(B * T, -1),
                lssg_feat_flat,
                gssg_feat_flat,
                stagnation_embedding,
            ]
        else:
            feats = [
                act_feat.reshape(B * T, -1),
                rgb_feat.reshape(B * T, -1),
                lssg_feat_flat,
                gssg_feat_flat,
                stagnation_embedding,
            ]
        out = torch.cat(feats, dim=-1)
        if return_stagnation_signal:
            return out.view(B, T, -1), lssg_hidden, gssg_hidden, new_stagnation_state, stagnation_signal_seq
        return out.view(B, T, -1), lssg_hidden, gssg_hidden, new_stagnation_state

    def save_mappings(self, path: str):
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "object_types.json"), "w") as f:
            json.dump(self.object_to_idx, f)
        with open(os.path.join(path, "relation_types.json"), "w") as f:
            json.dump(self.relation_to_idx, f)

    def load_mappings(self, path: str):
        object_types_file = os.path.join(path, "object_types.json")
        relation_types_file = os.path.join(path, "relation_types.json")
        if os.path.exists(object_types_file):
            with open(object_types_file, "r") as f:
                self.object_to_idx = json.load(f)
                self.object_to_idx = {k: int(v) for k, v in self.object_to_idx.items()}
        if os.path.exists(relation_types_file):
            with open(relation_types_file, "r") as f:
                self.relation_to_idx = json.load(f)
                self.relation_to_idx = {k: int(v) for k, v in self.relation_to_idx.items()}

    def save_model(self, path):
        os.makedirs(path, exist_ok=True)
        # Using simplified filename for brevity, adjust as needed
        filename = f"feature_encoder_{self.use_transformer}.pth"
        full_path = os.path.join(path, filename)
        torch.save(self.state_dict(), full_path)

    @classmethod
    def create_and_load_model(cls, model_path, mapping_path=None, device="cpu", **kwargs):
        # Requires known config during loading if filename parsing is removed/changed
        model = cls(mapping_path=mapping_path, **kwargs)
        model.load_weights(model_path, device)
        return model

    def load_weights(self, model_path, device="cpu"):
        state_dict = torch.load(model_path, map_location=device)
        self.load_state_dict(state_dict)
        self.to(device)


class LegacyActionEmbedding(nn.Module):
    """
    Embeds legacy discrete actions (16 actions = 8 directions × 2 lengths).
    For compatibility with multi-head code, accepts [N, 2] tensor but only uses first column.
    """

    def __init__(self, num_actions, emb_dim=32):
        super().__init__()
        # +1 for START token (-1 maps to index 0)
        self.action_emb = nn.Embedding(num_actions + 1, emb_dim)
        self.emb_dim = emb_dim
        self.num_actions = num_actions

    def forward(self, action_tuple):
        # action_tuple: [N, 2] but for legacy we only use first column (action index)
        # Second column is ignored (kept for compatibility with multi-head code)
        if action_tuple.dim() == 1:
            action_idx = action_tuple.clone()
        else:
            action_idx = action_tuple[:, 0].clone()

        # Handle -1 (start state) and -100 (padding) - map to 0
        action_idx[action_idx == -1] = 0
        action_idx[action_idx == -100] = 0

        # Shift indices by +1 to leave embedding 0 for padding
        return self.action_emb(action_idx + 1)


class MultiHeadActionEmbedding(nn.Module):
    """
    Embeds (direction, length) tuple.
    Summing embeddings to combine categorical features.
    """

    def __init__(self, num_directions, num_lengths, emb_dim=32):
        super().__init__()
        # +1 for START token (-1)
        self.dir_emb = nn.Embedding(num_directions + 1, emb_dim)
        self.len_emb = nn.Embedding(num_lengths + 1, emb_dim)
        self.emb_dim = emb_dim

    def forward(self, action_tuple):
        # action_tuple: [N, 2] -> (dir_idx, len_idx)
        dir_idx = action_tuple[:, 0].clone()
        len_idx = action_tuple[:, 1].clone()

        # Handle -1 (start state) and -100 (padding) mapping to 0
        dir_idx[dir_idx == -1] = 0
        dir_idx[dir_idx == -100] = 0
        len_idx[len_idx == -1] = 0
        len_idx[len_idx == -100] = 0

        # Shift indices by +1
        e_dir = self.dir_emb(dir_idx + 1)
        e_len = self.len_emb(len_idx + 1)

        return e_dir + e_len


class SimpleDepthCNN(nn.Module):
    """
    A lightweight CNN optimized for Depth maps.
    Significantly faster and smaller than ResNet18 (~50k params vs 11M).
    """

    def __init__(self, output_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            # Layer 1: Grobe Features (Kanten, Distanz-Gradienten)
            nn.Conv2d(1, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            # Layer 2: Mittlere Features
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            # Layer 3: Feine Features
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

        self.projection = nn.Linear(64, output_dim)

    def forward(self, x):
        # x shape: [B, 1, 224, 224]
        x = self.net(x)
        return self.projection(x)


class ResNetFeatureExtractor(nn.Module):
    def __init__(self, output_dim=512):
        super().__init__()
        resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.projection = nn.Linear(512, output_dim) if output_dim != 512 else nn.Identity()

    def forward(self, x):
        x = self.resnet(x)
        x = x.view(x.size(0), -1)
        return self.projection(x)


class EfficientNetV2SFeatureExtractor(nn.Module):
    def __init__(self, output_dim=512):
        super().__init__()
        model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)
        self.features = model.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        in_dim = model.classifier[1].in_features if hasattr(model, "classifier") else 1280
        self.projection = nn.Linear(in_dim, output_dim) if output_dim != in_dim else nn.Identity()

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.projection(x)


class InceptionNeXtTinyFeatureExtractor(nn.Module):
    def __init__(self, output_dim=512):
        super().__init__()
        self.model = timm.create_model("inception_next_tiny.sail_in1k", pretrained=True)
        if hasattr(self.model, "reset_classifier"):
            self.model.reset_classifier(num_classes=0, global_pool="avg")
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            feat = self.model(dummy)
            if feat.ndim > 2:
                feat = feat.mean(dim=(2, 3))
            in_dim = feat.shape[1]
        self.projection = nn.Linear(in_dim, output_dim) if output_dim != in_dim else nn.Identity()

    def forward(self, x):
        x = self.model(x)
        if x.ndim > 2:
            x = x.mean(dim=(2, 3))
        return self.projection(x)


def build_rgb_encoder(backbone: str, output_dim: int) -> nn.Module:
    name = (backbone or "resnet18").lower()
    if name in ["resnet", "resnet18", "rn18"]:
        return ResNetFeatureExtractor(output_dim)
    if name in ["efficientnet_v2_s", "efficientnetv2s", "effnetv2s"]:
        return EfficientNetV2SFeatureExtractor(output_dim)
    if name in ["inceptionnext_t", "inceptionnext-t", "inceptionnext", "inceptionnext_tiny"]:
        return InceptionNeXtTinyFeatureExtractor(output_dim)
    raise ValueError(f"Unknown backbone '{backbone}'.")


class NodeEdgeHGTEncoder(nn.Module):
    def __init__(self, in_channels, edge_in_channels, hidden_channels, out_channels, relation_types, num_heads=4, num_layers=2):
        super().__init__()
        # metadata for HGTConv (node and edge types)
        node_types = ["object"]
        edge_types = [("object", rel, "object") for rel in relation_types]
        metadata = (node_types, edge_types)
        self.layers = nn.ModuleList()
        self.layers.append(HGTConv(in_channels=in_channels, out_channels=hidden_channels, metadata=metadata, heads=num_heads))
        for _ in range(num_layers - 1):
            self.layers.append(HGTConv(in_channels=hidden_channels, out_channels=hidden_channels, metadata=metadata, heads=num_heads))
        self.fc = nn.Linear(hidden_channels, out_channels)
        self.edge_mlp = nn.Sequential(nn.Linear(edge_in_channels, hidden_channels), nn.ReLU(), nn.Linear(hidden_channels, out_channels))

    def forward(self, data: "HeteroData"):
        x_dict, edge_index_dict, edge_attr_dict = data.x_dict, data.edge_index_dict, data.edge_attr_dict

        for layer in self.layers:
            x_dict = layer(x_dict, edge_index_dict)

        node_embeds = self.fc(x_dict["object"])  # [num_nodes, out_channels]

        edge_embeds = []
        for rel in edge_attr_dict:
            edge_attr = edge_attr_dict[rel]  # [num_edges_of_this_type, edge_in_channels]
            if edge_attr.numel() > 0:
                edge_embeds.append(self.edge_mlp(edge_attr))
        if edge_embeds:
            edge_embeds = torch.cat(edge_embeds, dim=0)  # [total_num_edges, out_channels]
        else:
            edge_embeds = torch.zeros((0, self.fc.out_features), device=node_embeds.device)

        return node_embeds, edge_embeds


class SceneGraphLSTMEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, num_layers=num_layers)

    def forward(self, x_seq, hidden=None):
        if x_seq.dim() == 2:
            x_seq = x_seq.unsqueeze(1)
        elif x_seq.dim() != 3:
            raise ValueError(f"Expected input [B, T, D] or [B, D], got {x_seq.shape}")
        output, (hn, cn) = self.lstm(x_seq, hidden)
        return output, (hn, cn)


class SceneGraphTransformerEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2, num_heads=4, dropout=0.1, max_len=256):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim, max_len=max_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x_seq, pad_mask=None):
        x_seq = self.input_proj(x_seq)
        x_seq = self.pos_encoder(x_seq)
        if pad_mask is not None:
            return self.transformer(x_seq, src_key_padding_mask=pad_mask)
        else:
            return self.transformer(x_seq)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=256):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]
