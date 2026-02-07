import json
import os
import warnings
from abc import ABC, abstractmethod
from collections import defaultdict
from copy import deepcopy

import numpy as np

from components.graph.global_graph import GlobalSceneGraph
from components.graph.gt_graph import GTGraph
from components.graph.local_graph_builder import LocalSceneGraphBuilder


class AbstractThorEnv(ABC):
    """
    Abstract base class for AI2-THOR environments.
    Bundles common functionality for both online (ThorEnv) and offline (PrecomputedThorEnv) variants.
    """

    # Class-level caches
    _gt_graph_cache = {}
    _viewpoint_cache = {}

    def __init__(
        self,
        rho=0.02,
        scene_number=None,
        grid_size=0.25,
        max_actions=60,
        render=False,
        curriculum_stage=1,
        use_legacy_actions=False,
        action_space_mode="multi_head",
        stop_stagnation_steps=5,
        stop_stagnation_bonus=0.02,
    ):
        self.rho = rho
        self.grid_size = grid_size
        self.max_actions = max_actions
        self.render = render
        self.scene_number = 1 if scene_number is None else scene_number
        self.use_legacy_actions = use_legacy_actions
        self.action_space_mode = action_space_mode

        # --- Action Space Definition ---
        if self.use_legacy_actions:
            # Legacy action space: 16 discrete actions (identical to Curriculum Stage 1)
            # 8 directions (45° steps) × 2 lengths (0.0m, 0.3m)
            # This matches the complexity of curriculum stage 1 but uses single-head architecture
            self.action_angles = [0, 45, 90, 135, 180, 225, 270, 315]  # 8 directions
            self.action_lengths = [0.0, 0.3]  # 0.0m = STOP/rotation, 0.3m = movement

            # Generate all 16 combinations as discrete actions
            # Format: (angle, distance) tuples
            self.legacy_actions = [(angle, length) for angle in self.action_angles for length in self.action_lengths]
            self.num_actions = len(self.legacy_actions)  # 16

            # STOP action: (0°, 0.0m) is first in list (index 0)
            self.stop_index = 0

            # For compatibility with multi-head code
            self.num_directions = 1
            self.num_lengths = self.num_actions
        elif action_space_mode == "single_head_large":
            # Single-Head Large: 504 discrete actions (24 directions × 21 lengths)
            # Flat action space: flat_idx = direction_idx * 21 + length_idx
            self.action_angles = list(range(0, 360, 15))  # 24 directions
            self.action_lengths = [round(x * 0.1, 1) for x in range(0, 21)]  # 21 lengths (0.0 - 2.0)
            self.num_actions = 24 * 21  # 504
            self.num_directions = 24  # For compatibility
            self.num_lengths = 21
            self.legacy_actions = None
        else:
            # Multi-head action space: 24 directions × 21 lengths = 504 actions
            # Policy is always initialized with full action space
            self.action_angles = list(range(0, 360, 15))  # 24 directions
            # Include 0.0m as STOP action (agent can choose not to move)
            self.action_lengths = [round(x * 0.1, 1) for x in range(0, 21)]  # 21 lengths (0.0 - 2.0)
            self.num_directions = len(self.action_angles)
            self.num_lengths = len(self.action_lengths)

            # For compatibility
            self.num_actions = None
            self.legacy_actions = None

        # --- Curriculum Learning (only for multi-head and single-head-large) ---
        if not self.use_legacy_actions:
            self.curriculum_stage = curriculum_stage
            if action_space_mode == "single_head_large":
                self._setup_curriculum_masks_flat()
            else:
                self._setup_curriculum_masks()
        else:
            self.curriculum_stage = None

        # --- Shared State ---
        self.builder = LocalSceneGraphBuilder()
        self.global_sg = GlobalSceneGraph()
        self.state = None
        self.gt_graph = None
        self.viewpoints = defaultdict(set)
        self.last_score = 0.0
        self.step_count = 0
        self._last_num_viewpoints = 0  # reset exploration bonus tracker
        self._steps_since_discovery = 0
        self.stop_stagnation_steps = stop_stagnation_steps
        self.stop_stagnation_bonus = stop_stagnation_bonus
        self._time_penalty_total = 0.0
        self._time_penalty_step_count = 0

        # --- Episode Metrics ---
        self.total_path_length = 0.0
        self.visited_positions = set()
        self.reachable_positions = None
        self.exploration_coverage = 0.0

        # --- Episode Statistics ---
        self.num_collisions = 0
        self.num_pure_rotations = 0

    # ============================================================================
    # ACTION SPACE & CURRICULUM
    # ============================================================================

    def _setup_curriculum_masks(self):
        """
        Setup action masks based on curriculum stage.
        Policy is always initialized with full action space (24 dirs × 21 lengths),
        but we mask out invalid actions during sampling based on curriculum stage.

        STOP action: Only (0°, 0.0m) - both indices must be 0
        Pure rotation: (90°/180°/270°, 0.0m) - agent rotates without moving
        Movement: (any angle, >0.0m) - agent rotates and moves

        Stage 1: 8 directions × 2 lengths = 16 valid actions
        Stage 2: 8 directions × 6 lengths = 48 valid actions
        Stage 3: 12 directions × 9 lengths = 108 valid actions
        Stage 4: 16 directions × 13 lengths = 208 valid actions
        Stage 5: 20 directions × 17 lengths = 340 valid actions
        Stage 6: 24 directions × 21 lengths = 504 valid actions
        """
        if self.curriculum_stage == 1:
            # Stage 1: 8 directions (45° steps) matching baseline complexity + short movement
            valid_angles = [0, 45, 90, 135, 180, 225, 270, 315]
            valid_lengths = [0.0, 0.3]  # 0.0m = STOP, 0.3m = basic move
        elif self.curriculum_stage == 2:
            # Stage 2: Same directions + more length options
            valid_angles = [0, 45, 90, 135, 180, 225, 270, 315]
            valid_lengths = [0.0, 0.3, 0.5, 1.0, 1.5, 2.0]
        elif self.curriculum_stage == 3:
            # Stage 3: More directions (30° steps) + more lengths
            valid_angles = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]  # 12 directions
            valid_lengths = [0.0, 0.3, 0.5, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0]  # 9 lengths
        elif self.curriculum_stage == 4:
            # Stage 4: 16 directions (include 15° offset angles) + finer length control
            valid_angles = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180, 195, 210, 225]  # 16 directions
            valid_lengths = [0.0, 0.2, 0.3, 0.5, 0.7, 0.8, 1.0, 1.2, 1.4, 1.5, 1.7, 1.8, 2.0]  # 13 lengths
        elif self.curriculum_stage == 5:
            # Stage 5: 20 directions (most angles) + more length granularity
            valid_angles = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180, 210, 225, 240, 255, 270, 285, 315]  # 20 directions
            valid_lengths = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]  # 17 lengths
        else:  # Stage 6 or higher
            # Stage 6: Full action space
            valid_angles = self.action_angles
            valid_lengths = self.action_lengths

        # Convert to indices with tolerance for float comparison
        self.valid_direction_indices = [i for i, angle in enumerate(self.action_angles) if angle in valid_angles]
        self.valid_length_indices = []
        for i, length in enumerate(self.action_lengths):
            # Use small tolerance for float comparison
            if any(abs(length - vl) < 1e-5 for vl in valid_lengths):
                self.valid_length_indices.append(i)

        # Debug: Verify indices are within bounds
        assert all(
            0 <= i < self.num_directions for i in self.valid_direction_indices
        ), f"Invalid direction indices: {self.valid_direction_indices}, max={self.num_directions}"
        assert all(
            0 <= i < self.num_lengths for i in self.valid_length_indices
        ), f"Invalid length indices: {self.valid_length_indices}, max={self.num_lengths}"
        assert (
            len(self.valid_length_indices) > 0
        ), f"No valid length indices found! valid_lengths={valid_lengths}, action_lengths={self.action_lengths}"

    def _setup_curriculum_masks_flat(self):
        """
        Setup flat action indices for Single-Head-Large curriculum.
        Maps (n_dirs, n_lens) curriculum stages to flat action indices.

        Stage-specific action space matches multi-head curriculum for fair comparison:
        Stage 1: 8 directions × 2 lengths = 16 valid actions
        Stage 2: 8 directions × 6 lengths = 48 valid actions
        Stage 3: 12 directions × 9 lengths = 108 valid actions
        Stage 4: 24 directions × 21 lengths = 504 valid actions (full space)
        """
        # Stage-specific angles and lengths (identical to _setup_curriculum_masks)
        if self.curriculum_stage == 1:
            valid_angles = [0, 45, 90, 135, 180, 225, 270, 315]  # 8
            valid_lengths = [0.0, 0.3]  # 2
        elif self.curriculum_stage == 2:
            valid_angles = [0, 45, 90, 135, 180, 225, 270, 315]  # 8
            valid_lengths = [0.0, 0.3, 0.5, 1.0, 1.5, 2.0]  # 6
        elif self.curriculum_stage == 3:
            valid_angles = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]  # 12
            valid_lengths = [0.0, 0.3, 0.5, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0]  # 9
        else:  # Stage 4+
            valid_angles = self.action_angles  # 24
            valid_lengths = self.action_lengths  # 21

        # Convert to indices
        valid_dir_indices = [i for i, angle in enumerate(self.action_angles) if angle in valid_angles]
        valid_len_indices = []
        for i, length in enumerate(self.action_lengths):
            if any(abs(length - vl) < 1e-5 for vl in valid_lengths):
                valid_len_indices.append(i)

        # Expose masks for compatibility with multi-head logging/hooks
        self.valid_direction_indices = valid_dir_indices
        self.valid_length_indices = valid_len_indices

        # Generate Flat-Indices (cartesian product)
        # flat_idx = dir_idx * 21 + len_idx
        self.valid_flat_action_indices = []
        for dir_idx in valid_dir_indices:
            for len_idx in valid_len_indices:
                flat_idx = dir_idx * 21 + len_idx
                self.valid_flat_action_indices.append(flat_idx)

        # Validation
        expected = len(valid_dir_indices) * len(valid_len_indices)
        assert len(self.valid_flat_action_indices) == expected, \
            f"Flat indices mismatch: {len(self.valid_flat_action_indices)} != {expected}"

    def set_curriculum_stage(self, stage):
        """
        Update the curriculum stage and reconfigure action masks.
        Should be called by the training runner based on episode count.
        """
        if stage != self.curriculum_stage:
            self.curriculum_stage = stage
            if self.action_space_mode == "single_head_large":
                self._setup_curriculum_masks_flat()
                # print(f"[CURRICULUM] Switched to Stage {stage}: {len(self.valid_flat_action_indices)} valid flat actions")
            else:
                self._setup_curriculum_masks()
                num_valid = len(self.valid_direction_indices) * len(self.valid_length_indices)
                # print(
                #     f"[CURRICULUM] Switched to Stage {stage}: {len(self.valid_direction_indices)} directions × {len(self.valid_length_indices)} lengths = {num_valid} valid actions"
                # )

    def get_action_space_dims(self):
        """Returns the dimensions of the action heads for agent initialization."""
        if self.use_legacy_actions or self.action_space_mode == "single_head_large":
            return {"num_actions": self.num_actions}
        else:
            return {"num_directions": self.num_directions, "num_lengths": self.num_lengths}

    def get_actions(self):
        """Returns list of all actions for legacy action space (for IL dataset generation)."""
        if self.use_legacy_actions:
            return self.legacy_actions
        else:
            raise ValueError("get_actions() is only available for legacy action space")

    def get_action_masks(self):
        """
        Returns boolean masks for valid actions based on curriculum stage.
        Used for action masking during training.

        Returns:
            dict: For multi-head: {"direction": bool array, "length": bool array}
                  For legacy: {"action": bool array}
        """
        if self.use_legacy_actions:
            # Legacy actions: all actions are always valid
            return {"action": [True] * self.num_actions}
        else:
            # Multi-head: return masks based on curriculum stage
            dir_mask = [False] * self.num_directions
            len_mask = [False] * self.num_lengths

            for idx in self.valid_direction_indices:
                dir_mask[idx] = True
            for idx in self.valid_length_indices:
                len_mask[idx] = True

            return {"direction": dir_mask, "length": len_mask}

    def get_action_dim(self):
        """Fallback for old code - total action space size."""
        return self.num_directions * self.num_lengths

    def get_state_dim(self):
        """Deprecated method - state dimension varies by modality."""
        warnings.warn(
            "The state dimension cannot be reliably determined from the environment. " "This method should not be used.", UserWarning
        )
        return [3, 224, 224]

    # ============================================================================
    # GROUND TRUTH GRAPH
    # ============================================================================

    @classmethod
    def get_ground_truth_graph(cls, floorplan_name: str):
        """
        Loads or generates the ground truth scene graph for a floorplan.
        Uses class-level caching to avoid redundant loads.
        """
        if floorplan_name in cls._gt_graph_cache:
            return cls._gt_graph_cache[floorplan_name]

        save_path = os.path.join(os.path.dirname(__file__), "..", "data", "gt_graphs", f"{floorplan_name}.json")

        if not os.path.exists(save_path):
            print(f"⚠️ GT Graph for {floorplan_name} not found. Generating...")
            # Lazy import - only needed if GT graphs need to be generated
            from components.scripts.generate_gt_graphs import generate_gt_scene_graphs

            generate_gt_scene_graphs(floorplans=[floorplan_name])

        gt_graph = GTGraph().load_from_file(save_path)
        cls._gt_graph_cache[floorplan_name] = gt_graph
        return gt_graph

    def get_viewpoint_to_objects(self):
        """
        Lazily load viewpoint_to_objects mapping for the current scene.
        Kept separate to avoid pulling large data into every GTGraph user.
        """
        scene_id = self.scene_number
        if scene_id in self._viewpoint_cache:
            return self._viewpoint_cache[scene_id]

        floorplan_name = f"FloorPlan{scene_id}"
        path = os.path.join(os.path.dirname(__file__), "..", "data", "gt_graphs", f"{floorplan_name}.json")
        with open(path, "r") as f:
            data = json.load(f)

        v2o = data.get("viewpoint_to_objects", {})
        self._viewpoint_cache[scene_id] = v2o
        return v2o

    # ============================================================================
    # OBSERVATION PROCESSING
    # ============================================================================

    def _update_graphs_and_viewpoints(self, local_sg, agent_pos, agent_rot):
        """
        Updates the global scene graph and viewpoints tracking.
        Called by both environments during observation processing.

        Viewpoint is position-only (no rotation) to prevent spin-in-place loophole
        where agents could artificially boost visibility by rotating at same location.
        """
        # Viewpoint = grid-discretized position (no rotation component)
        # This prevents the loophole where spinning in place increases visibility
        current_viewpoint = (
            round(agent_pos["x"] / self.grid_size) * self.grid_size,
            round(agent_pos["z"] / self.grid_size) * self.grid_size,
        )

        # Add local scene graph with viewpoint tracking
        self.global_sg.add_local_sg(local_sg, current_viewpoint=current_viewpoint)

        # Track viewpoints for diversity scoring
        # Note: Diversity tracking uses position-only to avoid rewarding spin-in-place behavior.
        diversity_viewpoint = current_viewpoint

        for node in local_sg.nodes.values():
            self.viewpoints[node.object_id].add(diversity_viewpoint)

    # ============================================================================
    # EPISODE METRICS
    # ============================================================================

    def _position_to_key(self, pos):
        if isinstance(pos, dict):
            x = pos.get("x", 0.0)
            z = pos.get("z", 0.0)
        else:
            x, z = pos

        grid = self.grid_size
        x = round(round(x / grid) * grid, 4)
        z = round(round(z / grid) * grid, 4)
        return (x, z)

    def _reset_episode_metrics(self, start_pos=None):
        self.total_path_length = 0.0
        self.visited_positions = set()
        self.exploration_coverage = 0.0

        if start_pos is not None:
            self.visited_positions.add(self._position_to_key(start_pos))
            self._update_exploration_coverage()

    def _update_exploration_coverage(self, current_pos=None):
        if current_pos is not None:
            self.visited_positions.add(self._position_to_key(current_pos))

        if self.reachable_positions:
            self.exploration_coverage = len(self.visited_positions) / len(self.reachable_positions)
        else:
            self.exploration_coverage = 0.0

        return self.exploration_coverage

    # ============================================================================
    # SCORING & REWARD
    # ============================================================================

    def compute_score(self, obs):
        """
        Computes exploration score based on node/edge recall.
        Returns: (total_score, recall_node, recall_edge)
        """
        num_gt_objects = len(self.gt_graph.nodes)
        discovered_nodes = [n for n in self.global_sg.nodes.values() if n.visibility >= 0.8]
        num_discovered = len(discovered_nodes)
        recall_node = num_discovered / num_gt_objects if num_gt_objects > 0 else 0.0

        num_gt_edges = len(self.gt_graph.edges)
        num_discovered_edges = len(self.global_sg.edges) if hasattr(self.global_sg, "edges") else 0
        recall_edge = num_discovered_edges / num_gt_edges if num_gt_edges > 0 else 0.0

        termination_bonus = 0.0 if obs.terminated else 0.0
        score = recall_node + termination_bonus

        return score, recall_node, recall_edge

    def _compute_reward(self, obs):
        """
        Unified reward computation for both environments.
        Includes similarity, diversity, time penalty, and collision penalties..
        """
        lambda_node, lambda_p, lambda_d = 0.1, 0.5, 0.001
        rho = self.rho

        Rnode = obs.info.get("recall_node", 0.0)
        Redge = obs.info.get("recall_edge", 0.0)

        Pnode = np.mean([n.visibility for n in self.global_sg.nodes.values()]) if self.global_sg.nodes else 0.0
        Pedge = 1.0

        diversity = sum(len(v) for v in self.viewpoints.values())

        # Decode action to determine if it's a pure rotation
        success = obs.info.get("move_action_success", True)
        actual_dist = obs.info.get("actual_dist", 0.0)
        target_dist = obs.info.get("target_dist", 0.0)
        action_info = obs.info.get("action", (0, 0))

        if isinstance(action_info, (list, tuple)) and len(action_info) >= 2:
            dir_idx, len_idx = action_info[0], action_info[1]
        elif isinstance(action_info, int):
            # Legacy action space stored as index; decode to (dir_idx, len_idx)
            if self.use_legacy_actions and self.legacy_actions:
                angle, dist = self.legacy_actions[action_info]
                dir_idx = self.action_angles.index(angle) if angle in self.action_angles else 0
                len_idx = 0 if abs(dist) < 1e-6 else 1
            else:
                dir_idx, len_idx = 0, 0
        else:
            dir_idx, len_idx = 0, 0

        stop_action_flag = obs.info.get("stop_action") if isinstance(obs.info, dict) else None
        if stop_action_flag is None and isinstance(action_info, (list, tuple)) and len(action_info) >= 3:
            stop_action_flag = action_info[2] == 1
        if stop_action_flag is None:
            stop_action_flag = dir_idx == 0 and len_idx == 0

        # Identify pure rotation: rotating but not moving (len_idx == 0) and not STOP.
        is_pure_rotation = (dir_idx != 0 and len_idx == 0) and not stop_action_flag

        # Track statistics
        if is_pure_rotation:
            self.num_pure_rotations += 1
        if not success and not stop_action_flag:  # Don't count STOP as collision
            self.num_collisions += 1

        # Time penalty: accumulate per step so discounted steps don't retroactively
        # reduce penalties from earlier steps.
        if self.step_count > self._time_penalty_step_count:
            missed_steps = self.step_count - self._time_penalty_step_count - 1
            if missed_steps > 0:
                # Fallback: charge full penalty for any unaccounted steps (shouldn't happen in normal stepping).
                self._time_penalty_total += rho * missed_steps

            step_penalty = rho * (1.0 if is_pure_rotation else 1.0)
            self._time_penalty_total += step_penalty
            self._time_penalty_step_count = self.step_count

        adjusted_time_penalty = self._time_penalty_total

        # Similarity + Diversity - Adjusted Time Penalty
        sim = lambda_node * (Rnode + lambda_p * Pnode) + Redge + lambda_p * Pedge
        score = sim + lambda_d * diversity - adjusted_time_penalty

        reward = score - self.last_score
        self.last_score = score

        # Collision/Movement penalties and bonuses
        # STOP action (dir_idx==0 and len_idx==0): Completely neutral - no bonus, no penalty
        # Pure rotation (len_idx==0 but dir_idx!=0): half time penalty (handled above)
        # Movement (len_idx>0): Collision penalty or movement bonus based on success

        if stop_action_flag:
            pass  # Completely neutral - agent learns from exploration vs. time penalty trade-off
        elif not success or actual_dist < (target_dist - 0.05):
            # Collision penalty (for failed movements or rotations)
            # Reduced from -0.05 to -0.02 to allow learning through exploration
            collision_penalty = -0.02
            reward += collision_penalty
        else:
            # Movement bonus: small fixed bonus to make movement slightly better than rotation
            # Main incentive remains exploration (new viewpoints), not just moving far
            if actual_dist > 0.05:  # Successful movement (not pure rotation)
                movement_bonus = 0.005
                reward += movement_bonus

        # Exploration bonus: reward discovering new viewpoints
        current_num_viewpoints = len(self.viewpoints)
        last_num_viewpoints = getattr(self, "_last_num_viewpoints", 0)
        if current_num_viewpoints > last_num_viewpoints:
            exploration_bonus = 0.01
            reward += exploration_bonus
            self._steps_since_discovery = 0
        else:
            self._steps_since_discovery = getattr(self, "_steps_since_discovery", 0) + 1
        self._last_num_viewpoints = current_num_viewpoints

        if stop_action_flag and self._steps_since_discovery >= self.stop_stagnation_steps:
            reward += self.stop_stagnation_bonus

        return reward

    # ============================================================================
    # STATE MANAGEMENT
    # ============================================================================

    def get_env_state(self):
        """Captures the current environment state for restoration."""
        return {
            "state": deepcopy(self.state),
            "global_sg": deepcopy(self.global_sg),
            "viewpoints": deepcopy(self.viewpoints),
            "last_score": self.last_score,
            "step_count": self.step_count,
            "last_num_viewpoints": self._last_num_viewpoints,
            "steps_since_discovery": self._steps_since_discovery,
            "time_penalty_total": self._time_penalty_total,
            "time_penalty_step_count": self._time_penalty_step_count,
            "total_path_length": self.total_path_length,
            "visited_positions": deepcopy(self.visited_positions),
            "exploration_coverage": self.exploration_coverage,
        }

    def restore_env_state(self, env_state):
        """Restores a previously captured environment state."""
        self.state = deepcopy(env_state["state"])
        self.global_sg = deepcopy(env_state["global_sg"])
        self.viewpoints = deepcopy(env_state["viewpoints"])
        self.last_score = env_state["last_score"]
        self.step_count = env_state["step_count"]
        self._last_num_viewpoints = env_state.get("last_num_viewpoints", 0)
        self._steps_since_discovery = env_state.get("steps_since_discovery", 0)
        self._time_penalty_total = env_state.get("time_penalty_total", 0.0)
        self._time_penalty_step_count = env_state.get("time_penalty_step_count", self.step_count)
        self.total_path_length = env_state.get("total_path_length", 0.0)
        self.visited_positions = deepcopy(env_state.get("visited_positions", set()))
        self.exploration_coverage = env_state.get("exploration_coverage", 0.0)

    # ============================================================================
    # ABSTRACT METHODS (must be implemented by subclasses)
    # ============================================================================

    @abstractmethod
    def reset(self, scene_number=None, random_start=False, start_position=None, start_rotation=None):
        """Reset the environment to initial state."""
        pass

    @abstractmethod
    def step(self, action_tuple):
        """
        Execute an action and return observation.

        Args:
            action_tuple: (direction_index, length_index) specifying movement action

        Returns:
            Observation object with state, reward, done flags, and info
        """
        pass

    @abstractmethod
    def get_agent_state(self):
        """Get current agent position and rotation."""
        pass

    @abstractmethod
    def restore_agent_state(self, agent_state):
        """Restore agent to a previous state."""
        pass

    @abstractmethod
    def close(self):
        """Clean up resources."""
        pass
