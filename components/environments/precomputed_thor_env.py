import copy
import io
import math
import pickle
import random
from collections import defaultdict

import numpy as np
from PIL import Image

from components.environments.abstract_thor_env import AbstractThorEnv
from components.graph.local_graph_builder import LocalSceneGraphBuilder
from components.utils.observation import Observation


# ============================================================================
# HELPER CLASSES FOR STORAGE ABSTRACTION
# ============================================================================


class CachedEvent:
    """Wrapper to make dict look like an event and decode images on the fly."""

    def __init__(self, data):
        # Deep copy to avoid mutating global cache when modifying lastActionSuccess
        self.metadata = copy.deepcopy(data["metadata"])

        # Convert BytesIO to bytes to avoid multiprocessing BufferError in Python 3.13
        frame_bytes = data["frame_bytes"]
        self._frame_bytes = frame_bytes.getvalue() if hasattr(frame_bytes, "getvalue") else frame_bytes

        depth_bytes = data["depth_bytes"]
        self._depth_bytes = depth_bytes.getvalue() if hasattr(depth_bytes, "getvalue") else depth_bytes

        seg_bytes = data.get("seg_bytes")
        self._seg_bytes = seg_bytes.getvalue() if seg_bytes and hasattr(seg_bytes, "getvalue") else seg_bytes

        # Lazy loading caches
        self._frame_cache = None
        self._depth_cache = None
        self._seg_cache = None

    @property
    def frame(self):
        if self._frame_cache is None and self._frame_bytes is not None:
            self._frame_cache = np.array(Image.open(io.BytesIO(self._frame_bytes)))
        return self._frame_cache

    @property
    def depth_frame(self):
        if self._depth_cache is None and self._depth_bytes is not None:
            img = np.array(Image.open(io.BytesIO(self._depth_bytes)))
            self._depth_cache = img.astype(np.float32) * (50.0 / 65535.0)
        return self._depth_cache

    @property
    def instance_segmentation_frame(self):
        if self._seg_cache is None and self._seg_bytes is not None:
            self._seg_cache = np.array(Image.open(io.BytesIO(self._seg_bytes)))
        return self._seg_cache


class InMemoryTransitionTable:
    """In-memory storage for transition tables (from pickle files)."""

    def __init__(self, data_dict, metadata=None):
        self.data = data_dict
        self.metadata = metadata or {}

    def get(self, x, z, rot):
        raw_data = self.data.get((x, z, rot), None)
        if raw_data is None:
            return None
        return CachedEvent(raw_data)

    def get_valid_keys(self):
        return list(self.data.keys())

    def check_compatibility(self, required_grid, required_rot):
        """Check if stored data is compatible with required granularity."""
        stored_grid = self.metadata.get("grid_size")
        if stored_grid is not None and stored_grid > required_grid + 1e-5:
            return False, f"Stored grid {stored_grid} is coarser than required {required_grid}"

        # Heuristic rotation check
        keys = list(self.data.keys())
        if not keys:
            return True, ""

        rotations = {k[2] for k in keys}
        if len(rotations) > 1:
            sorted_rots = sorted(list(rotations))
            min_rot_diff = min([sorted_rots[i + 1] - sorted_rots[i] for i in range(len(sorted_rots) - 1)])
            if min_rot_diff > required_rot:
                return (False, f"Detected rotation step {min_rot_diff} is coarser than required {required_rot}")

        return True, ""


class LMDBTransitionTable:
    """LMDB-based storage for large transition tables."""

    def __init__(self, lmdb_path):
        try:
            import lmdb as _lmdb
        except ImportError as exc:
            raise ImportError(
                "LMDB backend requested but the 'lmdb' package is not installed. "
                "Install it with `pip install lmdb` or set `use_lmdb=False`."
            ) from exc
        self._lmdb = _lmdb
        self.env = self._lmdb.open(lmdb_path, readonly=True, lock=False)

    def _make_key(self, x, z, rot):
        return f"{x:.2f}|{z:.2f}|{rot}".encode("ascii")

    def get(self, x, z, rot):
        key = self._make_key(x, z, rot)
        with self.env.begin() as txn:
            val_bytes = txn.get(key)
            if val_bytes is None:
                return None
            raw_data = pickle.loads(val_bytes)
            return CachedEvent(raw_data)

    def get_valid_keys(self):
        keys = []
        with self.env.begin() as txn:
            cursor = txn.cursor()
            for key, _ in cursor:
                try:
                    parts = key.decode("ascii").split("|")
                    if len(parts) == 3:
                        keys.append((float(parts[0]), float(parts[1]), int(float(parts[2]))))
                except ValueError:
                    continue
        return keys

    def check_compatibility(self, required_grid, required_rot):
        """Heuristically checks if DB contains fine-grained enough data."""
        found_fine_grid = False
        found_fine_rot = False

        with self.env.begin() as txn:
            cursor = txn.cursor()
            count = 0
            for key, _ in cursor:
                if count > 10000:
                    break
                count += 1

                try:
                    parts = key.decode("ascii").split("|")
                    x, z, rot = float(parts[0]), float(parts[1]), int(float(parts[2]))

                    # Check rotation divisibility
                    if rot % 90 != 0 and rot % required_rot == 0:
                        found_fine_rot = True

                    # Check grid: if x is like 1.1 (not 1.0 or 1.25), suggests fine grid
                    if abs(x * 100) % 25 != 0:
                        found_fine_grid = True

                except ValueError:
                    continue

        # Return True by default - heuristics can fail on sparse maps
        return True, ""

    def close(self):
        self.env.close()


# ============================================================================
# PRECOMPUTED THOR ENVIRONMENT
# ============================================================================


class PrecomputedThorEnv(AbstractThorEnv):
    """
    Offline AI2-THOR environment using precomputed transition tables.
    Simulates continuous-like movement by iterating through discrete grid.
    Inherits common functionality from AbstractThorEnv.
    """

    _transition_cache = {}
    _cache_pid = None  # Track process ID for cache invalidation

    def __init__(
        self,
        rho=0.014,
        scene_number=None,
        render=False,
        include_event_in_info: bool = True,
        grid_size=0.1,
        transition_tables_path="components/data/transition_tables",
        max_actions=40,
        use_lmdb=True,
        curriculum_stage=1,
        use_legacy_actions=False,
        action_space_mode="multi_head",
        stop_stagnation_steps=5,
        stop_stagnation_bonus=0.02,
    ):
        super().__init__(
            rho=rho,
            scene_number=scene_number,
            grid_size=grid_size,
            max_actions=max_actions,
            render=render,
            curriculum_stage=curriculum_stage,
            use_legacy_actions=use_legacy_actions,
            action_space_mode=action_space_mode,
            stop_stagnation_steps=stop_stagnation_steps,
            stop_stagnation_bonus=stop_stagnation_bonus,
        )

        self.include_event_in_info = include_event_in_info
        self.transition_tables_path = transition_tables_path
        self.use_lmdb = use_lmdb

        # Movement resolution for collision checking
        self.step_resolution = 0.1

        # Load transition table
        self.transition_table = self._load_transition_table(
            self.scene_number,
            transition_tables_path=self.transition_tables_path,
            use_lmdb=self.use_lmdb,
            grid_size=self.grid_size,
            rot_step=15,
        )

        # Current agent state
        self.current_pos = None
        self.current_rot = None
        self.last_event = None

    # ============================================================================
    # TRANSITION TABLE LOADING
    # ============================================================================

    @classmethod
    def _load_transition_table(
        cls, scene_number: int, transition_tables_path: str = None, use_lmdb: bool = True, grid_size=0.1, rot_step=15
    ):
        """Load and cache transition table for a scene."""
        import os

        # Check if we're in a new process (fork/spawn) and clear LMDB cache if so
        current_pid = os.getpid()
        if cls._cache_pid is None:
            cls._cache_pid = current_pid
        elif cls._cache_pid != current_pid:
            # We're in a new process - clear LMDB entries from cache
            # Close any open LMDB environments first
            for key, table in list(cls._transition_cache.items()):
                if hasattr(table, "close"):
                    try:
                        table.close()
                    except:
                        pass
            cls._transition_cache.clear()
            cls._cache_pid = current_pid

        if transition_tables_path is None:
            transition_tables_path = "components/data/transition_tables"

        key = (transition_tables_path, scene_number, use_lmdb)
        if key in cls._transition_cache:
            return cls._transition_cache[key]

        table_wrapper = None

        if use_lmdb:
            file_path = os.path.join(transition_tables_path, f"FloorPlan{scene_number}.lmdb")
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"LMDB not found at {file_path}")
            table_wrapper = LMDBTransitionTable(file_path)
        else:
            file_path = os.path.join(transition_tables_path, f"FloorPlan{scene_number}.pkl")
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Pickle not found at {file_path}")

            with open(file_path, "rb") as f:
                data = pickle.load(f)

            if isinstance(data, dict) and "table" in data:
                table_data = data["table"]
                meta_data = data
            else:
                table_data = data
                meta_data = {}

            table_wrapper = InMemoryTransitionTable(table_data, metadata=meta_data)

        # Validate compatibility
        is_valid, msg = table_wrapper.check_compatibility(grid_size, rot_step)
        if not is_valid:
            print(f"⚠️ Warning for FloorPlan{scene_number}: {msg}")

        cls._transition_cache[key] = table_wrapper
        return table_wrapper

    # ============================================================================
    # RESET
    # ============================================================================

    def reset(self, scene_number=None, random_start=False, start_position=None, start_rotation=None):
        if scene_number is not None:
            self.scene_number = scene_number
            self.transition_table = self._load_transition_table(
                self.scene_number, transition_tables_path=self.transition_tables_path, use_lmdb=self.use_lmdb, grid_size=self.grid_size
            )

        # Reset shared state
        self.builder = LocalSceneGraphBuilder()
        self.global_sg.__init__()  # Reset global scene graph
        self.step_count = 0
        self.last_score = 0.0
        self.viewpoints = defaultdict(set)
        self._last_num_viewpoints = 0  # reset exploration bonus tracker
        self._steps_since_discovery = 0
        self._time_penalty_total = 0.0
        self._time_penalty_step_count = 0

        # Reset episode statistics
        self.num_collisions = 0
        self.num_pure_rotations = 0

        # Load ground truth graph
        self.gt_graph = self.get_ground_truth_graph(f"FloorPlan{self.scene_number}")

        # Determine starting position
        if random_start:
            valid_keys = self.transition_table.get_valid_keys()
            x, z, rot = random.choice(valid_keys)
        elif start_position is not None and start_rotation is not None:
            x = round(start_position["x"], 2)
            z = round(start_position["z"], 2)
            rot = int(start_rotation["y"]) % 360
        else:
            raise ValueError("Must provide either random_start=True or start position/rotation")

        event = self.transition_table.get(x, z, rot)
        if event is None:
            raise ValueError(f"Invalid start state ({x}, {z}, {rot}): not in table. " f"Check if table matches grid_size.")

        self.current_pos = (x, z)
        self.current_rot = rot
        self.last_event = event

        obs = self._build_observation(event, reset=True)
        if not hasattr(self.transition_table, "reachable_positions"):
            keys = self.transition_table.get_valid_keys()
            self.transition_table.reachable_positions = {self._position_to_key((x, z)) for x, z, _ in keys}
        self.reachable_positions = self.transition_table.reachable_positions
        self._reset_episode_metrics(start_pos=(x, z))
        self._compute_reward(obs)

        return obs

    # ============================================================================
    # STEP
    # ============================================================================

    # ============================================================================
    # LEGACY ACTION PRIMITIVES
    # ============================================================================

    def transition_step(self, action_str):
        """
        Execute a single primitive action (legacy format).
        Used when use_legacy_actions=True to execute movement and rotation primitives.

        Args:
            action_str: One of ["MoveAhead", "MoveRight", "MoveLeft", "MoveBack",
                               "RotateRight", "RotateLeft", "Pass"]

        Returns:
            event: The resulting event after the action
        """
        if self.current_pos is None:
            raise ValueError("Call reset() before stepping.")

        x, z = self.current_pos
        rot = self.current_rot
        new_x, new_z, new_rot = x, z, rot
        success = True

        # Rotation primitives
        if action_str == "RotateRight":
            new_rot = (rot + 90) % 360
        elif action_str == "RotateLeft":
            new_rot = (rot - 90) % 360
        elif action_str in ["LookUp", "LookDown"]:
            # Camera tilt actions - no position change, treated as Pass
            pass
        elif action_str.startswith("Move"):
            # Compute translation based on current orientation
            if action_str == "MoveAhead":
                dx, dz = 0, self.grid_size
            elif action_str == "MoveBack":
                dx, dz = 0, -self.grid_size
            elif action_str == "MoveRight":
                dx, dz = self.grid_size, 0
            elif action_str == "MoveLeft":
                dx, dz = -self.grid_size, 0
            else:
                dx, dz = 0, 0

            # Rotate translation by agent heading
            angle = rot % 360
            if angle == 90:
                dx, dz = dz, -dx
            elif angle == 180:
                dx, dz = -dx, -dz
            elif angle == 270:
                dx, dz = -dz, dx
            new_x, new_z = x + dx, z + dz
        else:
            # Pass or unknown
            pass

        # Lookup event in transition table
        event = self.transition_table.get(round(new_x, 2), round(new_z, 2), new_rot)
        if event is None:
            # Invalid transition -> stay in place
            event = self.transition_table.get(round(x, 2), round(z, 2), new_rot)
            new_x, new_z = x, z
            success = False

        # Update agent pose
        self.current_pos = (new_x, new_z)
        self.current_rot = new_rot
        event.metadata["lastActionSuccess"] = success
        self.last_event = event

        return event

    # ============================================================================
    # STEP
    # ============================================================================

    def step(self, action_tuple):
        """Execute action using iterative collision checking (multi-head) or primitive execution (legacy)."""
        if self.current_pos is None:
            raise ValueError("Call reset() before stepping.")

        # Decode single_head_large flat action index to (dir_idx, len_idx)
        if self.action_space_mode == "single_head_large":
            # Handle both int format and tuple format (flat_idx, -1)
            if isinstance(action_tuple, int):
                flat_idx = action_tuple
            else:
                flat_idx = action_tuple[0]
            # Decode: flat_idx = dir_idx * 21 + len_idx
            dir_idx = flat_idx // 21
            len_idx = flat_idx % 21
            action_tuple = (dir_idx, len_idx)

        if self.use_legacy_actions:
            # Legacy action space: 16 discrete actions (8 directions × 2 lengths)
            # Each action is a (angle, length) tuple
            action_idx = action_tuple if isinstance(action_tuple, int) else action_tuple[0]

            # Decode action into (angle, length) tuple
            angle, target_dist = self.legacy_actions[action_idx]

            is_stop_action = angle == 0 and target_dist == 0.0

            # Calculate new rotation (relative to current)
            start_x, start_z = self.current_pos
            current_rot = self.current_rot
            new_rot = (current_rot + angle) % 360

            # NEW ORDER: Move FIRST in current direction, THEN rotate
            # Compute movement direction based on CURRENT rotation (not new_rot)
            actual_dist = 0.0
            success = True
            num_steps = int(round(target_dist / self.step_resolution))

            curr_x, curr_z = start_x, start_z
            rad = math.radians(current_rot)  # Use current_rot for movement direction!
            step_dx = math.sin(rad) * self.step_resolution
            step_dz = math.cos(rad) * self.step_resolution

            # Try to move forward step by step in CURRENT direction
            event = self.last_event
            for _ in range(num_steps):
                next_x = round(curr_x + step_dx, 2)
                next_z = round(curr_z + step_dz, 2)
                next_event = self.transition_table.get(next_x, next_z, current_rot)

                if next_event is not None:
                    curr_x, curr_z = next_x, next_z
                    event = next_event
                    actual_dist += self.step_resolution
                else:
                    success = False
                    break

            self.current_pos = (curr_x, curr_z)

            # Apply rotation AFTER movement (preparation for next action)
            if angle != 0:
                rot_event = self.transition_table.get(curr_x, curr_z, new_rot)
                if rot_event is not None:
                    self.current_rot = new_rot
                    event = rot_event  # Use rotated view for observation
                else:
                    # Rotation not possible at final position - keep current rotation
                    success = False
            # If angle == 0, stay at current_rot

            self.last_event = event
            event.metadata["lastActionSuccess"] = success

            # For compatibility with info tracking
            dir_idx = self.action_angles.index(angle)
            len_idx = 0 if target_dist == 0.0 else 1
            # Create tuple for info storage
            action_tuple = (dir_idx, len_idx)

        else:
            # Multi-head action space: (dir_idx, len_idx[, stop_flag]) tuple
            stop_flag_present = isinstance(action_tuple, (tuple, list)) and len(action_tuple) == 3
            if stop_flag_present:
                dir_idx, len_idx, stop_flag = action_tuple
            else:
                dir_idx, len_idx = action_tuple
                stop_flag = 0

            # 1. Decode Action
            rel_rot = self.action_angles[dir_idx]
            target_dist = self.action_lengths[len_idx]

            if stop_flag_present and stop_flag == 1:
                rel_rot = 0
                target_dist = 0.0

            is_stop_action = stop_flag == 1 if stop_flag_present else (dir_idx == 0 and len_idx == 0)

            # 2. Determine new Rotation
            start_x, start_z = self.current_pos
            current_rot = self.current_rot
            new_rot = (current_rot + rel_rot) % 360

            # NEW ORDER: Move FIRST in current direction, THEN rotate
            # Compute movement direction based on CURRENT rotation (not new_rot)
            actual_dist = 0.0
            success = True
            num_steps = int(round(target_dist / self.step_resolution))

            curr_x, curr_z = start_x, start_z
            rad = math.radians(current_rot)  # Use current_rot for movement direction!
            step_dx = math.sin(rad) * self.step_resolution
            step_dz = math.cos(rad) * self.step_resolution

            # Try to move forward step by step in CURRENT direction
            event = self.last_event
            for _ in range(num_steps):
                next_x = round(curr_x + step_dx, 2)
                next_z = round(curr_z + step_dz, 2)
                next_event = self.transition_table.get(next_x, next_z, current_rot)

                if next_event is not None:
                    curr_x, curr_z = next_x, next_z
                    event = next_event
                    actual_dist += self.step_resolution
                else:
                    success = False
                    break

            self.current_pos = (curr_x, curr_z)

            # Apply rotation AFTER movement (preparation for next action)
            if rel_rot != 0:
                rot_event = self.transition_table.get(curr_x, curr_z, new_rot)
                if rot_event is not None:
                    self.current_rot = new_rot
                    event = rot_event  # Use rotated view for observation
                else:
                    # Rotation not possible at final position - keep current rotation
                    success = False
            # If rel_rot == 0, stay at current_rot

            self.last_event = event
            event.metadata["lastActionSuccess"] = success

        self.total_path_length += actual_dist
        self._update_exploration_coverage(self.current_pos)

        # 4. Process Observation
        obs = self._build_observation(event)
        self.step_count += 1

        # 5. Check termination conditions
        # STOP always ends the episode (terminated=True).
        # Max steps ends via truncation only (terminated=False, truncated=True).
        truncated = False
        terminated = False

        if is_stop_action:
            terminated = True
        elif self.step_count >= self.max_actions:
            truncated = True

        obs.terminated = terminated
        obs.truncated = truncated

        # 6. Compute score and reward
        score, recall_node, recall_edge = self.compute_score(obs)

        all_nodes_seen = len([k for k, n in self.global_sg.nodes.items() if n.visibility >= 0.8]) == len(self.gt_graph.nodes)

        obs.info = {
            "score": score,
            "recall_node": recall_node,
            "recall_edge": recall_edge,
            "action": action_tuple,
            "agent_pos": self.current_pos,
            "move_action_success": success,
            "actual_dist": actual_dist,
            "target_dist": target_dist,
            "max_steps_reached": truncated,
            "all_nodes_seen": all_nodes_seen,
            "success": all_nodes_seen and terminated,
            "stop_action": is_stop_action,
            "num_collisions": self.num_collisions,
            "num_pure_rotations": self.num_pure_rotations,
            "total_path_length": self.total_path_length,
            "exploration_coverage": self.exploration_coverage,
        }
        if self.include_event_in_info:
            obs.info["event"] = event

        obs.reward = self._compute_reward(obs)

        return obs

    # ============================================================================
    # OBSERVATION PROCESSING
    # ============================================================================

    def _build_observation(self, event, reset=False):
        """Build observation from cached event."""
        rgb = event.frame
        depth = getattr(event, "depth_frame", None)

        if depth is None:
            depth = np.zeros((rgb.shape[0], rgb.shape[1]), dtype=np.float32)

        # Build local scene graph
        local_sg = self.builder.build_from_metadata(event.metadata)

        # Update global graph and viewpoints (shared method)
        agent_view = event.metadata["agent"]
        agent_pos = {"x": agent_view["position"]["x"], "z": agent_view["position"]["z"]}
        agent_rot = agent_view["rotation"]["y"]
        self._update_graphs_and_viewpoints(local_sg, agent_pos, agent_rot)

        # State structure: [RGB, Depth, LocalSG, GlobalSG]
        self.state = [rgb, depth, local_sg, self.global_sg]

        info = {"event": event} if self.include_event_in_info else {}
        return Observation(state=self.state, info=info)

    # ============================================================================
    # AGENT STATE
    # ============================================================================

    def get_agent_state(self):
        """Get current agent position and rotation."""
        ag = self.last_event.metadata["agent"]
        return {"position": (ag["position"]["x"], ag["position"]["z"]), "rotation": ag["rotation"]["y"]}

    def restore_agent_state(self, state):
        """Restore agent to a previous state."""
        self.current_pos, self.current_rot = state["position"], state["rotation"]

    # ============================================================================
    # UNSUPPORTED FEATURES
    # ============================================================================

    def get_top_down_view(self):
        """Not available in precomputed environment."""
        return None

    def visualize_shortest_path(self, start, target):
        """Not available in precomputed environment."""
        return None

    # ============================================================================
    # CLEANUP
    # ============================================================================

    def close(self):
        """Clean up resources."""
        if hasattr(self, "transition_table") and hasattr(self.transition_table, "close"):
            self.transition_table.close()

    @classmethod
    def clear_cache(cls):
        """
        Close all cached LMDB environments and clear the cache.
        Should be called between training seeds to avoid using closed LMDB handles.
        """
        for key, table in list(cls._transition_cache.items()):
            if hasattr(table, "close"):
                try:
                    table.close()
                except Exception:
                    pass  # Already closed
        cls._transition_cache.clear()
