import platform
import random
import warnings
from collections import defaultdict

import numpy as np
from PIL import Image
from ai2thor.controller import Controller
from ai2thor.platform import CloudRendering

from components.environments.abstract_thor_env import AbstractThorEnv
from components.graph.local_graph_builder import LocalSceneGraphBuilder
from components.utils.observation import Observation

warnings.filterwarnings("ignore", message="could not connect to X Display*", category=UserWarning)


class ThorEnv(AbstractThorEnv):
    """
    Online AI2-THOR environment using the real simulator.
    Inherits common functionality from AbstractThorEnv.
    """

    def __init__(
        self,
        rho=0.02,
        scene_number=None,
        render=False,
        grid_size=0.25,
        max_actions=40,
        additional_images=True,
        device=None,
        curriculum_stage=1,
        use_legacy_actions=False,
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
            stop_stagnation_steps=stop_stagnation_steps,
            stop_stagnation_bonus=stop_stagnation_bonus,
        )

        self.visibilityDistance = 50
        self.additional_images = additional_images
        self.render = render if platform.system() == "Linux" else True

        # Initialize Controller
        controller_kwargs = dict(
            moveMagnitude=self.grid_size,
            gridSize=self.grid_size,
            visibilityDistance=self.visibilityDistance,
            renderDepthImage=True,
            renderSemanticSegmentation=additional_images,
            renderInstanceSegmentation=additional_images,
            gpu_device=device,
            width=224,
            height=224,
            snapToGrid=False,
            rotateStepDegrees=15,
        )

        if not self.render:
            controller_kwargs["platform"] = CloudRendering

        try:
            self.controller = Controller(**controller_kwargs)
        except TimeoutError as e:
            print(f"[TIMEOUT] Controller initialization timed out: {e}")
            raise  # Re-raise so generate_dataset can handle it

        # ThorEnv-specific state
        self.agent_state = None
        self.td_center_x = None
        self.td_center_z = None
        self.td_ortho_size = None
        self._reachable_positions_cache = None
        self._reachable_positions_cache_scene = None
        self._reachable_positions_cache_grid = None

    # ============================================================================
    # RESET
    # ============================================================================

    def reset(self, scene_number=None, random_start=False, start_position=None, start_rotation=None):
        if scene_number is not None:
            self.scene_number = scene_number

        try:
            self.controller.reset(
                scene=f"FloorPlan{self.scene_number}", gridSize=self.grid_size, snapToGrid=False, visibilityDistance=self.visibilityDistance
            )
        except TimeoutError:
            print(f"[TIMEOUT] Reset timed out for scene FloorPlan{self.scene_number}. Restarting controller...")
            self.reset_hard()
            return self.reset(scene_number=self.scene_number, random_start=random_start, start_position=start_position, start_rotation=start_rotation)

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

        # Determine starting position
        if random_start:
            reachable = self.safe_step(action="GetReachablePositions").metadata["actionReturn"]
            pos = random.choice(reachable)
            rot = {"x": 0, "y": random.choice([0, 90, 180, 270]), "z": 0}
        elif start_position and start_rotation:
            pos = start_position
            rot = start_rotation
        else:
            pos = None

        if pos is not None:
            self.safe_step(action="Teleport", position=pos, rotation=rot)

        self.reachable_positions = self._load_reachable_positions()
        event = self.safe_step(action="Pass")
        obs = self._process_event(event)

        # Standardize state structure: [RGB, Depth, LocalSG, GlobalSG]
        self.state = [obs.state[0], obs.state[1], obs.state[2], obs.state[3]]
        obs.state = self.state

        self._reset_episode_metrics(start_pos=event.metadata["agent"]["position"])

        self._compute_reward(obs)

        # Load ground truth graph
        self.gt_graph = self.get_ground_truth_graph(f"FloorPlan{self.scene_number}")

        # Setup top-down camera
        self.td_center_x, self.td_center_z, self.td_ortho_size = self.add_topdown_camera_covering_scene()

        return obs

    def _load_reachable_positions(self):
        if (
            self._reachable_positions_cache
            and self._reachable_positions_cache_scene == self.scene_number
            and self._reachable_positions_cache_grid == self.grid_size
        ):
            return self._reachable_positions_cache

        reachable = self.safe_step(action="GetReachablePositions").metadata["actionReturn"]
        positions = {self._position_to_key(pos) for pos in reachable}
        self._reachable_positions_cache = positions
        self._reachable_positions_cache_scene = self.scene_number
        self._reachable_positions_cache_grid = self.grid_size
        return positions

    # ============================================================================
    # STEP
    # ============================================================================

    def step(self, action):
        """
        Executes an action.
        - Legacy mode: action is int (0-15), decoded to (relative_angle, distance)
        - Multi-head mode: action is tuple (direction_index, length_index)
        - Single-head-large mode: action is flat index or tuple (flat_idx, -1)
        """
        # Decode single_head_large flat action index to (dir_idx, len_idx)
        if self.action_space_mode == "single_head_large":
            if isinstance(action, int):
                flat_idx = action
            else:
                flat_idx = action[0]
            # Decode: flat_idx = dir_idx * 21 + len_idx
            dir_idx = flat_idx // 21
            len_idx = flat_idx % 21
            action = (dir_idx, len_idx)

        if self.use_legacy_actions:
            return self._step_legacy(action)
        else:
            return self._step_multihead(action)

    def _step_legacy(self, action_idx):
        """
        Execute legacy action: (relative_angle, distance).

        NEW ORDER (safer for depth-based navigation):
        1. FIRST: Move in CURRENT direction (depth map is valid for this direction)
        2. THEN: Rotate (preparation for next action, depth will be updated)

        This ensures the agent only moves in directions it can currently see.
        """
        # 1. Decode action
        angle, target_dist = self.legacy_actions[action_idx]

        prev_pos = self.agent_state["position"]
        current_rot = self.controller.last_event.metadata["agent"]["rotation"]["y"]
        new_rot = (current_rot + angle) % 360

        # 2. Execute movement FIRST (in current direction, before rotation)
        move_event = None
        if target_dist > 0.0:
            move_event = self.safe_step(action="MoveAhead", moveMagnitude=target_dist)
            final_event = move_event
        else:
            # STOP action: just pass
            final_event = self.safe_step(action="Pass")

        # 3. Apply rotation AFTER movement (preparation for next step)
        rotation_event = None
        if angle != 0:
            rotation_event = self.safe_step(action="RotateRight", degrees=float(angle))
            # Use rotation event for observation (contains updated depth map for next action)
            if rotation_event is not None:
                final_event = rotation_event

        # 4. Compute distance traveled (based on movement, not rotation)
        success = move_event.metadata["lastActionSuccess"] if move_event is not None else True
        new_pos = final_event.metadata["agent"]["position"]

        p_vec = np.array([prev_pos["x"], prev_pos["z"]])
        n_vec = np.array([new_pos["x"], new_pos["z"]])
        actual_dist = np.linalg.norm(n_vec - p_vec)

        self.total_path_length += actual_dist
        self._update_exploration_coverage(new_pos)

        # 5. Process observation
        obs = self._process_event(final_event)
        self.step_count += 1

        # 6. Check termination
        is_stop_action = angle == 0 and target_dist == 0.0
        truncated = False
        terminated = False
        if is_stop_action:
            terminated = True
        elif self.step_count >= self.max_actions:
            truncated = True

        obs.truncated = truncated
        obs.terminated = terminated

        # 7. Compute score and reward
        score, recall_node, recall_edge = self.compute_score(obs)

        obs.info = {
            "event": final_event,
            "score": score,
            "recall_node": recall_node,
            "recall_edge": recall_edge,
            "action": action_idx,
            "prev_pos": (prev_pos["x"], prev_pos["z"]),
            "agent_pos": (new_pos["x"], new_pos["z"]),
            "move_action_success": success,
            "actual_dist": actual_dist,
            "target_dist": target_dist,
            "max_steps_reached": truncated,
            "stop_action": is_stop_action,
            "num_collisions": self.num_collisions,
            "num_pure_rotations": self.num_pure_rotations,
            "total_path_length": self.total_path_length,
            "exploration_coverage": self.exploration_coverage,
        }

        obs.reward = self._compute_reward(obs)

        return obs

    def _step_multihead(self, action_tuple):
        """
        Execute multi-head action: (direction_index, length_index).

        NEW ORDER (safer for depth-based navigation):
        1. FIRST: Move forward in CURRENT direction (depth map is valid)
        2. THEN: Apply relative rotation (preparation for next action)

        This ensures the agent only moves in directions it can currently see.
        """
        stop_flag_present = isinstance(action_tuple, (tuple, list)) and len(action_tuple) == 3
        if stop_flag_present:
            dir_idx, len_idx, stop_flag = action_tuple
        else:
            dir_idx, len_idx = action_tuple
            stop_flag = 0

        # 1. Decode Action
        relative_rotation = self.action_angles[dir_idx]
        target_dist = self.action_lengths[len_idx]

        if stop_flag_present and stop_flag == 1:
            relative_rotation = 0
            target_dist = 0.0

        # Store position before movement
        prev_pos = self.agent_state["position"]

        # 2. Execute Movement FIRST (in current direction, before rotation)
        move_event = self.safe_step(action="MoveAhead", moveMagnitude=target_dist)

        # 3. Apply Relative Rotation AFTER movement (preparation for next step)
        rotation_event = None
        if relative_rotation > 0:
            rotation_event = self.safe_step(action="RotateRight", degrees=float(relative_rotation))

        # 4. Check results & Compute Distance
        # Use move_event for success (movement is what matters for collision)
        success = move_event.metadata["lastActionSuccess"]
        new_pos = move_event.metadata["agent"]["position"]

        p_vec = np.array([prev_pos["x"], prev_pos["z"]])
        n_vec = np.array([new_pos["x"], new_pos["z"]])
        actual_dist = np.linalg.norm(n_vec - p_vec)

        self.total_path_length += actual_dist
        self._update_exploration_coverage(new_pos)

        # 5. Process Observation
        # Use rotation_event if it exists (contains updated depth map),
        # otherwise use move_event
        final_event = rotation_event if rotation_event is not None else move_event
        obs = self._process_event(final_event)
        self.step_count += 1

        # 6. Check termination conditions
        # Episode ends when:
        # - Agent explicitly chooses STOP (dir_idx==0 AND len_idx==0), OR
        # - Max steps reached
        # Pure rotations (len_idx==0 but dir_idx!=0) do NOT end the episode
        is_stop_action = stop_flag == 1 if stop_flag_present else (dir_idx == 0 and len_idx == 0)
        truncated = False
        terminated = False
        if is_stop_action:
            terminated = True
        elif self.step_count >= self.max_actions:
            truncated = True

        obs.truncated = truncated
        obs.terminated = terminated

        # 7. Compute score and reward
        score, recall_node, recall_edge = self.compute_score(obs)

        obs.info = {
            "event": final_event,
            "score": score,
            "recall_node": recall_node,
            "recall_edge": recall_edge,
            "action": action_tuple,
            "prev_pos": (prev_pos["x"], prev_pos["z"]),
            "agent_pos": (new_pos["x"], new_pos["z"]),
            "move_action_success": success,
            "actual_dist": actual_dist,
            "target_dist": target_dist,
            "max_steps_reached": truncated,
            "stop_action": is_stop_action,
            "num_collisions": self.num_collisions,
            "num_pure_rotations": self.num_pure_rotations,
            "total_path_length": self.total_path_length,
            "exploration_coverage": self.exploration_coverage,
        }

        obs.reward = self._compute_reward(obs)

        return obs

    # ============================================================================
    # OBSERVATION PROCESSING
    # ============================================================================

    def _process_event(self, event):
        """Process AI2-THOR event into observation format."""
        rgb = event.frame
        depth = event.depth_frame

        if depth is None:
            warnings.warn("Depth frame is None, creating zero depth map.", RuntimeWarning)
            depth = np.zeros((rgb.shape[0], rgb.shape[1]), dtype=np.float32)

        # Build local scene graph
        local_sg = self.builder.build_from_metadata(event.metadata)

        # Update global graph and viewpoints (shared method)
        agent_pos = event.metadata["agent"]["position"]
        agent_rot = event.metadata["agent"]["rotation"]["y"]
        self._update_graphs_and_viewpoints(local_sg, agent_pos, agent_rot)

        # State structure: [RGB, Depth, LocalSG, GlobalSG]
        return Observation(state=[rgb, depth, local_sg, self.global_sg], info={"event": event})

    # ============================================================================
    # CONTROLLER INTERACTION
    # ============================================================================

    def safe_step(self, *args, **kwargs):
        """
        Wraps controller.step to handle crashes/timeouts.

        Args:
            *args: Positional arguments to pass to controller.step
            **kwargs: Keyword arguments to pass to controller.step

        Returns:
            Event object from controller.step
        """
        try:
            if self.controller.last_event:
                self.agent_state = self.get_agent_state()
            return self.controller.step(*args, **kwargs)
        except TimeoutError:
            print(f"[TIMEOUT] Action '{kwargs.get('action', 'unknown')}' timed out. Restarting...")
            self.reset_hard()
            return self.controller.step(*args, **kwargs)

    def reset_hard(self):
        """
        Hard reset of the controller in case of crash.

        Stops the controller, recreates it with stored scene configuration,
        and restores the agent to previous position/rotation if available.

        Returns:
            None
        """
        try:
            self.controller.stop()
        except Exception:
            pass

        controller_kwargs = dict(
            moveMagnitude=self.grid_size,
            gridSize=self.grid_size,
            visibilityDistance=self.visibilityDistance,
            renderDepthImage=True,
            renderSemanticSegmentation=self.additional_images,
            renderInstanceSegmentation=self.additional_images,
            snapToGrid=False,
        )

        if not self.render:
            controller_kwargs["platform"] = CloudRendering

        self.controller = Controller(**controller_kwargs)
        self.reset(scene_number=self.scene_number)

        if self.agent_state:
            self.restore_agent_state(self.agent_state)

    # ============================================================================
    # AGENT STATE
    # ============================================================================

    def get_agent_state(self):
        """
        Get current agent position and rotation.

        Returns:
            dict with keys "position" and "rotation"
        """
        agent = self.controller.last_event.metadata["agent"]
        return {"position": agent["position"], "rotation": agent["rotation"]}

    def restore_agent_state(self, agent_state):
        """
        Restore agent to a previous state.

        Args:
            agent_state: dict with "position" and "rotation" keys

        Returns:
            None
        """
        event = self.controller.step(action="Teleport", position=agent_state["position"], rotation=agent_state["rotation"], horizon=0)
        if event.metadata["lastActionSuccess"]:
            self.agent_state = agent_state
        else:
            raise EnvironmentError("Agent State could not be restored.")
        self.controller.step(action="Pass")

    # ============================================================================
    # TOP-DOWN CAMERA
    # ============================================================================

    def add_topdown_camera_covering_scene(self, pad=0.10, desired_hw=None):
        """Setup orthographic top-down camera covering the entire scene."""
        ev = self.safe_step(action="Pass")
        bounds = ev.metadata["sceneBounds"]
        center = bounds["center"]
        size = bounds["size"]

        if desired_hw is not None:
            H, W = desired_hw
            aspect = W / H
        else:
            aspect = 1.0

        z_span = size["z"] + 2 * pad
        x_span = size["x"] + 2 * pad

        required_ortho_for_z = 0.5 * z_span
        required_ortho_for_x = 0.5 * x_span / max(1e-6, aspect)
        ortho_size = max(required_ortho_for_z, required_ortho_for_x)

        top_camera_height = size["y"] - 0.5

        self.safe_step(
            action="AddThirdPartyCamera",
            rotation=dict(x=90, y=0, z=0),
            position=dict(x=center["x"], y=top_camera_height, z=center["z"]),
            orthographic=True,
            orthographicSize=ortho_size,
        )

        ev2 = self.safe_step(action="Pass")
        frame = ev2.third_party_camera_frames[0]
        H, W, _ = frame.shape
        true_aspect = W / H

        if abs(true_aspect - aspect) > 1e-6:
            required_ortho_for_x = 0.5 * x_span / max(1e-6, true_aspect)
            ortho_size = max(required_ortho_for_z, required_ortho_for_x)

            self.safe_step(
                action="AddThirdPartyCamera",
                rotation=dict(x=90, y=0, z=0),
                position=dict(x=center["x"], y=top_camera_height, z=center["z"]),
                orthographic=True,
                orthographicSize=ortho_size,
            )

        return center["x"], center["z"], ortho_size

    def get_top_down_view(self):
        """Returns the current top-down view image as numpy array (H,W,3)."""
        event = self.safe_step(action="Pass")
        if hasattr(event, "third_party_camera_frames") and event.third_party_camera_frames:
            return event.third_party_camera_frames[0]
        raise RuntimeError("No third-party camera frames found.")

    def visualize_shortest_path(self, start, target):
        """Visualizes the shortest path from start to goal on top-down view."""
        if len(target) == 2:
            target = {"x": target["x"], "y": start["y"], "z": target["z"]}

        event = self.safe_step(action="GetShortestPathToPoint", position=start, target=target)
        path = event.metadata["actionReturn"]["corners"]

        event = self.safe_step(action="VisualizePath", positions=path, grid=False, endText="Target")

        if hasattr(event, "third_party_camera_frames") and event.third_party_camera_frames:
            return Image.fromarray(event.third_party_camera_frames[0])
        return None

    # ============================================================================
    # UTILITY
    # ============================================================================

    def try_action(self, action, agent_pos=None, agent_rot=None):
        """Try an action without permanently changing the environment state."""
        env_state = self.get_env_state()
        agent_state = self.get_agent_state()

        if agent_pos is not None and agent_rot is not None:
            self.safe_step(action="Teleport", position=agent_pos, rotation=dict(x=0, y=agent_rot, z=0))

        event = self.safe_step(action=action)

        self.restore_env_state(env_state)
        self.restore_agent_state(agent_state)

        return event.metadata["lastActionSuccess"]

    def close(self):
        """Clean up controller resources."""
        self.controller.stop()
