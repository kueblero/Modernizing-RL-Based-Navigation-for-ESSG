import ast
import copy
import math
import random
import re
from collections import namedtuple, deque

from components.environments.thor_env import ThorEnv


class ImitationLabeler:
    def __init__(self, env):
        self.env = env

    def compute_score(self, env, visibility_before: dict, event_before_action, viewpoints, alpha=0.8):
        event = env.controller.last_event
        visibility_after = {k: n.visibility for k, n in self.env.global_sg.nodes.items()}

        score = 0.0
        for obj_id, vis_after in visibility_after.items():
            vis_before = visibility_before.get(obj_id, 0.0)

            updated_vis = 1 - (1 - vis_before) * (1 - alpha * vis_after)

            delta_vis = updated_vis - vis_before

            score += delta_vis

            if vis_before < 0.8 <= updated_vis:
                score += 1

        # bonus for exploring a new viewpoint (using global_sg viewpoint history)
        if hasattr(self.env, "global_sg") and hasattr(self.env.global_sg, "get_visited_viewpoints"):
            # Convert agent position to grid-based viewpoint key (x, z)
            agent_pos = event.metadata["agent"]["position"]
            grid_x = round(agent_pos["x"] / self.env.grid_size) * self.env.grid_size
            grid_z = round(agent_pos["z"] / self.env.grid_size) * self.env.grid_size
            current_viewpoint = (grid_x, grid_z)

            visited_viewpoints = self.env.global_sg.get_visited_viewpoints()
            occupancy_bonus = 0.9 if current_viewpoint not in visited_viewpoints else 0.0
        else:
            occupancy_bonus = 0.0

        vp_key = next(iter(viewpoints))
        vp_pos, vp_rot = self.deserialize_viewpoint(vp_key)

        # Calculate distance to target viewpoint before and after action
        dx_before = abs(event_before_action.metadata["agent"]["position"]["x"] - vp_pos["x"])
        dz_before = abs(event_before_action.metadata["agent"]["position"]["z"] - vp_pos["z"])
        dist_before = dx_before + dz_before  # Manhattan distance

        dx_after = abs(event.metadata["agent"]["position"]["x"] - vp_pos["x"])
        dz_after = abs(event.metadata["agent"]["position"]["z"] - vp_pos["z"])
        dist_after = dx_after + dz_after

        # Reward getting closer to the viewpoint
        distance_bonus = 0.0
        if dist_after < dist_before:
            # Bonus proportional to distance reduction (normalized by grid size)
            distance_reduction = (dist_before - dist_after) / self.env.grid_size
            distance_bonus = distance_reduction * 2.0  # Scale factor to make it meaningful
        elif dist_after > dist_before:
            # Small penalty for moving away from target
            distance_increase = (dist_after - dist_before) / self.env.grid_size
            distance_bonus = -distance_increase * 1.0

        return score + distance_bonus + occupancy_bonus

    def recover_missing_viewpoints(self, viewpoints, threshold=0.2):
        """
        If some objects are not yet sufficiently visible and all viewpoints have been explored,
        reintroduce viewpoints that help cover the missing objects.
        """
        global_seen = {k for k, n in self.env.global_sg.nodes.items() if n.visibility >= 0.8}
        all_nodes = set(self.env.gt_graph.nodes.keys())

        missing = all_nodes - global_seen
        if not missing:
            return  # All done

        # Search for viewpoints that see these objects
        recovered_viewpoints = {}
        v2o = self.env.get_viewpoint_to_objects()

        for vp_key, obj_list in v2o.items():
            recovered = []
            for obj in obj_list:
                for obj_id, vis in obj.items():
                    if obj_id in missing and vis >= threshold:
                        recovered.append(obj_id)
            if recovered:
                recovered_viewpoints[vp_key] = recovered

        if recovered_viewpoints:
            print(f"Recovered {len(recovered_viewpoints)} viewpoints for missing objects: {missing}")
            viewpoints.update(recovered_viewpoints)

    def get_next_action(self, agent_pos, agent_rot, viewpoints, tol: float = 0.2):
        """
        Returns the best action index that brings the agent closer to the next viewpoint.
        Uses the new legacy action space: 16 actions = 8 directions × 2 lengths.

        Returns:
            target_angle: Desired rotation angle (0-315 in 45° steps)
            target_length: Desired movement distance (0.0m or 0.3m)
        """
        if isinstance(agent_rot, dict):
            agent_rot = int(round(agent_rot["y"], 0))

        # Filter out already visited viewpoints
        while viewpoints:
            vp_key = next(iter(viewpoints))
            vp_pos, vp_rot = self.deserialize_viewpoint(vp_key)
            if abs(agent_pos["x"] - vp_pos["x"]) < tol and abs(agent_pos["z"] - vp_pos["z"]) < tol:
                if agent_rot == vp_rot:
                    del viewpoints[vp_key]
                else:
                    # Agent is at position but wrong rotation - rotate in place
                    angle_diff = (vp_rot - agent_rot) % 360
                    # Find closest action angle (considers wrap-around automatically)
                    best_angle = min(
                        self.env.action_angles, key=lambda a: min(abs(a - angle_diff), abs(a - angle_diff + 360), abs(a - angle_diff - 360))
                    )
                    return best_angle, 0.0, vp_pos
            else:
                break

        if not viewpoints:
            self.recover_missing_viewpoints(viewpoints)
            if not viewpoints:
                raise ValueError("No viewpoints left to explore (even after recovery)")

            vp_items = sorted(
                viewpoints.items(),
                key=lambda item: abs(agent_pos["x"] - self.deserialize_viewpoint(item[0])[0]["x"])
                + abs(agent_pos["z"] - self.deserialize_viewpoint(item[0])[0]["z"]),
            )

            for vp_key, _ in vp_items:
                vp_pos_candidate, vp_rot_candidate = self.deserialize_viewpoint(vp_key)
                if abs(agent_pos["x"] - vp_pos_candidate["x"]) < tol and abs(agent_pos["z"] - vp_pos_candidate["z"]) < tol:
                    if agent_rot == vp_rot_candidate:
                        del viewpoints[vp_key]
                        continue
                    else:
                        angle_diff = (vp_rot_candidate - agent_rot) % 360
                        # Find closest action angle (considers wrap-around automatically)
                        best_angle = min(
                            self.env.action_angles,
                            key=lambda a: min(abs(a - angle_diff), abs(a - angle_diff + 360), abs(a - angle_diff - 360)),
                        )
                        return best_angle, 0.0, vp_pos_candidate
                else:
                    vp_pos, vp_rot = vp_pos_candidate, vp_rot_candidate
                    break

        if viewpoints:
            vp_key = next(iter(viewpoints))
            vp_pos, vp_rot = self.deserialize_viewpoint(vp_key)

        path_points = self.get_shortest_path_to_point(agent_pos, vp_pos)

        # Remove already-reached path points
        while path_points and abs(path_points[0]["x"] - agent_pos["x"]) < tol and abs(path_points[0]["z"] - agent_pos["z"]) < tol:
            path_points.pop(0)

        if not path_points:
            raise ValueError("No path points left to explore")

        # Calculate direction to next target point
        target = path_points[0]
        dx = target["x"] - agent_pos["x"]
        dz = target["z"] - agent_pos["z"]

        # Calculate world-space angle to target (0° = North/+Z, 90° = East/+X)
        target_world_angle = (math.degrees(math.atan2(dx, dz)) + 360) % 360

        # Calculate relative rotation needed from current heading
        relative_angle = (target_world_angle - agent_rot + 360) % 360

        # Find closest available action angle
        best_angle = min(
            self.env.action_angles, key=lambda a: min(abs(a - relative_angle), abs(a - relative_angle + 360), abs(a - relative_angle - 360))
        )

        target_length = self.env.grid_size

        return best_angle, target_length, target

    def select_best_action(self, viewpoints, planning_steps=3, replan_each_step=True, beam_width=4, use_beam_width=True):
        """
        Plans the best sequence of actions (up to planning_steps) to maximize node discovery.
        Returns either only the first action (for replan_each_step=True) or the whole sequence.

        Works directly with action indices (0-15) from the legacy action space.
        """

        def _is_action_successful(obs_action, target_len):
            """
            Check whether a simulated action actually moved/rotated as intended.
            For movement (>0.0m), require success flag and distance close to target.
            """
            if target_len > 0.0:
                if not obs_action.info.get("move_action_success", True):
                    return False
                if obs_action.info.get("actual_dist", 0.0) < target_len - 0.05:
                    return False
            return True

        env_state = self.env.get_env_state()
        agent_state = self.env.get_agent_state()
        visibility_before = {k: n.visibility for k, n in self.env.global_sg.nodes.items()}
        node_before = [k for k, n in self.env.global_sg.nodes.items() if n.visibility >= 0.8]
        total_node_count = len(self.env.gt_graph.nodes)

        if len(node_before) == total_node_count:
            return [self.env.stop_index]

        ActionSeq = namedtuple("ActionSeq", ["seq", "score", "positions", "rotations", "viewpoints"])
        env = None
        try:
            env = ThorEnv(use_legacy_actions=True, grid_size=self.env.grid_size)
            try:
                env.reset(scene_number=self.env.scene_number)
            except TimeoutError as e:
                print(f"[TIMEOUT] Failed to reset environment in select_best_action: {e}")
                raise
            queue = deque()

            # Get target action parameters for the initial step
            try:
                target_angle, target_length, target = self.get_next_action(agent_state["position"], agent_state["rotation"], viewpoints)
            except ValueError as e:
                raise ValueError(e)

            # Get target viewpoint position for alternative action selection
            vp_key = next(iter(viewpoints))
            vp_pos, _ = self.deserialize_viewpoint(vp_key)

            valid_action_indices = self.get_valid_action_indices(
                target_angle,
                target_length,
                agent_pos=agent_state["position"],
                agent_rot=agent_state["rotation"],
                target_pos=target,
                allow_alternatives=True,
            )

            for i in valid_action_indices:
                env.restore_env_state(env_state)
                env.restore_agent_state(agent_state)
                event_before_action = env.controller.last_event
                obs_new = env.step(i)
                action_len = env.legacy_actions[i][1] if env.legacy_actions else 0.0
                action_angle = env.legacy_actions[i][0] if env.legacy_actions else 0
                if not _is_action_successful(obs_new, action_len):
                    continue

                # NEW: Validate that Step 2 (forward movement after rotation) is also possible
                # This ensures we don't train the agent to rotate into walls or dead ends
                # Only needed when the target length > 0 (i.e., we don´t need to rotate in place to get viewpoint)

                if target_length > 0.0:
                    # If we can't move forward after this action, skip it
                    # We want the agent to always be able to continue moving after any action
                    # (This applies to both movement+rotation and pure rotation actions)
                    if not self.can_move_forward(env):
                        continue

                score = self.compute_score(env, visibility_before, event_before_action, viewpoints)
                agent_pos = obs_new.info["event"].metadata["agent"]["position"]
                agent_rot = obs_new.info["event"].metadata["agent"]["rotation"]
                queue.append(ActionSeq([i], score, [agent_pos], [agent_rot], [copy.deepcopy(viewpoints)]))

                if target_length == action_len and action_angle == target_angle:
                    return [i]  # Perfect match, return immediately

            if use_beam_width:
                queue = deque(sorted(queue, key=lambda x: x.score, reverse=True)[:beam_width])
            else:
                queue = deque(sorted(queue, key=lambda x: x.score, reverse=True))

            # Planning loop
            for _ in range(1, planning_steps):
                candidates = []
                for action_seq in queue:
                    if replan_each_step:
                        viewpoints_copy = {k: v[:] for k, v in viewpoints.items()}
                    else:
                        viewpoints_copy = viewpoints

                    try:
                        target_angle, target_length, target_pos = self.get_next_action(
                            action_seq.positions[-1], action_seq.rotations[-1], viewpoints_copy
                        )
                    except ValueError as e:
                        raise ValueError(e)

                    # Get target viewpoint position for alternative action selection
                    vp_key_loop = next(iter(viewpoints_copy))
                    vp_pos_loop, _ = self.deserialize_viewpoint(vp_key_loop)

                    valid_action_indices = self.get_valid_action_indices(
                        target_angle,
                        target_length,
                        agent_pos=action_seq.positions[-1],
                        agent_rot=action_seq.rotations[-1],
                        target_pos=target_pos,
                        allow_alternatives=True,
                    )
                    for action in valid_action_indices:
                        env.restore_env_state(env_state)
                        env.restore_agent_state(agent_state)

                        visibility_before = {k: n.visibility for k, n in env.global_sg.nodes.items()}
                        step_score = 0
                        failed = False
                        # Replay the sequence
                        for i, a in enumerate(action_seq.seq):
                            event_before = env.controller.last_event
                            action_len = env.legacy_actions[a][1] if env.legacy_actions else 0.0
                            replay_obs = env.step(a)
                            if not _is_action_successful(replay_obs, action_len):
                                failed = True
                                break
                            env.agent_state["position"] = action_seq.positions[i]
                            partial_score = self.compute_score(env, visibility_before, event_before, action_seq.viewpoints[i])
                            step_score += partial_score
                            visibility_before = {k: n.visibility for k, n in env.global_sg.nodes.items()}

                        if failed:
                            continue

                        event_before = env.controller.last_event
                        obs = env.step(action)
                        action_len = env.legacy_actions[action][1] if env.legacy_actions else 0.0
                        action_angle = env.legacy_actions[action][0] if env.legacy_actions else 0
                        if not _is_action_successful(obs, action_len):
                            continue

                        if target_length > 0.0 and not self.can_move_forward(env):
                            continue

                        current_nodes = [k for k, n in self.env.global_sg.nodes.items() if n.visibility >= 0.8]
                        if len(current_nodes) == total_node_count:
                            full_seq = action_seq.seq + [action, self.env.stop_index]
                            return full_seq

                        final_score = self.compute_score(env, visibility_before, event_before, viewpoints_copy)
                        total_score = step_score + final_score
                        if target_length == 0 and action_angle == target_angle:
                            total_score += 20.0  # Bonus for matching target angle
                        combined_pos = action_seq.positions + [obs.info["event"].metadata["agent"]["position"]]
                        combined_rot = action_seq.rotations + [obs.info["event"].metadata["agent"]["rotation"]]
                        combined_vp = action_seq.viewpoints + [copy.deepcopy(viewpoints_copy)]
                        candidates.append(
                            ActionSeq(action_seq.seq + [action], total_score, combined_pos, combined_rot, combined_vp)
                        )

                if use_beam_width:
                    queue = deque(sorted(candidates, key=lambda x: x.score, reverse=True)[:beam_width])
                else:
                    queue = deque(sorted(candidates, key=lambda x: x.score, reverse=True))

            if queue:
                max_score = queue[0].score
                best_seq_indices = [i for i, seq in enumerate(queue) if seq.score == max_score]
                best_seq_idx = random.choice(best_seq_indices)
                best_seq = queue[best_seq_idx]
            else:
                best_seq = ActionSeq([random.choice(valid_action_indices)], 0, [], [], [])

            if replan_each_step:
                return [best_seq.seq[0]]
            else:
                return best_seq.seq
        finally:
            if env is not None:
                env.close()

    @staticmethod
    def can_move_forward(env):
        """Check if the agent can move forward after performing a rotation.
        This is done by simulating a rotation, then testing a forward move.
        The environment state is restored afterwards.
        """
        # Save state after Step 1
        state_after_step1 = env.get_env_state()
        agent_state_after_step1 = env.get_agent_state()

        # Try a forward movement in the new direction (Step 2)
        test_move = env.safe_step(action="MoveAhead", moveMagnitude=env.grid_size)
        can_move_forward = test_move.metadata["lastActionSuccess"]

        # Restore state back to after Step 1 (undo the test move)
        env.restore_env_state(state_after_step1)
        env.restore_agent_state(agent_state_after_step1)
        return can_move_forward

    @staticmethod
    def extract_navmesh_positions_from_error(error_msg):
        """
        Extracts the 'closest navmesh positions' for both start and target from an error message.
        Returns two dicts (start, target) or (None, None) if parsing fails.
        """
        pattern = r"closest navmesh position \((-?\d+\.\d+), [\d\.]+, (-?\d+\.\d+)\)"
        matches = re.findall(pattern, error_msg)

        if len(matches) >= 2:
            start_pos = {"x": float(matches[0][0]), "y": 0.900999, "z": float(matches[0][1])}
            target_pos = {"x": float(matches[1][0]), "y": 0.900999, "z": float(matches[1][1])}
            return start_pos, target_pos
        return None, None

    def get_shortest_path_to_point(self, initial_position, target_position, tolerance=0.2):
        if "y" not in initial_position:
            initial_position = {**initial_position, "y": 0.900999}
        if "y" not in target_position:
            target_position = {**target_position, "y": 0.900999}

        try:
            event = self.env.controller.step(
                action="GetShortestPathToPoint", position=initial_position, target=target_position, raise_for_failure=True
            )
            return event.metadata["actionReturn"]["corners"]
        except Exception as e:
            # Try parsing fallback positions
            error_msg = str(e)
            snapped_start, snapped_target = self.extract_navmesh_positions_from_error(error_msg)
            if snapped_start is None or snapped_target is None:
                raise ValueError(f"Path failed and no usable navmesh correction found: {error_msg}")

            dx_start = abs(snapped_start["x"] - initial_position["x"])
            dz_start = abs(snapped_start["z"] - initial_position["z"])
            dx_target = abs(snapped_target["x"] - target_position["x"])
            dz_target = abs(snapped_target["z"] - target_position["z"])

            if dx_start <= tolerance and dz_start <= tolerance and dx_target <= tolerance and dz_target <= tolerance:
                retry_event = self.env.controller.step(
                    action="GetShortestPathToPoint", position=snapped_start, target=snapped_target, raise_for_failure=True
                )
                return retry_event.metadata["actionReturn"]["corners"]
            else:
                raise ValueError(
                    f"Navmesh snap too far from original positions. dx_start={dx_start}, dz_start={dz_start}, "
                    f"dx_target={dx_target}, dz_target={dz_target}. Error was: {error_msg}"
                )

    def get_valid_action_indices(
        self, target_angle, target_length, angle_tolerance=22.5, agent_pos=None, agent_rot=None, target_pos=None, allow_alternatives=False
    ):
        """
        Get all action indices that match the desired movement direction and length.

        NEW (with Move→Rotate action order):
        Uses 2-step lookahead to evaluate actions:
        - Step 1: Move in CURRENT direction (or stand), then rotate
        - Step 2: Move in NEW direction (after rotation)
        - Score: How close are we to target after BOTH steps?

        Args:
            target_angle: Desired rotation angle (0-315 in 45° steps) - USED FOR LOOKAHEAD
            target_length: Desired movement distance (0.0m or 0.3m)
            angle_tolerance: Tolerance in degrees for matching angles (default 22.5° = half-step)
            agent_pos: Current agent position (dict with 'x', 'z'). Required if allow_alternatives=True
            agent_rot: Current agent rotation (dict with 'y' or int). Required if allow_alternatives=True
            target_pos: Target position (dict with 'x', 'z'). Required if allow_alternatives=True
            allow_alternatives: If True, use 2-step lookahead to find actions that lead closer to target.
            When target_length > 0, consider both stand (0.0m) and move (grid_size) to decide which is better.

        Returns:
            List of action indices that match the criteria, sorted by 2-step lookahead quality
        """
        actions = self.env.get_actions()
        valid_indices = []

        # Extract rotation value
        if isinstance(agent_rot, dict):
            rot_current = agent_rot["y"]
        else:
            rot_current = agent_rot

        # Calculate current distance to target
        current_dist = None
        if allow_alternatives and agent_pos is not None and target_pos is not None:
            current_dist = math.sqrt((target_pos["x"] - agent_pos["x"]) ** 2 + (target_pos["z"] - agent_pos["z"]) ** 2)

        # For each action, compute 2-step lookahead score
        action_scores = []

        for i, (angle, length) in enumerate(actions):
            if i == self.env.stop_index and (target_angle != 0 or target_length != 0.0):
                continue  # Skip STOP unless specifically requested

            if not allow_alternatives:
                # Check if length matches (still useful as a filter)
                if abs(length - target_length) > 0.01:
                    continue
                # Old simple matching logic (for backward compatibility)
                angle_diff = min(abs(angle - target_angle), abs(angle - target_angle + 360), abs(angle - target_angle - 360))
                if angle_diff <= angle_tolerance:
                    valid_indices.append(i)
            else:
                if target_length == 0.0 and length != 0.0:
                    continue
                # NEW: 2-step lookahead with Move→Rotate order
                # Step 1: Move in CURRENT direction (rot_current), then rotate by 'angle'
                dx_step1 = length * math.sin(math.radians(rot_current))
                dz_step1 = length * math.cos(math.radians(rot_current))
                pos_after_step1_x = agent_pos["x"] + dx_step1
                pos_after_step1_z = agent_pos["z"] + dz_step1
                rot_after_step1 = (rot_current + angle) % 360

                # Step 2: Simulate next move in NEW direction (rot_after_step1)
                # Assume we'll move forward by grid_size (0.3m) in the next step
                next_move_dist = self.env.grid_size
                dx_step2 = next_move_dist * math.sin(math.radians(rot_after_step1))
                dz_step2 = next_move_dist * math.cos(math.radians(rot_after_step1))
                pos_after_step2_x = pos_after_step1_x + dx_step2
                pos_after_step2_z = pos_after_step1_z + dz_step2

                # Calculate distance to target after both steps
                dist_after_2_steps = math.sqrt((target_pos["x"] - pos_after_step2_x) ** 2 + (target_pos["z"] - pos_after_step2_z) ** 2)

                # Score: Negative distance (higher is better → smaller distance)
                # Add bonus for actions that match the target_angle (preferred direction)
                angle_diff = min(abs(angle - target_angle), abs(angle - target_angle + 360), abs(angle - target_angle - 360))
                angle_bonus = -angle_diff / 360.0  # Small bonus for matching preferred direction

                score = -dist_after_2_steps + angle_bonus

                if target_length == 0.0:
                    # At the target position: allow rotations even if the next forward move would increase distance.
                    action_scores.append((i, score))
                else:
                    # Only consider actions that lead closer to target (after 2 steps) than current position
                    if dist_after_2_steps < current_dist + 0.1:  # Allow small tolerance
                        action_scores.append((i, score))

        if allow_alternatives:
            # Sort by score (best first) and return indices
            action_scores.sort(key=lambda x: x[1], reverse=True)
            valid_indices = [idx for idx, _ in action_scores]

        return valid_indices if valid_indices else [0]  # Return STOP if no valid actions

    @classmethod
    def deserialize_viewpoint(cls, s: str):
        try:
            dict_part, rotation = s.split("_")
            pos_dict = ast.literal_eval(dict_part)
            return pos_dict, int(rotation)
        except Exception as e:
            raise ValueError(f"Failed to deserialize viewpoint: {s} ({e})")
