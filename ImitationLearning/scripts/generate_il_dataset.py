import copy
import json
import os
import pickle
import random
import warnings
from math import hypot
from typing import Dict, List

import matplotlib.pyplot as plt
from tqdm import tqdm

from ImitationLearning.labeling.imitation_labeler import ImitationLabeler
from components.environments.thor_env import ThorEnv
from components.utils.aco_tsp import SolveTSPUsingACO


def get_unique_starts(env, k=5):
    reachable = env.controller.step("GetReachablePositions").metadata["actionReturn"]
    random.shuffle(reachable)
    unique = set()
    starts = []

    for pos in reachable:
        pos_key = (round(pos["x"], 2), round(pos["z"], 2))
        if pos_key in unique:
            continue

        rot = {"x": 0, "y": random.choice([0, 90, 180, 270]), "z": 0}
        success, real_pos, real_rot = is_valid_start(env.controller, pos, rot)
        if success:
            rounded_pos_key = (round(real_pos["x"], 2), round(real_pos["z"], 2))
            if rounded_pos_key not in unique:
                unique.add(rounded_pos_key)
                starts.append((real_pos, real_rot))

        if len(starts) >= k:
            break

    return starts


def is_valid_start(controller, position, rotation):
    controller.step(action="Teleport", position=position, rotation=rotation, forceAction=True)
    controller.step("Pass")

    directions = ["MoveAhead", "MoveLeft", "MoveRight", "MoveBack"]
    for action in directions:
        result = controller.step(action)
        controller.step("Pass")
        if result.metadata.get("lastActionSuccess", False):
            agent_pos = result.metadata["agent"]["position"]
            agent_rot = result.metadata["agent"]["rotation"]
            return True, agent_pos, agent_rot

    return False, None, None


def aggregate_visibility(global_vis: float, local_vis: float, alpha: float = 0.8) -> float:
    return 1 - (1 - global_vis) * (1 - alpha * local_vis)


_VIEWPOINT_CACHE = {}


def load_viewpoint_to_objects(scene_id: int):
    """
    Load viewpoint_to_objects for a given scene without bloating GTGraph in memory.
    Caches per scene to avoid repeated disk IO.
    """
    if scene_id in _VIEWPOINT_CACHE:
        return _VIEWPOINT_CACHE[scene_id]

    floorplan = f"FloorPlan{scene_id}"
    path = os.path.join(os.path.dirname(__file__), "..", "..", "components", "data", "gt_graphs", f"{floorplan}.json")
    with open(path, "r") as f:
        data = json.load(f)

    v2o = data.get("viewpoint_to_objects", {})
    _VIEWPOINT_CACHE[scene_id] = v2o
    return v2o


def compute_minimal_viewpoint_cover(
    viewpoint_to_objects: Dict[str, List[Dict[str, float]]], threshold: float = 0.8, alpha: float = 0.8
) -> Dict[str, List[str]]:
    reachable_viewpoints = {}
    for vp, obj_list in viewpoint_to_objects.items():
        try:
            _, rot = ImitationLabeler.deserialize_viewpoint(vp)
        except ValueError:
            warnings.warn(f"Skipping malformed viewpoint key: {vp}")
            continue

        if rot % 45 == 0:
            reachable_viewpoints[vp] = obj_list

    if not reachable_viewpoints:
        raise ValueError("No reachable viewpoints with 45° increments found in viewpoint_to_objects.")

    viewpoint_to_objects = reachable_viewpoints

    # All unique object IDs
    all_objects = set()
    for obj_list in viewpoint_to_objects.values():
        for obj_dict in obj_list:
            all_objects.update(obj_dict.keys())

    # Initialize visibility for all objects
    visibility = {obj_id: 0.0 for obj_id in all_objects}
    selected_viewpoints = {}
    remaining_viewpoints = set(viewpoint_to_objects.keys())

    while any(v < threshold for v in visibility.values()):
        best_vp = None
        best_gain = 0
        best_new_vis = None

        for vp in remaining_viewpoints:
            temp_vis = copy.deepcopy(visibility)
            gain = 0.0
            for obj_dict in viewpoint_to_objects[vp]:
                for obj_id, local_vis in obj_dict.items():
                    if temp_vis[obj_id] >= threshold:
                        continue
                    updated_vis = aggregate_visibility(temp_vis[obj_id], local_vis, alpha)
                    gain += max(0.0, updated_vis - temp_vis[obj_id])
                    temp_vis[obj_id] = updated_vis

            if gain > best_gain:
                best_gain = gain
                best_vp = vp
                best_new_vis = temp_vis

        if best_vp is None:
            # No improvement possible anymore
            break

        visibility = best_new_vis
        selected_viewpoints[best_vp] = [list(d.keys())[0] for d in viewpoint_to_objects[best_vp]]
        remaining_viewpoints.remove(best_vp)

    return selected_viewpoints


def update_viewpoints(env, viewpoints):
    """
    Remove all objects from each viewpoint that are already seen in the global scene graph.
    Optionally, remove viewpoints with empty object lists.
    """
    seen = set([k for k, n in env.global_sg.nodes.items() if n.visibility >= 0.8])
    to_delete = []
    for vp, objs in viewpoints.items():
        filtered = [obj for obj in objs if obj not in seen]
        if filtered:
            viewpoints[vp] = filtered
        else:
            to_delete.append(vp)
    for vp in to_delete:
        del viewpoints[vp]


def get_shortest_viewpoint_path(start_x, start_z, viewpoints, use_aco=True):
    """
    Computes a short tour through all viewpoints using either ACO or greedy heuristic.
    The starting position (start_x, start_z) is NOT included in the TSP computation.
    Instead, the resulting path is rotated so that it starts at the viewpoint closest to the start position.
    """
    vp_keys = list(viewpoints.keys())
    vp_positions = {vp: ImitationLabeler.deserialize_viewpoint(vp)[0] for vp in vp_keys}
    vp_coords = {vp: (pos["x"], pos["z"]) for vp, pos in vp_positions.items()}

    if use_aco:
        points = [vp_coords[vp] for vp in vp_keys]
        tsp_solver = SolveTSPUsingACO(
            mode="MaxMin", colony_size=max(10, len(vp_keys)), steps=max(200, 20 * len(vp_keys)), nodes=points, labels=vp_keys
        )
        _, _ = tsp_solver.run()
        tour = tsp_solver.global_best_tour
        ordered_vps = [vp_keys[i] for i in tour]
    else:
        # Greedy TSP
        unvisited = set(vp_keys)
        ordered_vps = []
        curr_vp = vp_keys[0]  # arbitrary start
        curr_x, curr_z = vp_coords[curr_vp]
        ordered_vps.append(curr_vp)
        unvisited.remove(curr_vp)

        while unvisited:
            next_vp = min(unvisited, key=lambda vp: hypot(vp_coords[vp][0] - curr_x, vp_coords[vp][1] - curr_z))
            ordered_vps.append(next_vp)
            curr_x, curr_z = vp_coords[next_vp]
            unvisited.remove(next_vp)

    # Rotate path to start from viewpoint nearest to (start_x, start_z)
    distances_to_start = [hypot(vp_coords[vp][0] - start_x, vp_coords[vp][1] - start_z) for vp in ordered_vps]
    closest_idx = distances_to_start.index(min(distances_to_start))
    rotated_path = ordered_vps[closest_idx:] + ordered_vps[:closest_idx]

    return rotated_path


def has_valid_path(controller, start, target):
    """
    Checks whether a path exists from `start` to `target` by calling AI2-THOR's internal path planner.
    Returns True only if the path is valid and both endpoints are close enough to the NavMesh.
    """

    if "y" not in start:
        start = {**start, "y": 0.900999}
    if "y" not in target:
        target = {**target, "y": 0.900999}

    try:
        event = controller.step(action="GetShortestPathToPoint", position=start, target=target, raise_for_failure=True)
        return True
    except Exception as e:
        return False


def count_existing_trajectories(base_path, scene_id):
    """Count existing trajectory files for a given scene."""
    scene_name = f"FloorPlan{scene_id}"
    scene_dir = os.path.join(base_path, scene_name)

    if not os.path.exists(scene_dir):
        return 0

    pkl_files = [f for f in os.listdir(scene_dir) if f.endswith(".pkl")]
    return len(pkl_files)


def print_dataset_summary(base_path, scene_ids, target_per_scene):
    """Print summary of existing dataset and what needs to be generated."""
    print("\n" + "=" * 80)
    print("IL Dataset Status")
    print("=" * 80)

    summary = []
    total_existing = 0
    total_needed = 0

    for scene_id in scene_ids:
        if scene_id == 8:  # Skip scene 8 due to AI2-THOR bug
            continue

        existing = count_existing_trajectories(base_path, scene_id)
        needed = max(0, target_per_scene - existing)
        status = "✓ Complete" if needed == 0 else f"⚠ Need {needed} more"

        summary.append({"scene_id": scene_id, "existing": existing, "needed": needed, "status": status})

        total_existing += existing
        total_needed += needed

    # Print summary table
    print(f"{'Scene':<10} {'Existing':<12} {'Target':<10} {'Status':<20}")
    print("-" * 80)

    for item in summary:
        print(f"FloorPlan{item['scene_id']:<3} {item['existing']:<12} {target_per_scene:<10} {item['status']:<20}")

    print("-" * 80)
    print(
        f"Total:     {total_existing:<12} {len(summary)*target_per_scene:<10} "
        f"{'Need ' + str(total_needed) + ' more' if total_needed > 0 else '✓ Complete'}"
    )
    print("=" * 80 + "\n")

    return summary


def generate_dataset(
    max_steps=80, planning_steps=2, num_starts=10, max_stagnation=30, scene_numbers=None, visualize_path=False, debug=False
):
    base_path = os.path.join(os.path.dirname(__file__), "..", "..", "components", "data", "il_dataset")

    if scene_numbers is not None:
        scene_ids = sorted(scene_numbers)
    else:
        scene_ids = list(range(1, 31))

    # Print summary and filter scenes that need more data
    summary = print_dataset_summary(base_path, scene_ids, num_starts)

    # Filter to only scenes that need more trajectories
    scenes_to_generate = [(item["scene_id"], item["needed"]) for item in summary if item["needed"] > 0]

    if not scenes_to_generate:
        print("[INFO] All scenes have sufficient trajectories. Nothing to generate.")
        return

    print(f"[INFO] Generating trajectories for {len(scenes_to_generate)} scenes\n")

    scene_iter = tqdm(scenes_to_generate, desc="Scenes") if len(scenes_to_generate) > 1 else scenes_to_generate

    for scene_id, needed in scene_iter:
        needed = max(0, num_starts - count_existing_trajectories(base_path, scene_id))
        if needed == 0:
            tqdm.write(f"[INFO] FloorPlan{scene_id}: already complete. Skipping.")
            continue
        # Use legacy actions for IL dataset: 8 directions × 2 lengths = 16 discrete actions
        env = None
        try:
            env = ThorEnv(use_legacy_actions=True, grid_size=0.3)
            env.reset(scene_number=scene_id)
            # Generate more start positions than needed to have buffer
            all_starts = get_unique_starts(env, k=needed * 4)

            # Skip scene if not enough valid start positions found
            if len(all_starts) < 0:
                print(f"[SKIP] FloorPlan{scene_id}: No valid start positions found. " f"Likely NavMesh issue in this scene.")
                continue

        except TimeoutError as e:
            print(f"[TIMEOUT] Skipping scene {scene_id} due to timeout during initialization: {e}")
            continue
        finally:
            if env is not None:
                env.close()

        successful = 0
        attempts = 0

        with tqdm(total=needed, desc=f"Startpositions Scene {scene_id}", leave=True) as outer_bar:
            while attempts < len(all_starts):
                remaining_needed = max(0, num_starts - count_existing_trajectories(base_path, scene_id))
                outer_bar.set_postfix({"remaining": remaining_needed})
                if remaining_needed == 0:
                    tqdm.write(f"[INFO] FloorPlan{scene_id}: target reached by other workers. Stopping.")
                    break
                start_pos, start_rot = all_starts[attempts]
                attempts += 1

                # Use legacy actions for IL dataset: 8 directions × 2 lengths = 16 discrete actions
                env = None
                inner_bar = None
                try:
                    env = ThorEnv(use_legacy_actions=True, grid_size=0.3)
                    labeler = ImitationLabeler(env)
                except TimeoutError as e:
                    print(f"[TIMEOUT] Skipping start position due to timeout during environment creation: {e}")
                    if env is not None:
                        env.close()
                    continue

                try:
                    obs = env.reset(scene_number=scene_id, start_position=start_pos, start_rotation=start_rot)
                    event = obs.info["event"]
                    real_start_pos = event.metadata["agent"]["position"]
                    real_start_rot = event.metadata["agent"]["rotation"]

                    rounded_start_pos = {k: round(v, 2) for k, v in start_pos.items()}
                    rounded_real_pos = {k: round(v, 2) for k, v in real_start_pos.items()}
                    rounded_start_rot = {k: round(v, 2) for k, v in start_rot.items()}
                    rounded_real_rot = {k: round(v, 2) for k, v in real_start_rot.items()}

                    if not rounded_start_pos == rounded_real_pos or not rounded_start_rot == rounded_real_rot:
                        warnings.warn(
                            f"Start position and rotation do not match: {rounded_start_pos} vs {rounded_real_pos}, {rounded_start_rot} vs {rounded_real_rot}"
                        )

                    start_x = round(real_start_pos["x"], 2)
                    start_z = round(real_start_pos["z"], 2)
                    rot_y = round(real_start_rot["y"], 1)

                    data = []
                    steps = 0
                    last_action = -1
                    total_nodes = len(env.gt_graph.nodes)
                    stagnation_counter = 0

                    inner_bar = tqdm(total=total_nodes, desc=f"Scene {scene_id} Start {successful}", leave=False)

                    all_viewpoints = load_viewpoint_to_objects(scene_id)
                    filtered_viewpoints = all_viewpoints

                    viewpoints = compute_minimal_viewpoint_cover(filtered_viewpoints)
                    path = get_shortest_viewpoint_path(start_x, start_z, viewpoints)
                    viewpoints = {vp: viewpoints[vp] for vp in path}

                    path_images = []
                    completion_rate = 0.0  # Initialize for early stopping check
                    while not obs.terminated and steps < max_steps and stagnation_counter < max_stagnation:
                        if visualize_path:
                            if len(viewpoints) > 0:
                                path_images.append(
                                    env.visualize_shortest_path(
                                        obs.info["event"].metadata["agent"]["position"],
                                        ImitationLabeler.deserialize_viewpoint(list(viewpoints.keys())[0])[0],
                                    )
                                )
                                plt.figure(figsize=(4, 4))
                                plt.imshow(path_images[-1])
                                plt.axis("off")
                                plt.title(f"Shortest Path Visualization Step {steps}")
                                plt.show()
                        inner_bar.n = len([k for k, n in env.global_sg.nodes.items() if n.visibility >= 0.8])
                        inner_bar.set_postfix({"Steps": steps})
                        best_actions = labeler.select_best_action(viewpoints, planning_steps)
                        for action in best_actions:
                            prev_node_count = len([k for k, n in env.global_sg.nodes.items() if n.visibility >= 0.8])
                            obs = env.step(action)
                            # print()
                            # print(env.get_actions()[action])
                            current_node_count = len([k for k, n in env.global_sg.nodes.items() if n.visibility >= 0.8])

                            update_viewpoints(env, viewpoints)
                            if current_node_count > prev_node_count:
                                stagnation_counter = 0
                            else:
                                stagnation_counter += 1

                            save_obs = copy.deepcopy(obs)
                            save_obs.info["event"] = {}
                            # Persist scene graphs as dicts (state layout: [RGB, depth, local_sg, global_sg])
                            save_obs.state[2] = save_obs.state[2].to_dict()
                            save_obs.state[3] = save_obs.state[3].to_dict()
                            sample = {
                                "scene": scene_id,
                                "start_position": start_pos,
                                "start_rotation": start_rot,
                                "step": steps,
                                "obs": save_obs,
                                "last_action": last_action,
                                "num_actions": env.get_action_dim(),
                            }

                            data.append(sample)
                            last_action = action
                            steps += 1

                            # Early stopping: if 95%+ complete and stagnating for 10 steps, end episode successfully
                            completion_rate = current_node_count / total_nodes
                            early_stop_threshold = 7  # Steps of stagnation before early stopping
                            if completion_rate >= 0.95 and stagnation_counter >= early_stop_threshold:
                                # Remove the last 10 stagnating steps from the trajectory
                                data = data[:-early_stop_threshold]

                                # Add STOP action as final sample
                                # Use the state from the last valid step (after removing stagnating steps)
                                final_obs = copy.deepcopy(data[-1]["obs"])
                                final_obs.info["action"] = 0  # STOP action
                                final_sample = {
                                    "scene": scene_id,
                                    "start_position": start_pos,
                                    "start_rotation": start_rot,
                                    "step": len(data),
                                    "obs": final_obs,
                                    "last_action": data[-1]["obs"].info["action"],  # The action that led to this state
                                    "num_actions": env.get_action_dim(),
                                }
                                data.append(final_sample)

                                print(
                                    f"Early stopping at {completion_rate*100:.1f}% completion (removed last {early_stop_threshold} stagnating steps, added STOP action)"
                                )
                                obs.terminated = True
                                break

                            if obs.terminated or steps >= max_steps or stagnation_counter >= max_stagnation:
                                if steps >= max_steps:
                                    print(f"Max steps reached: {steps}")
                                if stagnation_counter >= max_stagnation:
                                    print(f"Max stagnation reached: {stagnation_counter}")
                                break

                    if obs.terminated or (completion_rate >= 0.95 and len(data) > 0):
                        # Save trajectory if successfully completed and still needed
                        remaining_needed = max(0, num_starts - count_existing_trajectories(base_path, scene_id))
                        if remaining_needed == 0:
                            tqdm.write(f"[INFO] FloorPlan{scene_id}: target reached by other workers. Stopping.")
                            break

                        # Add STOP action as final sample if not already present
                        # (Early stopping already adds it, so check if last action is not STOP)
                        if len(data) > 0 and data[-1]["obs"].info.get("action") != 0:
                            final_obs = copy.deepcopy(data[-1]["obs"])
                            final_obs.info["action"] = 0  # STOP action
                            final_sample = {
                                "scene": scene_id,
                                "start_position": start_pos,
                                "start_rotation": start_rot,
                                "step": len(data),
                                "obs": final_obs,
                                "last_action": data[-1]["obs"].info["action"],  # The action that led to this state
                                "num_actions": env.get_action_dim(),
                            }
                            data.append(final_sample)

                        scene_name = f"FloorPlan{scene_id}"
                        scene_dir = os.path.join(base_path, scene_name)
                        os.makedirs(scene_dir, exist_ok=True)
                        filename = f"{scene_name}_px_{start_x}_pz_{start_z}_ry_{rot_y}.pkl"
                        save_path = os.path.join(scene_dir, filename)

                        with open(save_path, "wb") as f:
                            pickle.dump(data, f)

                        outer_bar.update(1)
                        tqdm.write(f"Saved to {save_path}")
                        successful += 1
                        outer_bar.set_postfix({"remaining": remaining_needed})

                except (ValueError, TimeoutError) as e:
                    if debug:
                        raise  # Re-raise in debug mode for full stack trace
                    error_type = "Timeout" if isinstance(e, TimeoutError) else "ValueError"
                    print(f"Skipping start position {start_pos} / {start_rot} due to {error_type}: {e}")
                    continue
                except Exception as e:
                    print(f"[ERROR] Unexpected error during trajectory generation: {type(e).__name__}: {e}")
                    if debug:
                        raise
                    continue
                finally:
                    # Always clean up resources, even if an unexpected error occurs
                    if inner_bar is not None:
                        inner_bar.close()
                    if env is not None:
                        env.close()


if __name__ == "__main__":
    # seed = 42
    # random.seed(seed)
    # scene_numbers = [[1], [2, 3], [4, 5], [7, 9, 10, 11, 12]]
    scene_numbers = list(range(1, 28))
    # Set debug=True to disable error handling and see full stack traces
    generate_dataset(scene_numbers=scene_numbers, visualize_path=False, debug=False)
