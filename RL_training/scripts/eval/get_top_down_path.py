#!/usr/bin/env python3
"""
Generate top-down path visualizations for trained agents.

Usage:
    python RL_training/get_top_down_path.py

Supports the new unified config format (configs/scenario*.json) and
loads checkpoints from RL_training/runs/{scenario}_seed_{seed}_{timestamp}/checkpoints/
"""

import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from components.agents.ppo_agent import PPOAgent
from components.agents.reinforce_agent import ReinforceAgent
from components.environments.precomputed_thor_env import PrecomputedThorEnv
from components.environments.thor_env import ThorEnv
from components.environments.top_down_mapper import OrthoTopDownMapper


# ---- Drawing Utilities -------------------------------------------------------


def _blend_line(img_bgr, p1, p2, color, thickness=3, alpha=0.75):
    """Draw one anti-aliased line segment with alpha blending."""
    overlay = img_bgr.copy()
    cv2.line(overlay, p1, p2, color, thickness, lineType=cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, img_bgr, 1 - alpha, 0, dst=img_bgr)


def _draw_styled_polyline(img_bgr, pts, color, thickness=3, alpha=0.75, pattern="solid", dash_len=12, gap_len=8, dot_len=2):
    """
    Draw a polyline using a style: 'solid' | 'dashed' | 'dotted' | 'dashdot'.
    """
    if len(pts) < 2:
        return

    for (x1, y1), (x2, y2) in zip(pts[:-1], pts[1:]):
        vx, vy = x2 - x1, y2 - y1
        seg_len = int(np.hypot(vx, vy))
        if seg_len == 0:
            continue
        ux, uy = vx / seg_len, vy / seg_len

        if pattern == "solid":
            _blend_line(img_bgr, (x1, y1), (x2, y2), color, thickness, alpha)
            continue

        pos = 0
        cycle = None
        i = 0
        if pattern == "dashdot":
            cycle = [(dash_len, True), (gap_len, False), (dot_len, True), (gap_len, False)]

        while pos < seg_len:
            if pattern == "dashed":
                L_on, L_off = dash_len, gap_len
                on = True
            elif pattern == "dotted":
                L_on, L_off = dot_len, gap_len
                on = True
            elif pattern == "dashdot":
                L_on, on = cycle[i % 4]
                i += 1
                L_off = 0
            else:
                L_on, L_off, on = dash_len, gap_len, True

            L = min(L_on, seg_len - pos)
            xs = int(round(x1 + ux * pos))
            ys = int(round(y1 + uy * pos))
            xe = int(round(x1 + ux * (pos + L)))
            ye = int(round(y1 + uy * (pos + L)))
            if on:
                _blend_line(img_bgr, (xs, ys), (xe, ye), color, thickness, alpha)
            pos += L if pattern == "dashdot" else (L_on + L_off)


def draw_paths_on_topdown_multi(
    img_rgb, paths, mapper, colors, thickness=3, alpha=0.75, dash_len=14, gap_len=10, dot_len=2, start_marker=True, goal_marker_x=True
):
    """Draw multiple trajectories on one top-down image."""
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR).copy()

    color_map = {name: colors[i % len(colors)] for i, name in enumerate(paths.keys())} if isinstance(colors, list) else colors

    agent_names = list(paths.keys())
    n_agents = len(agent_names)

    if n_agents <= 2:
        styles = ["solid"] * n_agents
    else:
        base_styles = ["solid", "dashed", "dotted", "dashdot"]
        styles = [base_styles[i % len(base_styles)] for i in range(n_agents)]

    if start_marker and agent_names:
        first_traj = paths[agent_names[0]]
        if first_traj:
            u, v = mapper.world_to_pixel(*first_traj[0])
            cv2.circle(img_bgr, (u, v), 6, (255, 255, 255), thickness=-1, lineType=cv2.LINE_AA)

    for i, name in enumerate(agent_names):
        traj = paths[name]
        if not traj or len(traj) < 2:
            continue
        pts = [mapper.world_to_pixel(x, z) for (x, z) in traj]
        _draw_styled_polyline(
            img_bgr,
            pts,
            color=color_map[name],
            thickness=thickness,
            alpha=alpha,
            pattern=styles[i],
            dash_len=dash_len,
            gap_len=gap_len,
            dot_len=dot_len,
        )
        if goal_marker_x:
            u, v = pts[-1]
            s = 7
            # Make the first agent's (Baseline/red) goal marker thicker
            line_thickness = 3 if i == 0 else 2
            cv2.line(img_bgr, (u - s, v - s), (u + s, v + s), color_map[name], line_thickness, cv2.LINE_AA)
            cv2.line(img_bgr, (u - s, v + s), (u + s, v - s), color_map[name], line_thickness, cv2.LINE_AA)

    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def _truncate_to_width(text, max_px, font, scale, thick):
    """Truncate text with ellipsis so that rendered width <= max_px."""
    if max_px <= 0:
        return ""
    t = text
    (w, _), _ = cv2.getTextSize(t, font, scale, thick)
    if w <= max_px:
        return t
    ell = "..."
    while len(t) > 1:
        t = t[:-1]
        (w, _), _ = cv2.getTextSize(t + ell, font, scale, thick)
        if w <= max_px:
            return t + ell
    return ell


def add_bottom_legend_multiline(
    img_rgb,
    rows,
    bg_color=(255, 255, 255),
    font=cv2.FONT_HERSHEY_SIMPLEX,
    name_scale=0.38,
    stats_scale=0.54,
    thickness=1,
    line_gap=3,
    row_gap=6,
    left_pad=10,
    right_pad=10,
    top_pad=6,
    bottom_pad=6,
    swatch_w=16,
    swatch_h=9,
    swatch_gap=6,
):
    """Add a legend panel below the image with agent names, steps, and scores in compact format."""
    H, W, _ = img_rgb.shape
    text_x0 = left_pad + swatch_w + swatch_gap
    max_text_w = max(20, W - text_x0 - right_pad)

    total_h = top_pad + bottom_pad
    line_heights, rendered = [], []

    for name, steps, score, color in rows:
        clean_name = str(name).replace("_", " ")
        # Combine everything into one line: "Name (steps | score)"
        full_txt = f"{clean_name} ({steps} | {score:.2f})"
        full_txt = _truncate_to_width(full_txt, max_text_w, font, name_scale, thickness)

        (w, h), _ = cv2.getTextSize(full_txt, font, name_scale, thickness)

        # Height for single line
        entry_h = max(swatch_h + 4, h)
        line_heights.append(entry_h)
        total_h += entry_h + row_gap
        rendered.append((full_txt, color))

    total_h -= row_gap
    panel = np.full((total_h, W, 3), bg_color, dtype=np.uint8)

    y = top_pad
    for (full_txt, color), entry_h in zip(rendered, line_heights):
        cv2.rectangle(panel, (left_pad, y + 2), (left_pad + swatch_w, y + 2 + swatch_h), color, -1)

        (w, h), _ = cv2.getTextSize(full_txt, font, name_scale, thickness)
        text_y = y + h + 1

        cv2.putText(panel, full_txt, (text_x0, text_y), font, name_scale, (20, 20, 20), thickness, cv2.LINE_AA)

        y += entry_h + row_gap

    out_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    out_bgr = np.vstack([out_bgr, panel])
    return cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)


# ---- Config and Agent Loading ------------------------------------------------


def load_config(config_path: str) -> dict:
    """Load a unified config JSON file."""
    with open(config_path, "r") as f:
        return json.load(f)


def find_best_checkpoint(runs_dir: str, scenario_name: str) -> str | None:
    """
    Find the best checkpoint for a scenario.
    Looks for final_model.pth or best_model.pth in the runs directory.
    """
    runs_path = Path(runs_dir)
    if not runs_path.exists():
        return None

    # Find all runs matching this scenario
    matching_runs = []
    for run_dir in runs_path.iterdir():
        if run_dir.is_dir() and run_dir.name.startswith(scenario_name + "_seed_"):
            checkpoint_dir = run_dir / "checkpoints"
            if checkpoint_dir.exists():
                # Prefer best_model.pth, fallback to final_model.pth
                best_model = checkpoint_dir / "best_model.pth"
                final_model = checkpoint_dir / "final_model.pth"
                if best_model.exists():
                    matching_runs.append((run_dir, best_model))
                elif final_model.exists():
                    matching_runs.append((run_dir, final_model))

    if not matching_runs:
        return None

    # Return the most recent run (sorted by timestamp in name)
    matching_runs.sort(key=lambda x: x[0].name, reverse=True)
    return str(matching_runs[0][1])


def create_agent(config: dict, env, device: str):
    """Create an agent based on the config."""
    agent_config = config.get("agent", {})
    navigation_config = config.get("navigation", {})

    agent_name = agent_config.get("name", "").lower()

    if agent_name == "ppo":
        return PPOAgent(env=env, navigation_config=navigation_config, agent_config=agent_config, device=device)
    elif agent_name == "reinforce":
        return ReinforceAgent(env=env, navigation_config=navigation_config, agent_config=agent_config, device=device)
    else:
        raise ValueError(f"Unknown agent type: {agent_name}")


# ---- Trajectory Collection ---------------------------------------------------


def get_trajectory(agent, env, scene_number, start_pos, start_rot):
    """Run a single episode and return the trajectory of (x, z) positions."""
    # Reset agent's internal state before each episode
    agent.reset()

    obs = env.reset(scene_number=scene_number, random_start=False, start_position=start_pos, start_rotation=start_rot)

    traj_xz = [(start_pos["x"], start_pos["z"])]

    while not (obs.terminated or obs.truncated):
        with torch.no_grad():
            result = agent.get_action(obs)
            action = result[0]
        obs = env.step(action)
        pos = obs.info.get("agent_pos", None)
        if pos:
            traj_xz.append(pos)

    return traj_xz, obs.info.get("score", 0.0)


def get_initial_top_down_image(scene_number, start_pos, start_rot):
    """Get the initial top-down image from the environment."""
    env = ThorEnv()
    obs = env.reset(scene_number=scene_number, random_start=True)
    start_pos["y"] = obs.info["event"].metadata["agent"]["position"]["y"]
    obs = env.reset(scene_number=scene_number, start_position=start_pos, start_rotation=start_rot)
    image = env.get_top_down_view()
    center_x, center_z, ortho_size = env.td_center_x, env.td_center_z, env.td_ortho_size
    env.close()
    image_height, image_width = image.shape[:2]
    return image, image_height, image_width, center_x, center_z, ortho_size


# ---- Main Pipeline -----------------------------------------------------------


def get_top_down_paths(
    agent_configs: dict[str, tuple[str, str]],
    scene_numbers: list[int],
    num_starts_per_scene: int = 3,
    out_dir: str = None,
    runs_dir: str = "RL_training/runs",
):
    """
    Generate top-down path visualizations for multiple agents.

    Args:
        agent_configs: Dict mapping display_name -> (config_path, checkpoint_path or None)
                      If checkpoint_path is None, will auto-find from runs_dir
        scene_numbers: List of scene numbers to visualize
        num_starts_per_scene: Number of random starting positions per scene
        out_dir: Output directory for images
        runs_dir: Directory containing training runs (for auto-finding checkpoints)
    """
    out_dir = out_dir or "RL_training/topdown_visualizations"
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    # BGR colors for drawing
    base_colors = [
        (0, 0, 255),  # red
        (0, 255, 0),  # green
        (255, 0, 0),  # blue
        (0, 255, 255),  # yellow
        (255, 0, 255),  # magenta
        (255, 255, 0),  # cyan
    ]

    # Load configs and find checkpoints
    agents_data = {}
    for display_name, (config_path, checkpoint_path) in agent_configs.items():
        config = load_config(config_path)
        scenario_name = Path(config_path).stem

        if checkpoint_path is None:
            checkpoint_path = find_best_checkpoint(runs_dir, scenario_name)
            if checkpoint_path is None:
                print(f"[WARN] No checkpoint found for {display_name} ({scenario_name}), skipping")
                continue

        agents_data[display_name] = {"config": config, "checkpoint": checkpoint_path, "scenario": scenario_name}
        print(f"[INFO] {display_name}: {checkpoint_path}")

    if not agents_data:
        print("[ERROR] No agents with valid checkpoints found")
        return

    # Create shared environments for all scenes (to avoid LMDB issues)
    # Create one environment for getting random starts
    first_config = next(iter(agents_data.values()))["config"]
    first_env_config = first_config.get("env", {})
    temp_env = PrecomputedThorEnv(
        use_legacy_actions=first_env_config.get("use_legacy_actions", False), curriculum_stage=first_env_config.get("curriculum_stage", 1)
    )

    # Create separate environment for each agent (don't share to avoid state issues)
    shared_envs_cache = {}
    agents_cache = {}

    for display_name, data in agents_data.items():
        config = data["config"]
        checkpoint = data["checkpoint"]
        env_config = config.get("env", {})
        max_actions = env_config.get("max_actions", 40)
        use_legacy = env_config.get("use_legacy_actions", False)
        curriculum = env_config.get("curriculum_stage", 1)
        action_space_mode = env_config.get("action_space_mode", "multi_head")

        # Use display_name as key to ensure each agent gets its own environment
        env = PrecomputedThorEnv(
            use_legacy_actions=use_legacy, curriculum_stage=curriculum, max_actions=max_actions, action_space_mode=action_space_mode
        )
        shared_envs_cache[display_name] = env

        # Create agent once and cache it
        agent = create_agent(config, env, device)
        agent.load_weights(model_path=checkpoint, device=device)
        agent.eval()
        agents_cache[display_name] = agent

        print(f"[INFO] Created agent {display_name}: {type(agent).__name__}")

    try:
        for scene_number in scene_numbers:
            for start_idx in range(num_starts_per_scene):
                print(f"[INFO] Scene {scene_number}, Start {start_idx + 1}/{num_starts_per_scene}")

                # Retry logic: first retry same position (agents are non-deterministic), then try new position
                max_position_retries = 10
                max_same_position_retries = 5

                for position_retry in range(max_position_retries):
                    # Get a random start position
                    obs = temp_env.reset(scene_number=scene_number, random_start=True)
                    start_pos = obs.info["event"].metadata["agent"]["position"]
                    start_rot = obs.info["event"].metadata["agent"]["rotation"]

                    # Get top-down image
                    top_down_img, img_h, img_w, cx, cz, ortho_size = get_initial_top_down_image(scene_number, start_pos, start_rot)
                    mapper = OrthoTopDownMapper(cx, cz, ortho_size, img_h, img_w)

                    # Try same position multiple times (agents are non-deterministic)
                    has_problem = True
                    for same_pos_retry in range(max_same_position_retries):
                        # Run all agents
                        paths = {}
                        colors = {}
                        scores = {}
                        has_truncated = False

                        for idx, (display_name, data) in enumerate(agents_data.items()):
                            env_config = data["config"].get("env", {})
                            max_actions = env_config.get("max_actions", 40)

                            # Get cached agent and environment
                            agent = agents_cache[display_name]
                            env = shared_envs_cache[display_name]

                            # Run agent once
                            traj_xz, score = get_trajectory(agent, env, scene_number, start_pos, start_rot)

                            paths[display_name] = traj_xz
                            colors[display_name] = base_colors[idx % len(base_colors)]
                            scores[display_name] = score

                            # Check if trajectory was truncated (trajectory length includes start position)
                            if len(traj_xz) > max_actions:
                                has_truncated = True

                            print(f"    {display_name}: {len(traj_xz)} steps, score={score:.2f}")

                        # Check if Baseline and PPO SH-16 have identical scores (indicates both agents are stuck/spinning)
                        has_identical_scores = False
                        agent_names_list = list(agents_data.keys())
                        if len(agent_names_list) >= 2:
                            baseline_score = scores.get(agent_names_list[0], -1)
                            ppo_sh16_score = scores.get(agent_names_list[1], -1)
                            # Skip if scores are identical (regardless of value)
                            if abs(baseline_score - ppo_sh16_score) < 0.01:
                                has_identical_scores = True
                                print(
                                    f"    [SKIP] Baseline and {agent_names_list[1]} have identical scores ({baseline_score:.2f}) - both likely stuck"
                                )

                        # Check if we should retry with same position
                        has_problem = has_truncated or has_identical_scores

                        if not has_problem:
                            break  # Success, exit same_position_retry loop
                        else:
                            if same_pos_retry < max_same_position_retries - 1:
                                reason = "truncation" if has_truncated else "identical scores"
                                print(
                                    f"    [RETRY] Issue detected ({reason}), retrying same position (attempt {same_pos_retry + 2}/{max_same_position_retries})"
                                )

                    # If no problem after retries with same position, we're done
                    if not has_problem:
                        break
                    else:
                        if position_retry < max_position_retries - 1:
                            print(
                                f"    [RETRY] Still issues after {max_same_position_retries} retries, trying new position (attempt {position_retry + 2}/{max_position_retries})"
                            )
                        else:
                            print(f"    [WARN] Could not find good start after {max_position_retries} positions, using current one")

                # Draw all paths
                combined = draw_paths_on_topdown_multi(top_down_img, paths, mapper, colors)

                # Add legend
                legend_rows = [(name, len(paths[name]) - 1, scores[name], colors[name]) for name in paths.keys()]
                combined_with_legend = add_bottom_legend_multiline(combined, legend_rows)

                # Save
                agent_names = "_".join(agents_data.keys())
                out_path = Path(out_dir) / agent_names / f"scene{scene_number}_start{start_idx}.png"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out_path), cv2.cvtColor(combined_with_legend, cv2.COLOR_RGB2BGR))
                print(f"    Saved: {out_path}")

    finally:
        # Close all environments at the very end
        temp_env.close()
        for env in shared_envs_cache.values():
            env.close()


# ---- CLI ---------------------------------------------------------------------


def set_working_directory():
    """Set working directory to project root."""
    desired_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
    current_directory = os.getcwd()

    if current_directory != desired_directory:
        os.chdir(desired_directory)
        sys.path.append(desired_directory)
        print(f"Working directory changed to: {desired_directory}")


if __name__ == "__main__":
    set_working_directory()

    # Define which agents to visualize
    # Format: display_name -> (config_path, checkpoint_path or None for auto-find)
    agent_configs = {
        "Baseline": ("configs/baseline_scenario_reinforce_nodepth_il.json", None),
        "PPO SH-16": ("configs/scenario2_ppo_legacy.json", None),
        "PPO SH-504": ("configs/scenario6_ppo_singlehead_no_curriculum.json", None),
        "PPO MH": ("configs/scenario3_ppo_multihead_no_curriculum.json", None),
    }

    # Eval scenes (held-out)
    eval_scenes = [28, 29, 30]

    get_top_down_paths(
        agent_configs=agent_configs, scene_numbers=eval_scenes, num_starts_per_scene=15, out_dir="RL_training/topdown_visualizations"
    )
