#!/usr/bin/env python3
"""
Aggregates and visualizes training and evaluation results from multiple runs.

Supports:
- Training metrics (Rollout/* or Block/*)
- Evaluation metrics (Eval/*) from the eval/ subdirectory
- Aggregation across multiple seeds of a scenario
- CLI with sensible defaults

Usage:
    python aggregate_runs.py                           # Everything with defaults
    python aggregate_runs.py --mode eval               # Only eval plots
    python aggregate_runs.py --raw --smooth 50         # Block data with custom smoothing
"""

import argparse
import json
import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator

# ---- Constants ---------------------------------------------------------------

# Z-values for common confidence levels (replaces scipy.stats.norm.ppf)
Z_VALUES = {0.90: 1.645, 0.95: 1.960, 0.99: 2.576}

# Legend labels for scenarios (matching the paper)
# R=REINFORCE, PPO=PPO, SH=Single-Head, MH=Multi-Head, IL=Imitation Learning, D=Depth w/ Coll.-Aux, CL=Curriculum Learning
SCENARIO_LABELS = {
    "baseline_scenario_reinforce_nodepth_il": "Baseline (R+SH16+IL)",
    "scenario0_reinforce_il": "S0 (R+SH16+IL+D)",
    "scenario1_reinforce_no_il": "S1 (R+SH16)",
    "scenario2_1_ppo_legacy_no_depth": "S2 (PPO+SH16)",
    "scenario2_ppo_legacy": "S3 (PPO+SH16+D)",
    "scenario6_ppo_singlehead_no_curriculum": "S4 (PPO+SH504+D)",
    "scenario7_ppo_singlehead_curriculum": "S5 (PPO+SH504+D+CL)",
    "scenario5_ppo_multihead_no_depth": "S6 (PPO+MH504)",
    "scenario3_ppo_multihead_no_curriculum": "S7 (PPO+MH504+D)",
    "scenario8_ppo_multihead_curriculum_no_depth": "S8 (PPO+MH504+CL)",
    "scenario4_ppo_multihead_curriculum": "S9 (PPO+MH504+D+CL)",
}

# Sorting order for scenarios
SCENARIO_ORDER = [
    "baseline_scenario_reinforce_nodepth_il",
    "scenario0_reinforce_il",
    "scenario1_reinforce_no_il",
    "scenario2_1_ppo_legacy_no_depth",
    "scenario2_ppo_legacy",
    "scenario6_ppo_singlehead_no_curriculum",
    "scenario7_ppo_singlehead_curriculum",
    "scenario5_ppo_multihead_no_depth",
    "scenario3_ppo_multihead_no_curriculum",
    "scenario8_ppo_multihead_curriculum_no_depth",
    "scenario4_ppo_multihead_curriculum",
]


def _scenario_sort_key(scenario_name: str) -> int:
    """Returns sort index for a scenario."""
    try:
        return SCENARIO_ORDER.index(scenario_name)
    except ValueError:
        return 999  # Unknown scenarios at the end


def get_z_value(ci_level: float) -> float:
    """Returns the z-value for a confidence level."""
    if ci_level in Z_VALUES:
        return Z_VALUES[ci_level]
    # Fallback: linear interpolation or approximation
    # For most cases, 1.96 (95%) is sufficient
    return 1.96


# ---- Tag Definitions --------------------------------------------------------

# Block-level tags (unsmoothed - for custom smoothing)
BLOCK_TAGS = {
    "score": "Block/Mean_Score",
    "steps": "Block/Mean_Steps",
    "reward": "Block/Mean_Reward",
    "collision_rate": "Reward/collision_rate",
    "path_length": "Block/Mean_Path_Length",
    "term_trunc_ratio": "Block/Terminated_Truncated_Ratio",
}

# Rollout-window tags (already smoothed during training)
ROLLOUT_TAGS = {
    "score": "Rollout/Mean_Score",
    "steps": "Rollout/Mean_Steps",
    "reward": "Rollout/Mean_Reward",
    "collision_rate": "Reward/collision_rate",
    "steps_for_score_1": "Rollout/Steps_for_score_1",
    "path_length": "Rollout/Mean_Path_Length",
    "term_trunc_ratio": "Block/Terminated_Truncated_Ratio",
}

# Evaluation tags (from eval/ subdirectory)
EVAL_TAGS = {
    "score": "Eval/Mean_Score",
    "steps": "Eval/Mean_Steps",
    "reward": "Eval/Mean_Reward",
    "std_score": "Eval/Std_Score",
    "std_steps": "Eval/Std_Steps",
    "std_reward": "Eval/Std_Reward",
    "path_length": "Eval/Mean_Path_Length",
    "coverage": "Eval/Mean_Coverage",
    "collision_rate": "Eval/Mean_Collision_Rate",
}

# Action type tags (for computing move success rate)
# Note: total is computed as movements + pure_rotations + idle + stops
ACTION_TAGS = {
    "movements": "actions/movements",
    "pure_rotations": "actions/pure_rotations",
    "idle": "actions/idle",
    "stops": "actions/stops",
}

# Post-training evaluation tags (from post_train_eval/ subdirectory)
POST_TRAIN_EVAL_TAGS = {
    "score": "eval/score",
    "steps": "eval/steps",
    "reward": "eval/reward",
    "path_length": "eval/path_length",
    "movement_success_rate": "eval/movement_success_rate",
}


# ---- Matplotlib defaults for print-quality figures -------------------------

# Colorblind-friendly palette (extended for 10+ scenarios)
# Basis: Wong 2011, extended with additional well-distinguishable colors
COLORBLIND_COLORS = [
    "#000000",  # Black (Baseline)
    "#E69F00",  # Orange (S0)
    "#56B4E9",  # Light blue (S1)
    "#009E73",  # Green (S2 - scenario2_1)
    "#00D9A3",  # Teal (S3 - scenario2)
    "#F0E442",  # Yellow (S4 - scenario6)
    "#0072B2",  # Blue (S5 - scenario7)
    "#D55E00",  # Red-orange (S6 - scenario5)
    "#CC79A7",  # Pink (S7 - scenario3)
    "#8B4513",  # Saddle brown (S8 - scenario8)
    "#7570B3",  # Purple (S9 - scenario4)
]

# Line styles for better distinction when curves are similar
# Baseline, S0, S1 have similar curves -> use different line styles
SCENARIO_LINESTYLES = {
    "baseline_scenario_reinforce_nodepth_il": "-",  # Solid
    "scenario0_reinforce_il": "--",  # Dashed
    "scenario1_reinforce_no_il": "-.",  # Dash-dot
    "scenario2_1_ppo_legacy_no_depth": "-",  # Solid
    "scenario2_ppo_legacy": "--",  # Dashed
    "scenario6_ppo_singlehead_no_curriculum": "-",  # Solid
    "scenario7_ppo_singlehead_curriculum": "-",  # Solid
    "scenario5_ppo_multihead_no_depth": "-",  # Solid
    "scenario3_ppo_multihead_no_curriculum": "--",  # Solid
    "scenario4_ppo_multihead_curriculum": "--",  # Solid
    "scenario8_ppo_multihead_curriculum_no_depth": "-",  # Solid
}

plt.rcParams.update(
    {
        "figure.figsize": (10, 7),
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.linestyle": ":",
        "grid.alpha": 0.35,
        "axes.titleweight": "bold",
        "axes.prop_cycle": plt.cycler(color=COLORBLIND_COLORS),
        "font.size": 16,
        "axes.labelsize": 18,
        "axes.titlesize": 20,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "legend.fontsize": 15,
        "legend.frameon": False,
        "legend.handlelength": 2.5,
        "legend.handleheight": 1.2,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    }
)


# ---- Utils ------------------------------------------------------------------


def _ensure_dir(path):
    """Creates directory if it does not exist."""
    if path and not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)
    return path


def _nice_label(scenario_name: str) -> str:
    """Converts scenario names to readable labels."""
    # Use predefined label if available
    if scenario_name in SCENARIO_LABELS:
        return SCENARIO_LABELS[scenario_name]

    # Fallback: Remove prefixes like "scenario2_" or "baseline_scenario_"
    name = scenario_name
    if name.startswith("baseline_scenario_"):
        name = name[len("baseline_scenario_") :]
    elif re.match(r"scenario\d+_", name):
        name = re.sub(r"^scenario\d+_", "", name)

    # Format: underscore to space, title case
    return name.replace("_", " ").title()


def optimal_legend_ncol(n: int, max_ncol: int = None) -> int:
    """
    Calculates optimal number of columns for legend layout.

    Args:
        n: Number of legend entries
        max_ncol: Maximum number of columns (None = no limit)

    Returns:
        Optimal number of columns

    Logic:
    - 1-5 entries: all in one row (ncol = n)
    - 6+ entries: distribute evenly across rows (ncol = ceil(n/2))
    - Respects max_ncol if provided

    Examples (without max_ncol):
    - 5 entries: 1 row with 5 columns
    - 6 entries: 2 rows with 3 columns each
    - 7 entries: 2 rows with 4 and 3 columns
    - 8 entries: 2 rows with 4 columns each

    Examples (max_ncol=5):
    - 11 entries: 3 rows with 4-4-3 columns (instead of 2 rows with 6-5)
    """
    import math

    if max_ncol is None:
        max_ncol = n  # No limit

    if n <= 5:
        return min(n, max_ncol)

    optimal = math.ceil(n / 2)
    return min(optimal, max_ncol)


def moving_average(x, w, align="center", pad_mode="reflect"):
    """
    Moving average with same output length.

    Args:
        x: Input series
        w: Window size (>=1)
        align: 'center', 'left' (causal), or 'right'
        pad_mode: Padding mode ('reflect', 'edge', 'constant')

    Returns:
        np.ndarray: Smoothed series with len == len(x)
    """
    x = np.asarray(x, dtype=float)
    if w <= 1 or len(x) == 0:
        return x.copy()

    kernel = np.ones(w, dtype=float) / w

    if align == "center":
        pad_left = (w - 1) // 2
        pad_right = w - 1 - pad_left
        x_pad = np.pad(x, (pad_left, pad_right), mode=pad_mode)
        y = np.convolve(x_pad, kernel, mode="valid")
    elif align == "left":
        x_pad = np.pad(x, (w - 1, 0), mode=pad_mode)
        y = np.convolve(x_pad, kernel, mode="valid")
    elif align == "right":
        x_pad = np.pad(x, (0, w - 1), mode=pad_mode)
        y = np.convolve(x_pad, kernel, mode="valid")
    else:
        raise ValueError("align must be 'center', 'left', or 'right'.")

    return y[: len(x)]


def causal_smooth(x, w):
    """
    Causal smoothing with expanding window at the start.

    Matches the rollout-window behavior during training:
    - At the start (i < w): average over all previous values
    - From i >= w: average over the last w values

    Args:
        x: Input series
        w: Window size (>=1)

    Returns:
        np.ndarray: Smoothed series with len == len(x)
    """
    x = np.asarray(x, dtype=float)
    if w <= 1 or len(x) == 0:
        return x.copy()

    # Pandas rolling with min_periods=1 does exactly this
    return pd.Series(x).rolling(w, min_periods=1).mean().values


# Metrics that are not automatically smoothed in the rollout window
UNSMOOTHED_METRICS = {"collision_rate"}

# Default rollout window size (as in training)
DEFAULT_ROLLOUT_WINDOW = 20


# ---- Run Discovery ----------------------------------------------------------


def discover_runs(base_dir: str) -> dict[str, list[str]]:
    """
    Finds runs and groups them by scenario name.

    Expects structure: {base_dir}/{scenario}_seed_{seed}_{timestamp}/

    Args:
        base_dir: Base directory with runs

    Returns:
        dict: {scenario_name: [run_dir1, run_dir2, ...]}
    """
    runs = {}

    if not os.path.isdir(base_dir):
        print(f"[WARN] Directory not found: {base_dir}")
        return runs

    for entry in os.listdir(base_dir):
        run_dir = os.path.join(base_dir, entry)
        if not os.path.isdir(run_dir):
            continue

        # Parse "{scenario}_seed_{seed}_{timestamp}" pattern
        match = re.match(r"(.+)_seed_\d+_\d{8}_\d{6}$", entry)
        if match:
            scenario_name = match.group(1)
            if scenario_name not in runs:
                runs[scenario_name] = []
            runs[scenario_name].append(run_dir)

    # Sort run directories per scenario
    for scenario in runs:
        runs[scenario] = sorted(runs[scenario])

    return runs


def get_training_event_file(run_dir: str) -> str | None:
    """Gets the training event file from the root of the run directory."""
    for f in os.listdir(run_dir):
        if f.startswith("events.out.tfevents."):
            return os.path.join(run_dir, f)
    return None


def get_eval_event_file(run_dir: str) -> str | None:
    """Gets the eval event file from the eval/ subdirectory."""
    eval_dir = os.path.join(run_dir, "eval")
    if not os.path.isdir(eval_dir):
        return None
    for f in os.listdir(eval_dir):
        if f.startswith("events.out.tfevents."):
            return os.path.join(eval_dir, f)
    return None


# ---- TensorBoard I/O --------------------------------------------------------


def get_config_from_event(event_path: str) -> dict | None:
    """Extracts the config from a TensorBoard event file."""
    ea = event_accumulator.EventAccumulator(event_path)
    ea.Reload()
    tags = ea.Tags()
    config_tag = "full_config/text_summary"
    if config_tag not in tags.get("tensors", []):
        return None
    try:
        tensor_events = ea.Tensors(config_tag)
        raw = tensor_events[0].tensor_proto.string_val[0]
        cfg_text = raw.decode("utf-8")
        config = json.loads(cfg_text)
        return config
    except Exception:
        return None


def extract_scalar_from_event(event_path: str, tag: str) -> tuple[list, list]:
    """Loads scalar values for a tag from a TensorBoard event file."""
    ea = event_accumulator.EventAccumulator(event_path)
    ea.Reload()
    try:
        events = ea.Scalars(tag)
        steps = [e.step for e in events]
        values = [e.value for e in events]
        return steps, values
    except KeyError:
        return [], []


def list_available_tags(event_path: str) -> dict:
    """Lists all available tags in an event file."""
    ea = event_accumulator.EventAccumulator(event_path)
    ea.Reload()
    return ea.Tags()


def validate_scenario_configs(scenario_name: str, run_dirs: list[str]) -> tuple[bool, list[str]]:
    """
    Validates whether all runs of a scenario have the same configuration.

    Args:
        scenario_name: Scenario name
        run_dirs: List of run directories

    Returns:
        tuple: (is_valid, warnings) - is_valid=True if all configs match, warnings=list of warnings
    """
    if len(run_dirs) <= 1:
        return True, []

    configs = []
    for run_dir in run_dirs:
        event_file = get_training_event_file(run_dir)
        if event_file is None:
            continue
        config = get_config_from_event(event_file)
        if config is not None:
            configs.append((run_dir, config))

    if len(configs) <= 1:
        return True, []

    # Use first run as reference
    ref_dir, ref_config = configs[0]
    warnings = []
    is_valid = True

    # Keys to ignore (normally differ between runs)
    ignore_keys = {
        "seed",
        "run_name",
        "timestamp",
        "device",
        "num_workers",
        "run_id",
        "output_dir",
        "wandb_run_id",
        "wandb_run_name",
        # Parameters that can differ due to config changes
        "agent.adaptive_entropy",
        "agent.entropy_coef_max",
        "agent.target_entropy",
        "training.save_interval",
    }

    # Recursive function to compare nested dictionaries
    def compare_dicts(d1, d2, path=""):
        diffs = []
        all_keys = set(d1.keys()) | set(d2.keys())

        for key in all_keys:
            full_path = f"{path}.{key}" if path else key

            # Check both the key and the full_path against ignore_keys
            if key in ignore_keys or full_path in ignore_keys:
                continue

            if key not in d1:
                diffs.append(f"    - {full_path}: missing in reference")
            elif key not in d2:
                diffs.append(f"    - {full_path}: missing in current run")
            elif isinstance(d1[key], dict) and isinstance(d2[key], dict):
                diffs.extend(compare_dicts(d1[key], d2[key], full_path))
            elif d1[key] != d2[key]:
                diffs.append(f"    - {full_path}: {d1[key]} != {d2[key]}")

        return diffs

    # Compare all runs with reference
    for run_dir, config in configs[1:]:
        diffs = compare_dicts(ref_config, config)
        if diffs:
            is_valid = False
            run_name = os.path.basename(run_dir)
            warnings.append(f"  ❌ Different config in {run_name}:")
            warnings.extend(diffs)

    if is_valid:
        warnings.append("  ✓ All runs have consistent configurations")

    return is_valid, warnings


def aggregate_move_success_rate(run_dirs: list[str], event_type: str = "training") -> dict | None:
    """
    Calculates move success rate per block: (movements - collisions) / movements.

    This is a fairer metric than overall collision_rate because it only considers
    move actions, not rotations (which can't collide).

    Always uses Block/* tags (not Rollout/*) for accurate per-block statistics.

    Args:
        run_dirs: List of run directories for the same scenario
        event_type: "training" or "eval"

    Returns:
        dict with {steps, mean, std, all_runs, n_seeds} or None
    """
    all_steps = []
    all_success_rates = []
    all_run_dirs = []

    # Select tags based on mode
    steps_tag = BLOCK_TAGS["steps"]
    collision_rate_tag = BLOCK_TAGS["collision_rate"]

    for run_dir in run_dirs:
        if event_type == "training":
            event_file = get_training_event_file(run_dir)
        else:
            event_file = get_eval_event_file(run_dir)

        if event_file is None:
            continue

        # Load required metrics
        steps_vals, values_steps = extract_scalar_from_event(event_file, steps_tag)
        collision_steps, values_collision_rate = extract_scalar_from_event(event_file, collision_rate_tag)

        # Load all action types
        movements_steps, values_movements = extract_scalar_from_event(event_file, ACTION_TAGS["movements"])
        pure_rotations_steps, values_pure_rotations = extract_scalar_from_event(event_file, ACTION_TAGS["pure_rotations"])
        idle_steps, values_idle = extract_scalar_from_event(event_file, ACTION_TAGS["idle"])
        stops_steps, values_stops = extract_scalar_from_event(event_file, ACTION_TAGS["stops"])

        # Check if we have at least movements and collision_rate
        if not (steps_vals and collision_steps and values_collision_rate and movements_steps and values_movements):
            continue

        # Find minimum length across all available metrics
        min_len = min(len(steps_vals), len(collision_steps), len(movements_steps))
        if min_len == 0:
            continue

        # Trim all arrays to same length
        steps_vals = steps_vals[:min_len]
        values_collision_rate = np.array(values_collision_rate[:min_len])
        values_movements = np.array(values_movements[:min_len])

        # Calculate total_actions from action types (movements + pure_rotations + idle + stops)
        # Not all runs have all action types, so we handle missing ones
        total_actions = values_movements.copy()

        if pure_rotations_steps and len(values_pure_rotations) >= min_len:
            total_actions += np.array(values_pure_rotations[:min_len])

        if idle_steps and len(values_idle) >= min_len:
            total_actions += np.array(values_idle[:min_len])

        if stops_steps and len(values_stops) >= min_len:
            total_actions += np.array(values_stops[:min_len])

        # Calculate: collisions = collision_rate * total_actions
        collisions = values_collision_rate * total_actions

        # Calculate: move_success_rate = (movements - collisions) / movements
        # Avoid division by zero
        move_success_rate = np.where(
            values_movements > 0,
            (values_movements - collisions) / values_movements,
            1.0,  # If no movements, success rate is 1.0 by convention
        )

        all_steps.append(steps_vals)
        all_success_rates.append(move_success_rate)
        all_run_dirs.append(run_dir)

    if not all_steps:
        return None

    # Filter: Keep only runs with maximum length
    max_len = max(len(s) for s in all_steps)
    filtered_steps = []
    filtered_rates = []
    excluded_runs = []

    for i, (steps, rates, run_dir) in enumerate(zip(all_steps, all_success_rates, all_run_dirs)):
        if len(steps) == max_len:
            filtered_steps.append(steps)
            filtered_rates.append(rates)
        else:
            excluded_runs.append((os.path.basename(run_dir), len(steps), max_len))

    if excluded_runs:
        print(f"[WARN] {len(excluded_runs)} run(s) excluded from move success rate (incomplete length):")
        for run_name, actual_len, expected_len in excluded_runs:
            print(f"  - {run_name}: {actual_len}/{expected_len} blocks")

    if not filtered_steps:
        return None

    all_trimmed = np.array([np.asarray(r, dtype=float) for r in filtered_rates])

    return {
        "steps": np.array(filtered_steps[0]),
        "mean": np.mean(all_trimmed, axis=0),
        "std": np.std(all_trimmed, axis=0),
        "all_runs": all_trimmed,
        "n_seeds": len(filtered_rates),
    }


def aggregate_post_train_eval(base_dir: str, scenario_name: str, metric_tag: str) -> dict | None:
    """
    Aggregates post-training evaluation data for a scenario.

    Post-training eval data is stored centrally in:
    {base_dir}/post_train_eval/{scenario_name}/

    All seeds for a scenario are logged to one shared event file.

    Args:
        base_dir: Base directory with runs (e.g., "RL_training/runs")
        scenario_name: Scenario name (e.g., "scenario1")
        metric_tag: TensorBoard tag (e.g., "eval/score")

    Returns:
        dict with keys: mean, std, n_episodes (or None if no data)
    """
    # Look for event file in central post_train_eval directory
    post_eval_dir = os.path.join(base_dir, "post_train_eval", scenario_name)

    if not os.path.isdir(post_eval_dir):
        return None

    # Find event file
    event_file = None
    for f in os.listdir(post_eval_dir):
        if f.startswith("events.out.tfevents."):
            event_file = os.path.join(post_eval_dir, f)
            break

    if event_file is None:
        return None

    # Extract scalar values (all episodes from all seeds)
    _, values = extract_scalar_from_event(event_file, metric_tag)
    if values is None or len(values) == 0:
        return None

    return {"mean": float(np.mean(values)), "std": float(np.std(values)), "n_episodes": len(values)}


def aggregate_post_train_eval_movement_success_rate(base_dir: str, scenario_name: str) -> dict | None:
    """
    Aggregates movement success rate for post-training eval.

    Movement success rate is computed as a cumulative rate:
    total_successful_movements / total_movements (across all episodes)

    This is different from other metrics where we take the mean of per-episode values.

    Args:
        base_dir: Base directory with runs (e.g., "RL_training/runs")
        scenario_name: Scenario name (e.g., "scenario1")

    Returns:
        dict with keys: rate, total_movements, total_successful, n_episodes (or None if no data)
    """
    # Look for event file in central post_train_eval directory
    post_eval_dir = os.path.join(base_dir, "post_train_eval", scenario_name)

    if not os.path.isdir(post_eval_dir):
        return None

    # Find event file
    event_file = None
    for f in os.listdir(post_eval_dir):
        if f.startswith("events.out.tfevents."):
            event_file = os.path.join(post_eval_dir, f)
            break

    if event_file is None:
        return None

    # Extract movement counts
    _, movement_counts = extract_scalar_from_event(event_file, "eval/movement_count")
    _, successful_movements = extract_scalar_from_event(event_file, "eval/successful_movements")

    if movement_counts is None or successful_movements is None:
        return None

    if len(movement_counts) == 0 or len(successful_movements) == 0:
        return None

    # Compute cumulative rate
    total_movements = np.sum(movement_counts)
    total_successful = np.sum(successful_movements)

    if total_movements == 0:
        return None

    rate = float(total_successful / total_movements)

    return {
        "rate": rate,
        "total_movements": int(total_movements),
        "total_successful": int(total_successful),
        "n_episodes": len(movement_counts),
    }


def aggregate_derived_metric(run_dirs: list[str], numerator_tag: str, denominator_tag: str, event_type: str = "training") -> dict | None:
    """
    Calculates a derived metric as a ratio of two tags.

    Args:
        run_dirs: List of run directories for the same scenario
        numerator_tag: TensorBoard tag for numerator
        denominator_tag: TensorBoard tag for denominator
        event_type: "training" or "eval"

    Returns:
        dict with {steps, mean, std, all_runs, n_seeds} or None
    """
    all_steps = []
    all_ratios = []
    all_run_dirs = []

    for run_dir in run_dirs:
        if event_type == "training":
            event_file = get_training_event_file(run_dir)
        else:
            event_file = get_eval_event_file(run_dir)

        if event_file is None:
            continue

        steps_num, values_num = extract_scalar_from_event(event_file, numerator_tag)
        steps_den, values_den = extract_scalar_from_event(event_file, denominator_tag)

        if steps_num and values_num and steps_den and values_den:
            # Both must have the same steps
            if steps_num != steps_den:
                continue

            # Calculate ratio, avoid division by zero
            ratio = np.array([n / d if d > 0 else 0 for n, d in zip(values_num, values_den)], dtype=float)
            all_steps.append(steps_num)
            all_ratios.append(ratio)
            all_run_dirs.append(run_dir)

    if not all_steps:
        return None

    # Filter: Keep only runs with maximum length
    max_len = max(len(s) for s in all_steps)
    filtered_steps = []
    filtered_ratios = []
    excluded_runs = []

    for i, (steps, ratio, run_dir) in enumerate(zip(all_steps, all_ratios, all_run_dirs)):
        if len(steps) == max_len:
            filtered_steps.append(steps)
            filtered_ratios.append(ratio)
        else:
            excluded_runs.append((os.path.basename(run_dir), len(steps), max_len))

    # Print warning if runs are excluded
    if excluded_runs:
        print(f"[WARN] {len(excluded_runs)} run(s) excluded from derived metric (incomplete length):")
        for run_name, actual_len, expected_len in excluded_runs:
            print(f"  - {run_name}: {actual_len}/{expected_len} blocks")

    if not filtered_steps:
        return None

    all_trimmed = np.array([np.asarray(r, dtype=float) for r in filtered_ratios])

    return {
        "steps": np.array(filtered_steps[0]),
        "mean": np.mean(all_trimmed, axis=0),
        "std": np.std(all_trimmed, axis=0),
        "all_runs": all_trimmed,
        "n_seeds": len(filtered_ratios),
    }


def aggregate_scenario_seeds(run_dirs: list[str], tag: str, event_type: str = "training") -> dict | None:
    """
    Aggregates data across multiple seeds of a scenario.

    Args:
        run_dirs: List of run directories for the same scenario
        tag: TensorBoard tag
        event_type: "training" or "eval"

    Returns:
        dict with {steps, mean, std, all_runs, n_seeds} or None
    """
    all_steps = []
    all_values = []
    all_run_dirs = []

    for run_dir in run_dirs:
        if event_type == "training":
            event_file = get_training_event_file(run_dir)
        else:
            event_file = get_eval_event_file(run_dir)

        if event_file is None:
            continue

        steps, values = extract_scalar_from_event(event_file, tag)
        if steps and values:
            all_steps.append(steps)
            all_values.append(values)
            all_run_dirs.append(run_dir)

    if not all_steps:
        return None

    # Filter: Keep only runs with maximum length
    max_len = max(len(s) for s in all_steps)
    filtered_steps = []
    filtered_values = []
    excluded_runs = []

    for i, (steps, values, run_dir) in enumerate(zip(all_steps, all_values, all_run_dirs)):
        if len(steps) == max_len:
            filtered_steps.append(steps)
            filtered_values.append(values)
        else:
            excluded_runs.append((os.path.basename(run_dir), len(steps), max_len))

    # Print warning if runs are excluded
    if excluded_runs:
        print(f"[WARN] {len(excluded_runs)} run(s) excluded (incomplete length):")
        for run_name, actual_len, expected_len in excluded_runs:
            print(f"  - {run_name}: {actual_len}/{expected_len} blocks")

    if not filtered_steps:
        return None

    all_trimmed = np.array([np.asarray(v, dtype=float) for v in filtered_values])

    return {
        "steps": np.array(filtered_steps[0]),
        "mean": np.mean(all_trimmed, axis=0),
        "std": np.std(all_trimmed, axis=0),
        "all_runs": all_trimmed,
        "n_seeds": len(filtered_values),
    }


# ---- Plotting ---------------------------------------------------------------


def plot_metric(
    base_dir: str,
    tag: str,
    ylabel: str,
    title: str,
    max_blocks: int = 1000,
    ylim: tuple | None = None,
    smooth: int = 1,
    save_path: str | None = None,
    show: bool = True,
    event_type: str = "training",
    plot_ci: bool = True,
    plot_seeds: bool = False,
    band_type: str = "ci",
    ci_level: float = 0.95,
    auto_rollout_smooth: bool = False,
    validate_configs: bool = False,
):
    """
    Plots a metric for all scenarios.

    Args:
        base_dir: Base directory with runs
        tag: TensorBoard tag
        ylabel: Y-axis label
        title: Plot title
        max_blocks: Maximum for X-axis
        ylim: Y-axis limits
        smooth: Smoothing window (1 = no smoothing)
        save_path: Save path for plot
        show: Show plot
        event_type: "training" or "eval"
        plot_ci: Show confidence band
        plot_seeds: Show individual seed lines
        band_type: "ci", "se", or "iqr"
        ci_level: Confidence level for CI
        validate_configs: Validate configs between runs
    """
    if save_path:
        _ensure_dir(os.path.dirname(save_path))

    plt.figure(figsize=(10, 6))
    runs = discover_runs(base_dir)

    if not runs:
        print(f"[WARN] No runs found in {base_dir}")
        return

    # Config validation for all scenarios
    if validate_configs:
        print("\n" + "=" * 80)
        print("CONFIG VALIDATION")
        print("=" * 80)
        any_errors = False
        for scenario_name, run_dirs in sorted(runs.items(), key=lambda x: _scenario_sort_key(x[0])):
            is_valid, warnings = validate_scenario_configs(scenario_name, run_dirs)
            if warnings:
                print(f"\n{_nice_label(scenario_name)} ({len(run_dirs)} runs):")
                for warning in warnings:
                    print(warning)
                if not is_valid:
                    any_errors = True

        if not any_errors:
            print("\n✓ All scenarios have consistent configurations")
        print("=" * 80 + "\n")

    plotted = 0
    for scenario_name, run_dirs in sorted(runs.items(), key=lambda x: _scenario_sort_key(x[0])):
        data = aggregate_scenario_seeds(run_dirs, tag, event_type)
        if data is None:
            print(f"[WARN] No data for {scenario_name} (tag: {tag})")
            continue

        steps = data["steps"]
        mean = data["mean"]
        std = data["std"]
        all_runs = data["all_runs"]
        n = data["n_seeds"]

        # Apply rollout-window smoothing (causal, with expanding window at the start)
        if auto_rollout_smooth:
            mean = causal_smooth(mean, DEFAULT_ROLLOUT_WINDOW)
            std = causal_smooth(std, DEFAULT_ROLLOUT_WINDOW)
            all_runs = np.array([causal_smooth(run, DEFAULT_ROLLOUT_WINDOW) for run in all_runs])

        label = _nice_label(scenario_name)

        # Apply smoothing
        if smooth > 1:
            mean_plot = moving_average(mean, smooth)
            x_plot = steps[: len(mean_plot)]
        else:
            mean_plot = mean
            x_plot = steps

        # Get line style for this scenario
        linestyle = SCENARIO_LINESTYLES.get(scenario_name, "-")

        # Main line
        plt.plot(x_plot, mean_plot, label=f"{label} (n={n})", linewidth=2, linestyle=linestyle)

        # Optional seed lines
        if plot_seeds and all_runs is not None:
            for seed_vals in all_runs:
                sv = moving_average(seed_vals, smooth) if smooth > 1 else seed_vals
                plt.plot(steps[: len(sv)], sv, alpha=0.15, linewidth=1)

        # Confidence band
        if plot_ci:
            se = std / max(1, np.sqrt(n))

            if band_type == "iqr":
                lower_raw = np.quantile(all_runs, 0.25, axis=0)
                upper_raw = np.quantile(all_runs, 0.75, axis=0)
            elif band_type == "se":
                lower_raw = mean - se
                upper_raw = mean + se
            else:  # ci
                z = get_z_value(ci_level)
                lower_raw = mean - z * se
                upper_raw = mean + z * se

            if smooth > 1:
                lower = moving_average(lower_raw, smooth)
                upper = moving_average(upper_raw, smooth)
                x_band = steps[: len(lower)]
            else:
                lower, upper = lower_raw, upper_raw
                x_band = x_plot

            plt.fill_between(x_band, lower, upper, alpha=0.15)

        plotted += 1

    if plotted == 0:
        print(f"[WARN] No curves plotted for tag '{tag}'")
        plt.close()
        return

    plt.legend(ncol=optimal_legend_ncol(plotted), fontsize=12)
    if ylim is not None:
        plt.ylim(ylim)
    plt.xlim([0, max_blocks])
    plt.xlabel("Block")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[INFO] Saved: {save_path}")
    if show:
        plt.show()
    else:
        plt.close()


def plot_derived_metric(
    base_dir: str,
    numerator_tag: str,
    denominator_tag: str,
    ylabel: str,
    title: str,
    max_blocks: int = 1000,
    ylim: tuple | None = None,
    smooth: int = 1,
    save_path: str | None = None,
    show: bool = True,
    event_type: str = "training",
    validate_configs: bool = False,
    ci_level: float = 0.95,
):
    """
    Plots a derived metric (ratio) for all scenarios.

    Args:
        base_dir: Base directory with runs
        numerator_tag: TensorBoard tag for numerator
        denominator_tag: TensorBoard tag for denominator
        ylabel: Y-axis label
        title: Plot title
        max_blocks: Maximum for X-axis
        ylim: Y-axis limits
        smooth: Smoothing window
        save_path: Save path for plot
        show: Show plot
        event_type: "training" or "eval"
        validate_configs: Validate configs between runs
        ci_level: Confidence level for CI
    """
    if save_path:
        _ensure_dir(os.path.dirname(save_path))

    plt.figure(figsize=(10, 6))
    runs = discover_runs(base_dir)

    if not runs:
        print(f"[WARN] No runs found in {base_dir}")
        return

    # Config validation (only if desired)
    if validate_configs:
        print("\n" + "=" * 80)
        print("CONFIG VALIDATION")
        print("=" * 80)
        any_errors = False
        for scenario_name, run_dirs in sorted(runs.items(), key=lambda x: _scenario_sort_key(x[0])):
            is_valid, warnings = validate_scenario_configs(scenario_name, run_dirs)
            if warnings:
                print(f"\n{_nice_label(scenario_name)} ({len(run_dirs)} runs):")
                for warning in warnings:
                    print(warning)
                if not is_valid:
                    any_errors = True

        if not any_errors:
            print("\n✓ All scenarios have consistent configurations")
        print("=" * 80 + "\n")

    plotted = 0
    z = get_z_value(ci_level)

    for scenario_name, run_dirs in sorted(runs.items(), key=lambda x: _scenario_sort_key(x[0])):
        data = aggregate_derived_metric(run_dirs, numerator_tag, denominator_tag, event_type)
        if data is None:
            print(f"[WARN] No data for {scenario_name} (derived metric)")
            continue

        steps = data["steps"]
        mean = data["mean"]
        std = data["std"]
        n = data["n_seeds"]

        label = _nice_label(scenario_name)

        # Smoothing anwenden
        if smooth > 1:
            mean_plot = moving_average(mean, smooth)
            x_plot = steps[: len(mean_plot)]
        else:
            mean_plot = mean
            x_plot = steps

        # Get line style for this scenario
        linestyle = SCENARIO_LINESTYLES.get(scenario_name, "-")

        # Main line
        plt.plot(x_plot, mean_plot, label=f"{label} (n={n})", linewidth=2, linestyle=linestyle)

        # Confidence band
        se = std / max(1, np.sqrt(n))
        ci_half = z * se

        if smooth > 1:
            lower = moving_average(mean - ci_half, smooth)
            upper = moving_average(mean + ci_half, smooth)
            x_band = steps[: len(lower)]
        else:
            lower = mean - ci_half
            upper = mean + ci_half
            x_band = x_plot

        plt.fill_between(x_band, lower, upper, alpha=0.15)

        plotted += 1

    if plotted == 0:
        print(f"[WARN] No curves plotted for derived metric")
        plt.close()
        return

    plt.legend(ncol=optimal_legend_ncol(plotted), fontsize=12)
    if ylim is not None:
        plt.ylim(ylim)
    plt.xlim([0, max_blocks])
    plt.xlabel("Block")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[INFO] Saved: {save_path}")
    if show:
        plt.show()
    else:
        plt.close()


def plot_move_success_rate(
    base_dir: str,
    max_blocks: int = 1000,
    ylim: tuple | None = None,
    smooth: int = 1,
    save_path: str | None = None,
    show: bool = True,
    validate_configs: bool = False,
    ci_level: float = 0.95,
):
    """
    Plots move success rate per block (fairer than overall collision_rate).

    Move success rate = (movements - collisions) / movements
    This only considers move actions, not rotations.

    Always uses Block/* tags for accurate per-block statistics.

    Args:
        base_dir: Base directory with runs
        max_blocks: Maximum for X-axis
        ylim: Y-axis limits
        smooth: Smoothing window
        save_path: Save path for plot
        show: Show plot
        validate_configs: Validate configs between runs
        ci_level: Confidence level for CI
    """
    if save_path:
        _ensure_dir(os.path.dirname(save_path))

    plt.figure(figsize=(10, 6))
    runs = discover_runs(base_dir)

    if not runs:
        print(f"[WARN] No runs found in {base_dir}")
        return

    plotted = 0
    z = get_z_value(ci_level)

    for scenario_name, run_dirs in sorted(runs.items(), key=lambda x: _scenario_sort_key(x[0])):
        data = aggregate_move_success_rate(run_dirs, "training")
        if data is None:
            print(f"[WARN] No data for {scenario_name} (move success rate)")
            continue

        steps = data["steps"]
        mean = data["mean"]
        std = data["std"]
        n = data["n_seeds"]

        label = _nice_label(scenario_name)

        # Apply smoothing
        if smooth > 1:
            mean_plot = moving_average(mean, smooth)
            x_plot = steps[: len(mean_plot)]
        else:
            mean_plot = mean
            x_plot = steps

        # Get line style
        linestyle = SCENARIO_LINESTYLES.get(scenario_name, "-")

        # Main line
        plt.plot(x_plot, mean_plot, label=f"{label} (n={n})", linewidth=2, linestyle=linestyle)

        # Confidence band
        se = std / max(1, np.sqrt(n))
        ci_half = z * se

        if smooth > 1:
            lower = moving_average(mean - ci_half, smooth)
            upper = moving_average(mean + ci_half, smooth)
            x_band = steps[: len(lower)]
        else:
            lower = mean - ci_half
            upper = mean + ci_half
            x_band = x_plot

        plt.fill_between(x_band, lower, upper, alpha=0.15)

        plotted += 1

    if plotted == 0:
        print(f"[WARN] No curves plotted for move success rate")
        plt.close()
        return

    plt.legend(ncol=optimal_legend_ncol(plotted), fontsize=12)
    if ylim is not None:
        plt.ylim(ylim)
    plt.xlim([0, max_blocks])
    plt.xlabel("Block")
    plt.ylabel("Move Success Rate")
    plt.title("Train Move Success Rate")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[INFO] Saved: {save_path}")
    if show:
        plt.show()
    else:
        plt.close()


def plot_training_metric(
    base_dir: str,
    metric: str = "score",
    max_blocks: int = 1000,
    ylim: tuple | None = None,
    smooth: int = 1,
    save_path: str | None = None,
    show: bool = True,
    use_block_tags: bool = False,
    **kwargs,
):
    """
    Plots a training metric.

    Args:
        metric: "score", "steps", "reward", or "steps_for_score_1"
        use_block_tags: True for Block/* tags (custom smoothing), False for Rollout/*
    """
    tags = BLOCK_TAGS if use_block_tags else ROLLOUT_TAGS

    if metric not in tags:
        print(f"[ERROR] Unknown metric: {metric}. Available: {list(tags.keys())}")
        return

    tag = tags[metric]

    titles = {
        "score": "Train Score",
        "steps": "Train Steps",
        "reward": "Train Reward",
        "collision_rate": "Train Collision Rate",
        "steps_for_score_1": "Train Steps for Score 1",
        "path_length": "Train Path Length",
        "term_trunc_ratio": "Train Terminated/Truncated Ratio",
    }

    ylabels = {
        "score": "Score",
        "steps": "Steps",
        "reward": "Reward",
        "collision_rate": "Collision Rate",
        "steps_for_score_1": "Steps",
        "path_length": "Path Length",
        "term_trunc_ratio": "Term./Trunc. Ratio",
    }

    # Automatic rollout-window smoothing for unsmoothed metrics
    auto_rollout_smooth = not use_block_tags and metric in UNSMOOTHED_METRICS

    plot_metric(
        base_dir=base_dir,
        tag=tag,
        ylabel=ylabels.get(metric, metric),
        title=titles.get(metric, metric),
        max_blocks=max_blocks,
        ylim=ylim,
        smooth=smooth,
        save_path=save_path,
        show=show,
        event_type="training",
        auto_rollout_smooth=auto_rollout_smooth,
        **kwargs,
    )


def plot_eval_metric(
    base_dir: str,
    metric: str = "score",
    max_blocks: int = 1000,
    ylim: tuple | None = None,
    save_path: str | None = None,
    show: bool = True,
    **kwargs,
):
    """
    Plots an eval metric.

    Args:
        metric: "score", "steps", "reward"
    """
    if metric not in EVAL_TAGS:
        print(f"[ERROR] Unknown eval metric: {metric}. Available: {list(EVAL_TAGS.keys())}")
        return

    tag = EVAL_TAGS[metric]

    titles = {"score": "Evaluation Score (Held-Out)", "steps": "Evaluation Steps (Held-Out)", "reward": "Evaluation Reward (Held-Out)"}

    ylabels = {"score": "Score", "steps": "Steps", "reward": "Reward"}

    plot_metric(
        base_dir=base_dir,
        tag=tag,
        ylabel=ylabels.get(metric, metric),
        title=titles.get(metric, metric),
        max_blocks=max_blocks,
        ylim=ylim,
        smooth=1,  # Do not extra smooth eval data
        save_path=save_path,
        show=show,
        event_type="eval",
        **kwargs,
    )


def plot_training_summary(
    base_dir: str,
    metrics: list[str] = None,
    max_blocks: int = 1000,
    smooth: int = 1,
    save_path: str | None = None,
    show: bool = True,
    use_block_tags: bool = False,
    ci_level: float = 0.95,
    validate_configs: bool = False,
    **kwargs,
):
    """
    Creates a multi-panel plot for multiple training metrics.

    Args:
        metrics: List of metrics (default: ["score", "steps", "move_success_rate"])
        validate_configs: Validate configs between runs
    """
    if metrics is None:
        metrics = ["score", "steps", "move_success_rate"]

    if save_path:
        _ensure_dir(os.path.dirname(save_path))

    n_metrics = len(metrics)
    # For single metric double width, for 2+ metrics normal width
    width_per_metric = 10 if n_metrics == 1 else 5
    fig, axes = plt.subplots(1, n_metrics, figsize=(width_per_metric * n_metrics, 5), squeeze=False)
    axes = axes[0]

    tags = BLOCK_TAGS if use_block_tags else ROLLOUT_TAGS
    runs = discover_runs(base_dir)

    if not runs:
        print(f"[WARN] No runs found in {base_dir}")
        return

    # Config validation for all scenarios (only once at the start)
    if validate_configs:
        print("\n" + "=" * 80)
        print("CONFIG VALIDATION")
        print("=" * 80)
        any_errors = False
        for scenario_name, run_dirs in sorted(runs.items(), key=lambda x: _scenario_sort_key(x[0])):
            is_valid, warnings = validate_scenario_configs(scenario_name, run_dirs)
            if warnings:
                print(f"\n{_nice_label(scenario_name)} ({len(run_dirs)} runs):")
                for warning in warnings:
                    print(warning)
                if not is_valid:
                    any_errors = True

        if not any_errors:
            print("\n✓ All scenarios have consistent configurations")
        print("=" * 80 + "\n")

    ylabels = {
        "score": "Score",
        "steps": "Steps",
        "reward": "Reward",
        "collision_rate": "Collision Rate",
        "move_success_rate": "Move Success Rate",
        "path_length": "Path Length",
        "term_trunc_ratio": "Term./Trunc. Ratio",
    }
    titles = {
        "score": "Train Score",
        "steps": "Train Steps",
        "reward": "Train Reward",
        "collision_rate": "Train Collision Rate",
        "move_success_rate": "Move Success Rate",
        "path_length": "Train Path Length",
        "term_trunc_ratio": "Terminated/Truncated Ratio",
    }

    for ax, metric in zip(axes, metrics):
        # Special handling for move_success_rate (derived metric)
        if metric == "move_success_rate":
            z = get_z_value(ci_level)

            for scenario_name, run_dirs in sorted(runs.items(), key=lambda x: _scenario_sort_key(x[0])):
                data = aggregate_move_success_rate(run_dirs, "training")
                if data is None:
                    continue

                steps = data["steps"]
                mean = data["mean"]
                std = data["std"]
                n = data["n_seeds"]

                se = std / max(1, np.sqrt(n))
                ci_half = z * se

                label = _nice_label(scenario_name)

                if smooth > 1:
                    mean_plot = moving_average(mean, smooth)
                    x_plot = steps[: len(mean_plot)]
                    lower = moving_average(mean - ci_half, smooth)
                    upper = moving_average(mean + ci_half, smooth)
                else:
                    mean_plot = mean
                    x_plot = steps
                    lower = mean - ci_half
                    upper = mean + ci_half

                # Get line style for this scenario
                linestyle = SCENARIO_LINESTYLES.get(scenario_name, "-")

                ax.plot(x_plot, mean_plot, label=f"{label}", linewidth=2, linestyle=linestyle)
                ax.fill_between(x_plot, lower[: len(x_plot)], upper[: len(x_plot)], alpha=0.15)

            ax.set_xlabel("Block")
            ax.set_ylabel(ylabels.get(metric, metric))
            ax.set_title(titles.get(metric, metric))
            ax.set_xlim([0, max_blocks])
            continue

        # Normal metrics (from tags)
        if metric not in tags:
            ax.set_title(f"Unknown: {metric}")
            continue

        tag = tags[metric]

        # Automatic rollout-window smoothing for unsmoothed metrics
        # (only if not using block tags)
        needs_rollout_smooth = not use_block_tags and metric in UNSMOOTHED_METRICS

        z = get_z_value(ci_level)

        for scenario_name, run_dirs in sorted(runs.items(), key=lambda x: _scenario_sort_key(x[0])):
            data = aggregate_scenario_seeds(run_dirs, tag, "training")
            if data is None:
                continue

            steps = data["steps"]
            mean = data["mean"]
            std = data["std"]
            n = data["n_seeds"]

            # Apply rollout-window smoothing (causal, with expanding window at the start)
            if needs_rollout_smooth:
                mean = causal_smooth(mean, DEFAULT_ROLLOUT_WINDOW)
                std = causal_smooth(std, DEFAULT_ROLLOUT_WINDOW)

            se = std / max(1, np.sqrt(n))
            ci_half = z * se

            label = _nice_label(scenario_name)

            if smooth > 1:
                mean_plot = moving_average(mean, smooth)
                x_plot = steps[: len(mean_plot)]
                lower = moving_average(mean - ci_half, smooth)
                upper = moving_average(mean + ci_half, smooth)
            else:
                mean_plot = mean
                x_plot = steps
                lower = mean - ci_half
                upper = mean + ci_half

            # Get line style for this scenario
            linestyle = SCENARIO_LINESTYLES.get(scenario_name, "-")

            ax.plot(x_plot, mean_plot, label=f"{label}", linewidth=2, linestyle=linestyle)
            ax.fill_between(x_plot, lower[: len(x_plot)], upper[: len(x_plot)], alpha=0.15)

        ax.set_xlabel("Block")
        ax.set_ylabel(ylabels.get(metric, metric))
        ax.set_title(titles.get(metric, metric))
        ax.set_xlim([0, max_blocks])

    # Shared legend with optimal layout (max 4 columns)
    handles, labels = axes[0].get_legend_handles_labels()
    n_scenarios = len(handles)
    ncol = optimal_legend_ncol(n_scenarios, max_ncol=4)
    legend_height = 0.08 + 0.04 * ((n_scenarios - 1) // ncol)
    fig.legend(handles, labels, loc="lower center", ncol=ncol, frameon=False, fontsize=12)
    fig.tight_layout(rect=[0, legend_height, 1, 1])

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[INFO] Saved: {save_path}")
    if show:
        plt.show()
    else:
        plt.close()


def plot_training_vs_eval(
    base_dir: str,
    metric: str = "reward",
    max_blocks: int = 1000,
    ylim: tuple | None = None,
    smooth: int = 1,
    save_path: str | None = None,
    show: bool = True,
    use_block_tags: bool = False,
    ci_level: float = 0.95,
):
    """
    Plots training and eval data for a metric in two separate subplots.

    Args:
        base_dir: Base directory with runs
        metric: Metric name (e.g., "reward", "score", "steps")
        max_blocks: Maximum for X-axis
        ylim: Y-axis limits (used for both plots)
        smooth: Smoothing window for training
        save_path: Save path for plot
        show: Show plot
        use_block_tags: True for Block/* tags, False for Rollout/*
        ci_level: Confidence level for CI
    """
    if save_path:
        _ensure_dir(os.path.dirname(save_path))

    runs = discover_runs(base_dir)

    if not runs:
        print(f"[WARN] No runs found in {base_dir}")
        return

    train_tags = BLOCK_TAGS if use_block_tags else ROLLOUT_TAGS

    if metric not in train_tags:
        print(f"[ERROR] Unknown metric: {metric}")
        return

    if metric not in EVAL_TAGS:
        print(f"[ERROR] Metric {metric} not available in eval")
        return

    train_tag = train_tags[metric]
    eval_tag = EVAL_TAGS[metric]

    z = get_z_value(ci_level)

    # Create figure with two subplots
    fig, (ax_train, ax_eval) = plt.subplots(1, 2, figsize=(16, 6))

    # Color palette - use consistent colors per scenario
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # Collect all y-values for automatic scaling
    all_y_values = []

    plotted = 0

    for idx, (scenario_name, run_dirs) in enumerate(sorted(runs.items(), key=lambda x: _scenario_sort_key(x[0]))):
        color = colors[idx % len(colors)]
        label = _nice_label(scenario_name)
        linestyle = SCENARIO_LINESTYLES.get(scenario_name, "-")

        # Training data
        train_data = aggregate_scenario_seeds(run_dirs, train_tag, "training")
        if train_data is not None:
            steps = train_data["steps"]
            mean = train_data["mean"]
            std = train_data["std"]
            n = train_data["n_seeds"]

            # Smoothing for training
            if smooth > 1:
                mean_plot = moving_average(mean, smooth)
                x_plot = steps[: len(mean_plot)]
            else:
                mean_plot = mean
                x_plot = steps

            # Plot training
            ax_train.plot(x_plot, mean_plot, label=f"{label}", linewidth=2, color=color, linestyle=linestyle)

            # Confidence band
            se = std / max(1, np.sqrt(n))
            ci_half = z * se
            if smooth > 1:
                lower = moving_average(mean - ci_half, smooth)
                upper = moving_average(mean + ci_half, smooth)
            else:
                lower = mean - ci_half
                upper = mean + ci_half
            ax_train.fill_between(x_plot, lower[: len(x_plot)], upper[: len(x_plot)], alpha=0.15, color=color)

            # Collect y-values for scaling
            all_y_values.extend(mean_plot)

        # Eval data
        eval_data = aggregate_scenario_seeds(run_dirs, eval_tag, "eval")
        if eval_data is not None:
            steps = eval_data["steps"]
            mean = eval_data["mean"]
            std = eval_data["std"]
            n = eval_data["n_seeds"]

            # Light smoothing for eval
            mean = causal_smooth(mean, 3)
            std = causal_smooth(std, 3)

            # Plot eval
            ax_eval.plot(steps, mean, label=f"{label}", linewidth=2, color=color, linestyle=linestyle)

            # Confidence band
            se = std / max(1, np.sqrt(n))
            ci_half = z * se
            ax_eval.fill_between(steps, mean - ci_half, mean + ci_half, alpha=0.15, color=color)

            # Collect y-values for scaling
            all_y_values.extend(mean)

        if train_data is not None or eval_data is not None:
            plotted += 1

    if plotted == 0:
        print(f"[WARN] No curves plotted for {metric}")
        plt.close()
        return

    # Set equal y-axis limits for both plots
    if ylim is not None:
        ax_train.set_ylim(ylim)
        ax_eval.set_ylim(ylim)
    elif all_y_values:
        # Automatic scaling based on all data
        y_min = np.min(all_y_values)
        y_max = np.max(all_y_values)
        y_margin = (y_max - y_min) * 0.05
        ax_train.set_ylim([y_min - y_margin, y_max + y_margin])
        ax_eval.set_ylim([y_min - y_margin, y_max + y_margin])

    # Set x-axis limits
    ax_train.set_xlim([0, max_blocks])
    ax_eval.set_xlim([0, max_blocks])

    # Labels and titles
    ax_train.set_xlabel("Block")
    ax_train.set_ylabel(metric.capitalize())
    ax_train.set_title(f"Train {metric.capitalize()}")

    ax_eval.set_xlabel("Block")
    ax_eval.set_ylabel(metric.capitalize())
    ax_eval.set_title(f"Eval {metric.capitalize()}")

    # Shared legend below both plots with optimal layout (max 4 columns)
    handles, labels = ax_train.get_legend_handles_labels()
    n_scenarios = len(handles)
    ncol = optimal_legend_ncol(n_scenarios, max_ncol=4)
    legend_height = 0.08 + 0.04 * ((n_scenarios - 1) // ncol)
    fig.legend(handles, labels, loc="lower center", ncol=ncol, frameon=False, fontsize=12)
    fig.tight_layout(rect=[0, legend_height, 1, 1])

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[INFO] Saved: {save_path}")
    if show:
        plt.show()
    else:
        plt.close()


def plot_scatter_pair(
    base_dir: str,
    x_tag: str,
    y1_tag: str,
    y2_tag: str,
    xlabel: str,
    y1_label: str,
    y2_label: str,
    title_prefix: str,
    save_path: str | None = None,
    show: bool = True,
    event_type: str = "training",
    use_block_tags: bool = True,
    max_points: int = 5000,
    scenario_filter: list[str] | None = None,
):
    """
    Creates two scatter plots side by side with shared legend.

    Args:
        base_dir: Base directory with runs
        x_tag: TensorBoard tag for X-axis (shared for both plots)
        y1_tag: TensorBoard tag for Y-axis (left plot)
        y2_tag: TensorBoard tag for Y-axis (right plot)
        xlabel: X-axis label
        y1_label: Y-axis label for left plot
        y2_label: Y-axis label for right plot
        title_prefix: Prefix for title (e.g., "Training" or "Eval")
        save_path: Save path for plot
        show: Show plot
        event_type: "training" or "eval"
        use_block_tags: Block/* or Rollout/* tags
        max_points: Maximum number of points per scenario (downsampling)
        scenario_filter: List of scenario names to plot (None = all)
    """
    if save_path:
        _ensure_dir(os.path.dirname(save_path))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    runs = discover_runs(base_dir)

    if not runs:
        print(f"[WARN] No runs found in {base_dir}")
        return

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    plotted = 0

    for idx, (scenario_name, run_dirs) in enumerate(sorted(runs.items(), key=lambda x: _scenario_sort_key(x[0]))):
        # Filter scenarios if specified
        if scenario_filter is not None and scenario_name not in scenario_filter:
            continue

        color = colors[idx % len(colors)]
        label = _nice_label(scenario_name)

        # Collect data for both Y-axes
        all_x = []
        all_y1 = []
        all_y2 = []

        for run_dir in run_dirs:
            if event_type == "training":
                event_file = get_training_event_file(run_dir)
            else:
                event_file = get_eval_event_file(run_dir)

            if event_file is None:
                continue

            _, x_values = extract_scalar_from_event(event_file, x_tag)
            _, y1_values = extract_scalar_from_event(event_file, y1_tag)
            _, y2_values = extract_scalar_from_event(event_file, y2_tag)

            if x_values and y1_values and y2_values:
                # Training: take last 100 blocks per run
                # Eval: take only last checkpoint (agent finished training)
                if event_type == "eval":
                    all_x.append(x_values[-1])
                    all_y1.append(y1_values[-1])
                    all_y2.append(y2_values[-1])
                else:
                    min_len = min(len(x_values), len(y1_values), len(y2_values))
                    last_n = min(100, min_len)
                    all_x.extend(x_values[-last_n:])
                    all_y1.extend(y1_values[-last_n:])
                    all_y2.extend(y2_values[-last_n:])

        if not all_x:
            continue

        # Downsampling if too many points
        if len(all_x) > max_points:
            indices = np.random.choice(len(all_x), max_points, replace=False)
            all_x = [all_x[i] for i in indices]
            all_y1 = [all_y1[i] for i in indices]
            all_y2 = [all_y2[i] for i in indices]

        # Plot settings based on event_type
        if event_type == "eval":
            scatter_kwargs = {"alpha": 0.8, "s": 150, "edgecolors": "white", "linewidths": 0.5}
        else:
            scatter_kwargs = {"alpha": 0.5, "s": 20}

        # Plot on both axes
        ax1.scatter(all_x, all_y1, label=label, color=color, **scatter_kwargs)
        ax2.scatter(all_x, all_y2, color=color, **scatter_kwargs)  # No label (only once in legend)
        plotted += 1

    if plotted == 0:
        print(f"[WARN] No data for scatter plot pair")
        plt.close()
        return

    # Axis labels and titles
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(y1_label)
    title1 = f"{title_prefix}: {y1_label} vs {xlabel}" if title_prefix else f"{y1_label} vs {xlabel}"
    ax1.set_title(title1)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel(xlabel)
    ax2.set_ylabel(y2_label)
    title2 = f"{title_prefix}: {y2_label} vs {xlabel}" if title_prefix else f"{y2_label} vs {xlabel}"
    ax2.set_title(title2)
    ax2.grid(True, alpha=0.3)

    # Shared legend below both plots with optimal layout
    handles, labels = ax1.get_legend_handles_labels()
    legend_scale = 1 if event_type == "eval" else 2.5
    n_scenarios = len(handles)
    ncol = optimal_legend_ncol(n_scenarios)
    legend_height = 0.08 + 0.04 * ((n_scenarios - 1) // ncol)
    fig.legend(handles, labels, loc="lower center", ncol=ncol, frameon=False, fontsize=12, markerscale=legend_scale)
    fig.tight_layout(rect=[0, legend_height, 1, 1])

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[INFO] Saved: {save_path}")
    if show:
        plt.show()
    else:
        plt.close()


def plot_scatter_triple(
    base_dir: str,
    x1_tag: str,
    y1_tag: str,
    x2_tag: str,
    y2_tag: str,
    x3_tag: str,
    y3_tag: str,
    x1_label: str,
    y1_label: str,
    x2_label: str,
    y2_label: str,
    x3_label: str,
    y3_label: str,
    title_prefix: str,
    save_path: str | None = None,
    show: bool = True,
    event_type: str = "training",
    use_block_tags: bool = True,
    max_points: int = 5000,
    scenario_filter: list[str] | None = None,
):
    """
    Creates three scatter plots side by side with shared legend.

    Args:
        base_dir: Base directory with runs
        x1_tag, y1_tag: TensorBoard tags for first plot
        x2_tag, y2_tag: TensorBoard tags for second plot
        x3_tag, y3_tag: TensorBoard tags for third plot
        x1_label, y1_label: Axis labels for first plot
        x2_label, y2_label: Axis labels for second plot
        x3_label, y3_label: Axis labels for third plot
        title_prefix: Prefix for titles (e.g., "Train" or "Eval")
        save_path: Save path for plot
        show: Show plot
        event_type: "training" or "eval"
        use_block_tags: Block/* or Rollout/* tags
        max_points: Maximum number of points per scenario (downsampling)
        scenario_filter: List of scenario names to plot (None = all)
    """
    if save_path:
        _ensure_dir(os.path.dirname(save_path))

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 6))
    runs = discover_runs(base_dir)

    if not runs:
        print(f"[WARN] No runs found in {base_dir}")
        return

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    plotted = 0

    for idx, (scenario_name, run_dirs) in enumerate(sorted(runs.items(), key=lambda x: _scenario_sort_key(x[0]))):
        # Filter scenarios if specified
        if scenario_filter is not None and scenario_name not in scenario_filter:
            continue

        color = colors[idx % len(colors)]
        label = _nice_label(scenario_name)

        # Collect data for all three plots
        all_x1, all_y1 = [], []
        all_x2, all_y2 = [], []
        all_x3, all_y3 = [], []

        for run_dir in run_dirs:
            if event_type == "training":
                event_file = get_training_event_file(run_dir)
            else:
                event_file = get_eval_event_file(run_dir)

            if event_file is None:
                continue

            _, x1_values = extract_scalar_from_event(event_file, x1_tag)
            _, y1_values = extract_scalar_from_event(event_file, y1_tag)
            _, x2_values = extract_scalar_from_event(event_file, x2_tag)
            _, y2_values = extract_scalar_from_event(event_file, y2_tag)
            _, x3_values = extract_scalar_from_event(event_file, x3_tag)
            _, y3_values = extract_scalar_from_event(event_file, y3_tag)

            if event_type == "eval":
                # Only last checkpoint
                if x1_values and y1_values:
                    all_x1.append(x1_values[-1])
                    all_y1.append(y1_values[-1])
                if x2_values and y2_values:
                    all_x2.append(x2_values[-1])
                    all_y2.append(y2_values[-1])
                if x3_values and y3_values:
                    all_x3.append(x3_values[-1])
                    all_y3.append(y3_values[-1])
            else:
                # Last 100 blocks
                if x1_values and y1_values:
                    min_len = min(len(x1_values), len(y1_values))
                    last_n = min(100, min_len)
                    all_x1.extend(x1_values[-last_n:])
                    all_y1.extend(y1_values[-last_n:])
                if x2_values and y2_values:
                    min_len = min(len(x2_values), len(y2_values))
                    last_n = min(100, min_len)
                    all_x2.extend(x2_values[-last_n:])
                    all_y2.extend(y2_values[-last_n:])
                if x3_values and y3_values:
                    min_len = min(len(x3_values), len(y3_values))
                    last_n = min(100, min_len)
                    all_x3.extend(x3_values[-last_n:])
                    all_y3.extend(y3_values[-last_n:])

        if not (all_x1 or all_x2 or all_x3):
            continue

        # Plot settings
        if event_type == "eval":
            scatter_kwargs = {"alpha": 0.8, "s": 150, "edgecolors": "white", "linewidths": 0.5}
        else:
            scatter_kwargs = {"alpha": 0.5, "s": 20}

        # Plot on all three axes
        if all_x1:
            ax1.scatter(all_x1, all_y1, label=label, color=color, **scatter_kwargs)
        if all_x2:
            ax2.scatter(all_x2, all_y2, color=color, **scatter_kwargs)
        if all_x3:
            ax3.scatter(all_x3, all_y3, color=color, **scatter_kwargs)

        plotted += 1

    if plotted == 0:
        print(f"[WARN] No data for scatter plot triple")
        plt.close()
        return

    # Axis labels and titles
    ax1.set_xlabel(x1_label)
    ax1.set_ylabel(y1_label)
    title1 = f"{title_prefix}: {y1_label} vs {x1_label}" if title_prefix else f"{y1_label} vs {x1_label}"
    ax1.set_title(title1)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel(x2_label)
    ax2.set_ylabel(y2_label)
    title2 = f"{title_prefix}: {y2_label} vs {x2_label}" if title_prefix else f"{y2_label} vs {x2_label}"
    ax2.set_title(title2)
    ax2.grid(True, alpha=0.3)

    ax3.set_xlabel(x3_label)
    ax3.set_ylabel(y3_label)
    title3 = f"{title_prefix}: {y3_label} vs {x3_label}" if title_prefix else f"{y3_label} vs {x3_label}"
    ax3.set_title(title3)
    ax3.grid(True, alpha=0.3)

    # Shared legend below all plots
    handles, labels = ax1.get_legend_handles_labels()
    legend_scale = 1 if event_type == "eval" else 2.5
    n_scenarios = len(handles)
    ncol = optimal_legend_ncol(n_scenarios)
    legend_height = 0.08 + 0.04 * ((n_scenarios - 1) // ncol)
    fig.legend(handles, labels, loc="lower center", ncol=ncol, frameon=False, fontsize=12, markerscale=legend_scale)
    fig.tight_layout(rect=[0, legend_height, 1, 1])

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[INFO] Saved: {save_path}")
    if show:
        plt.show()
    else:
        plt.close()


def plot_scatter_2d(
    base_dir: str,
    x_tag: str,
    y_tag: str,
    xlabel: str,
    ylabel: str,
    title: str,
    save_path: str | None = None,
    show: bool = True,
    event_type: str = "training",
    use_block_tags: bool = True,
    max_points: int = 5000,
    scenario_filter: list[str] | None = None,
):
    """
    Creates a scatter plot for two metrics.

    Data points:
    - Training: Last 100 blocks per run, all seeds combined
      → e.g., 5 seeds × 100 blocks = 500 points per scenario
    - Eval: Only last checkpoint per run (agent finished training), all seeds combined
      → e.g., 5 seeds × 1 checkpoint = 5 aggregated points per scenario
      → Each point is the average over 30 episodes

    Args:
        base_dir: Base directory with runs
        x_tag: TensorBoard tag for X-axis
        y_tag: TensorBoard tag for Y-axis
        xlabel: X-axis label
        ylabel: Y-axis label
        title: Plot title
        save_path: Save path for plot
        show: Show plot
        event_type: "training" or "eval"
        use_block_tags: Block/* or Rollout/* tags
        max_points: Maximum number of points per scenario (downsampling)
        scenario_filter: List of scenario names to plot (None = all)
    """
    if save_path:
        _ensure_dir(os.path.dirname(save_path))

    plt.figure(figsize=(10, 8))
    runs = discover_runs(base_dir)

    if not runs:
        print(f"[WARN] No runs found in {base_dir}")
        return

    plotted = 0

    for scenario_name, run_dirs in sorted(runs.items(), key=lambda x: _scenario_sort_key(x[0])):
        # Filter scenarios if specified
        if scenario_filter is not None and scenario_name not in scenario_filter:
            continue
        all_x = []
        all_y = []

        for run_dir in run_dirs:
            if event_type == "training":
                event_file = get_training_event_file(run_dir)
            else:
                event_file = get_eval_event_file(run_dir)

            if event_file is None:
                continue

            _, x_values = extract_scalar_from_event(event_file, x_tag)
            _, y_values = extract_scalar_from_event(event_file, y_tag)

            if x_values and y_values:
                # Training: take last 100 blocks per run
                # Eval: take only last checkpoint (agent finished training)
                #       This is 1 aggregated value over 30 episodes per run
                if event_type == "eval":
                    # Only last checkpoint
                    all_x.append(x_values[-1])
                    all_y.append(y_values[-1])
                else:
                    # Last 100 blocks
                    min_len = min(len(x_values), len(y_values))
                    last_n = min(100, min_len)
                    all_x.extend(x_values[-last_n:])
                    all_y.extend(y_values[-last_n:])

        if not all_x:
            continue

        # Downsampling if too many points
        if len(all_x) > max_points:
            indices = np.random.choice(len(all_x), max_points, replace=False)
            all_x = [all_x[i] for i in indices]
            all_y = [all_y[i] for i in indices]

        label = _nice_label(scenario_name)

        # Eval: Large, clearly visible points (only ~5 points per scenario)
        # Training: Smaller, transparent points (many points)
        if event_type == "eval":
            plt.scatter(all_x, all_y, label=label, alpha=0.8, s=150, edgecolors="white", linewidths=0.5)
        else:
            plt.scatter(all_x, all_y, label=label, alpha=0.5, s=20)
        plotted += 1

    if plotted == 0:
        print(f"[WARN] No data for scatter plot")
        plt.close()
        return

    # Eval: Smaller markers in legend (points in plot are large)
    # Training: Larger markers in legend (points in plot are small)
    legend_scale = 1 if event_type == "eval" else 2.5
    ncol = optimal_legend_ncol(plotted)
    plt.legend(markerscale=legend_scale, ncol=ncol)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[INFO] Saved: {save_path}")
    if show:
        plt.show()
    else:
        plt.close()


def plot_eval_summary(
    base_dir: str,
    metrics: list[str] = None,
    max_blocks: int = 1000,
    save_path: str | None = None,
    show: bool = True,
    ci_level: float = 0.95,
    use_block_tags: bool = False,
    smooth: int = 1,
    **kwargs,
):
    """
    Creates a multi-panel plot for multiple eval metrics.

    Args:
        metrics: List of metrics (default: ["score", "reward", "move_success_rate"])
        use_block_tags: True for Block/* tags, False for Rollout/* (for collision_rate)
        smooth: Smoothing window for training metrics (e.g., move_success_rate)
    """
    if metrics is None:
        metrics = ["score", "reward", "move_success_rate"]

    if save_path:
        _ensure_dir(os.path.dirname(save_path))

    n_metrics = len(metrics)
    # Same size as training summary for consistency
    width_per_metric = 10 if n_metrics == 1 else 5
    fig, axes = plt.subplots(1, n_metrics, figsize=(width_per_metric * n_metrics, 5), squeeze=False)
    axes = axes[0]

    runs = discover_runs(base_dir)

    if not runs:
        print(f"[WARN] No runs found in {base_dir}")
        return

    ylabels = {
        "score": "Node Recall",
        "steps": "Episode Length",
        "reward": "Reward",
        "path_length": "Path Length",
        "coverage": "Coverage",
        "collision_rate": "Collision Rate",
        "move_success_rate": "Move Success Rate",
    }
    titles = {
        "score": "Node Recall",
        "steps": "Episode Length",
        "reward": "Reward",
        "path_length": "Path Length",
        "coverage": "Coverage",
        "collision_rate": "Collision Rate",
        "move_success_rate": "Move Success Rate",
    }

    z = get_z_value(ci_level)

    # Color palette - use consistent colors per scenario
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for ax, metric in zip(axes, metrics):
        # Special handling for move_success_rate (from training data)
        if metric == "move_success_rate":
            for idx, (scenario_name, run_dirs) in enumerate(sorted(runs.items(), key=lambda x: _scenario_sort_key(x[0]))):
                data = aggregate_move_success_rate(run_dirs, "training")
                if data is None:
                    continue

                steps = data["steps"]
                mean = data["mean"]
                std = data["std"]
                n = data["n_seeds"]

                se = std / max(1, np.sqrt(n))
                ci_half = z * se

                label = _nice_label(scenario_name)
                color = colors[idx % len(colors)]
                linestyle = SCENARIO_LINESTYLES.get(scenario_name, "-")

                # Apply same smoothing as training summary
                if smooth > 1:
                    mean_plot = moving_average(mean, smooth)
                    x_plot = steps[: len(mean_plot)]
                    lower = moving_average(mean - ci_half, smooth)
                    upper = moving_average(mean + ci_half, smooth)
                else:
                    mean_plot = mean
                    x_plot = steps
                    lower = mean - ci_half
                    upper = mean + ci_half

                ax.plot(x_plot, mean_plot, label=f"{label}", linewidth=2, linestyle=linestyle, color=color)
                ax.fill_between(x_plot, lower[: len(x_plot)], upper[: len(x_plot)], alpha=0.15, color=color)

            ax.set_xlabel("Block")
            ax.set_ylabel(ylabels.get(metric, metric))
            ax.set_title(titles.get(metric, metric))
            ax.set_xlim([0, max_blocks])
            continue

        # Check if metric exists in EVAL_TAGS
        if metric not in EVAL_TAGS:
            ax.set_title(f"Unknown: {metric}")
            continue

        tag = EVAL_TAGS[metric]
        event_type = "eval"
        needs_rollout_smooth = False

        for idx, (scenario_name, run_dirs) in enumerate(sorted(runs.items(), key=lambda x: _scenario_sort_key(x[0]))):
            data = aggregate_scenario_seeds(run_dirs, tag, event_type)
            if data is None:
                continue

            steps = data["steps"]
            mean = data["mean"]
            std = data["std"]
            n = data["n_seeds"]

            # Apply rollout-window smoothing for unsmoothed training metrics
            if needs_rollout_smooth:
                mean = causal_smooth(mean, DEFAULT_ROLLOUT_WINDOW)
                std = causal_smooth(std, DEFAULT_ROLLOUT_WINDOW)
            elif event_type == "eval":
                # Light smoothing for eval (few data points)
                mean = causal_smooth(mean, 3)
                std = causal_smooth(std, 3)

            se = std / max(1, np.sqrt(n))
            ci_half = z * se

            label = _nice_label(scenario_name)
            color = colors[idx % len(colors)]
            linestyle = SCENARIO_LINESTYLES.get(scenario_name, "-")

            # Apply additional smoothing (same as training summary)
            if smooth > 1 and metric == "collision_rate":
                mean_plot = moving_average(mean, smooth)
                x_plot = steps[: len(mean_plot)]
                lower = moving_average(mean - ci_half, smooth)
                upper = moving_average(mean + ci_half, smooth)
            else:
                mean_plot = mean
                x_plot = steps
                lower = mean - ci_half
                upper = mean + ci_half

            ax.plot(x_plot, mean_plot, label=f"{label}", linewidth=2, linestyle=linestyle, color=color)
            ax.fill_between(x_plot, lower[: len(x_plot)], upper[: len(x_plot)], alpha=0.15, color=color)

        ax.set_xlabel("Block")
        ax.set_ylabel(ylabels.get(metric, metric))
        ax.set_title(titles.get(metric, metric))
        ax.set_xlim([0, max_blocks])

    # Shared legend with optimal layout (max 4 columns)
    handles, labels = axes[0].get_legend_handles_labels()
    n_scenarios = len(handles)
    ncol = optimal_legend_ncol(n_scenarios, max_ncol=4)
    legend_height = 0.08 + 0.04 * ((n_scenarios - 1) // ncol)
    fig.legend(handles, labels, loc="lower center", ncol=ncol, frameon=False, fontsize=12)
    fig.tight_layout(rect=[0, legend_height, 1, 1])

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[INFO] Saved: {save_path}")
    if show:
        plt.show()
    else:
        plt.close()


# ---- Table Export -----------------------------------------------------------


def export_final_values(
    base_dir: str, tag: str, last_n: int = 200, event_type: str = "training", output_file: str | None = None
) -> pd.DataFrame:
    """
    Calculates final values (average of last N values) for all scenarios.

    If fewer than last_n data points are available, all are used.

    Returns:
        DataFrame with columns: Scenario, Mean, Std, Num_Seeds, Num_Points
    """
    results = []
    runs = discover_runs(base_dir)

    for scenario_name, run_dirs in sorted(runs.items(), key=lambda x: _scenario_sort_key(x[0])):
        data = aggregate_scenario_seeds(run_dirs, tag, event_type)
        if data is None or len(data["mean"]) == 0:
            continue

        # Use min(last_n, available points)
        actual_n = min(last_n, len(data["mean"]))
        mean_last = float(np.mean(data["mean"][-actual_n:]))
        std_last = float(np.std(data["mean"][-actual_n:]))

        results.append(
            {
                "Scenario": _nice_label(scenario_name),
                "Mean": mean_last,
                "Std": std_last,
                "Num_Seeds": data["n_seeds"],
                "Num_Points": actual_n,
            }
        )

    df = pd.DataFrame(results)
    if not df.empty:
        df = df.sort_values(by="Mean", ascending=False).reset_index(drop=True)

    if output_file:
        _ensure_dir(os.path.dirname(output_file))
        df.to_csv(output_file, index=False)
        print(f"[INFO] Tabelle gespeichert: {output_file}")

    return df


def export_training_tables(
    base_dir: str, last_n: int = 20, output_dir: str = "RL_training/runs/abb", use_block_tags: bool = False
) -> dict[str, pd.DataFrame]:
    """Exports training tables for all metrics (last 20 blocks by default)."""
    _ensure_dir(output_dir)
    tags = BLOCK_TAGS if use_block_tags else ROLLOUT_TAGS

    tables = {}
    for metric, tag in tags.items():
        output_file = os.path.join(output_dir, f"training_{metric}.csv")
        df = export_final_values(base_dir, tag, last_n, "training", output_file)
        tables[metric] = df

    # Add Move Success Rate table
    move_success_results = []
    runs = discover_runs(base_dir)
    for scenario_name, run_dirs in sorted(runs.items(), key=lambda x: _scenario_sort_key(x[0])):
        data = aggregate_move_success_rate(run_dirs, "training")
        if data is None or len(data["mean"]) == 0:
            continue

        actual_n = min(last_n, len(data["mean"]))
        mean_last = float(np.mean(data["mean"][-actual_n:]))
        std_last = float(np.mean(data["std"][-actual_n:]))

        move_success_results.append(
            {
                "Scenario": _nice_label(scenario_name),
                "Mean": mean_last,
                "Std": std_last,
                "Num_Seeds": data["n_seeds"],
                "Num_Points": actual_n,
            }
        )

    df_move_success = pd.DataFrame(move_success_results)
    if not df_move_success.empty:
        df_move_success = df_move_success.sort_values(by="Mean", ascending=False).reset_index(drop=True)
        output_file = os.path.join(output_dir, "training_move_success_rate.csv")
        df_move_success.to_csv(output_file, index=False)
        print(f"[INFO] Tabelle gespeichert: {output_file}")
    tables["move_success_rate"] = df_move_success

    return tables


def export_eval_tables(
    base_dir: str, last_n: int = 1, output_dir: str = "RL_training/runs/abb", add_move_success=False
) -> dict[str, pd.DataFrame]:
    """Exports eval tables for all metrics (last 1 block by default)."""
    _ensure_dir(output_dir)

    tables = {}
    for metric in ["score", "steps", "reward"]:
        tag = EVAL_TAGS[metric]
        output_file = os.path.join(output_dir, f"eval_{metric}.csv")
        df = export_final_values(base_dir, tag, last_n, "eval", output_file)
        tables[metric] = df

    if add_move_success:
        # Add Move Success Rate table (using training data since eval doesn't have action types logged)
        move_success_results = []
        runs = discover_runs(base_dir)

        for scenario_name, run_dirs in sorted(runs.items(), key=lambda x: _scenario_sort_key(x[0])):
            data = aggregate_move_success_rate(run_dirs, "training")
            if data is None or len(data["mean"]) == 0:
                continue

            actual_n = min(last_n, len(data["mean"]))
            mean_last = float(np.mean(data["mean"][-actual_n:]))
            std_last = float(np.mean(data["std"][-actual_n:]))

            move_success_results.append(
                {
                    "Scenario": _nice_label(scenario_name),
                    "Mean": mean_last,
                    "Std": std_last,
                    "Num_Seeds": data["n_seeds"],
                    "Num_Points": actual_n,
                }
            )

        df_move_success = pd.DataFrame(move_success_results)
        if not df_move_success.empty:
            df_move_success = df_move_success.sort_values(by="Mean", ascending=False).reset_index(drop=True)
            output_file = os.path.join(output_dir, "eval_move_success_rate.csv")
            df_move_success.to_csv(output_file, index=False)
            print(f"[INFO] Tabelle gespeichert: {output_file}")
        tables["move_success_rate"] = df_move_success

    return tables


def export_post_train_eval_tables(base_dir: str, output_dir: str = "RL_training/runs/abb") -> dict[str, pd.DataFrame]:
    """
    Exports post-training evaluation tables for all metrics.

    Post-training eval is run once per trained agent on final weights.
    Data is stored centrally in {base_dir}/post_train_eval/{scenario_name}/

    Args:
        base_dir: Base directory with runs
        output_dir: Output directory for CSV tables

    Returns:
        dict: {metric_name: DataFrame}
    """
    _ensure_dir(output_dir)

    runs = discover_runs(base_dir)
    if not runs:
        print(f"[WARN] No runs found in {base_dir}")
        return {}

    tables = {}

    # Export each metric
    for metric_name, metric_tag in POST_TRAIN_EVAL_TAGS.items():
        results = []

        # Special handling for movement_success_rate (cumulative calculation)
        if metric_name == "movement_success_rate":
            for scenario_name, run_dirs in sorted(runs.items(), key=lambda x: _scenario_sort_key(x[0])):
                data = aggregate_post_train_eval_movement_success_rate(base_dir, scenario_name)
                if data is None:
                    continue

                results.append(
                    {
                        "Scenario": _nice_label(scenario_name),
                        "Rate": data["rate"],
                        "Total_Movements": data["total_movements"],
                        "Total_Successful": data["total_successful"],
                        "Num_Episodes": data["n_episodes"],
                    }
                )

            df = pd.DataFrame(results)
            if not df.empty:
                df = df.sort_values(by="Rate", ascending=False).reset_index(drop=True)
                output_file = os.path.join(output_dir, f"post_train_eval_{metric_name}.csv")
                df.to_csv(output_file, index=False)
                print(f"[INFO] Tabelle gespeichert: {output_file}")

        else:
            # Standard metrics (mean/std)
            for scenario_name, run_dirs in sorted(runs.items(), key=lambda x: _scenario_sort_key(x[0])):
                data = aggregate_post_train_eval(base_dir, scenario_name, metric_tag)
                if data is None:
                    continue

                results.append(
                    {"Scenario": _nice_label(scenario_name), "Mean": data["mean"], "Std": data["std"], "Num_Episodes": data["n_episodes"]}
                )

            df = pd.DataFrame(results)
            if not df.empty:
                # Sort by mean value (descending for score/coverage, ascending for steps/path_length)
                ascending = metric_name in ["steps", "path_length"]
                df = df.sort_values(by="Mean", ascending=ascending).reset_index(drop=True)

                output_file = os.path.join(output_dir, f"post_train_eval_{metric_name}.csv")
                df.to_csv(output_file, index=False)
                print(f"[INFO] Tabelle gespeichert: {output_file}")

        tables[metric_name] = df

    return tables


# ---- Metrics Summary --------------------------------------------------------


def print_metrics_summary(base_dir: str, last_n: int = 100, event_type: str = "training", use_block_tags: bool = False) -> pd.DataFrame:
    """
    Prints a clear table of all metrics for all scenarios.

    Args:
        base_dir: Base directory with runs
        last_n: Number of last blocks for calculation
        event_type: "training" or "eval"
        use_block_tags: True for Block/* tags, False for Rollout/*

    Returns:
        DataFrame with all metrics
    """
    runs = discover_runs(base_dir)
    if not runs:
        print(f"[WARN] No runs found in {base_dir}")
        return pd.DataFrame()

    # Choose tags based on event_type
    if event_type == "eval":
        tags = EVAL_TAGS
    else:
        tags = BLOCK_TAGS if use_block_tags else ROLLOUT_TAGS

    results = []

    for scenario_name, run_dirs in sorted(runs.items(), key=lambda x: _scenario_sort_key(x[0])):
        row = {"Scenario": _nice_label(scenario_name), "Seeds": len(run_dirs)}

        for metric_name, tag in tags.items():
            data = aggregate_scenario_seeds(run_dirs, tag, event_type)
            if data is None or len(data["mean"]) == 0:
                row[f"{metric_name}_mean"] = np.nan
                row[f"{metric_name}_std"] = np.nan
                row[f"{metric_name}_min"] = np.nan
                row[f"{metric_name}_max"] = np.nan
                row[f"{metric_name}_trend"] = np.nan
                continue

            mean_vals = data["mean"]
            std_vals = data["std"]
            all_runs = data["all_runs"]
            actual_n = min(last_n, len(mean_vals))

            # Last N values
            last_vals = mean_vals[-actual_n:]
            last_stds = std_vals[-actual_n:]

            # Mean and std of last N
            # mean: average over time (and already averaged over seeds)
            # std: average of std over seeds (not std over time!)
            row[f"{metric_name}_mean"] = float(np.mean(last_vals))
            row[f"{metric_name}_std"] = float(np.mean(last_stds))

            # Min/Max of last N: across all seeds and time
            last_runs = all_runs[:, -actual_n:]
            row[f"{metric_name}_min"] = float(np.min(last_runs))
            row[f"{metric_name}_max"] = float(np.max(last_runs))

            # Trend: compare first half vs second half of last N
            if actual_n >= 10:
                half = actual_n // 2
                first_half = np.mean(last_vals[:half])
                second_half = np.mean(last_vals[half:])
                trend = second_half - first_half
                row[f"{metric_name}_trend"] = float(trend)
            else:
                row[f"{metric_name}_trend"] = np.nan

        # Calculate derived metrics (Score/Path, Coverage/Path for eval)
        if "score" in tags and "path_length" in tags:
            score_path_data = aggregate_derived_metric(run_dirs, tags["score"], tags["path_length"], event_type)
            if score_path_data is not None and len(score_path_data["mean"]) > 0:
                actual_n = min(last_n, len(score_path_data["mean"]))
                last_vals = score_path_data["mean"][-actual_n:]
                last_stds = score_path_data["std"][-actual_n:]
                row["score_per_path_mean"] = float(np.mean(last_vals))
                row["score_per_path_std"] = float(np.mean(last_stds))
            else:
                row["score_per_path_mean"] = np.nan
                row["score_per_path_std"] = np.nan

        # Coverage/Path for eval
        if event_type == "eval" and "coverage" in tags and "path_length" in tags:
            cov_path_data = aggregate_derived_metric(run_dirs, tags["coverage"], tags["path_length"], event_type)
            if cov_path_data is not None and len(cov_path_data["mean"]) > 0:
                actual_n = min(last_n, len(cov_path_data["mean"]))
                last_vals = cov_path_data["mean"][-actual_n:]
                last_stds = cov_path_data["std"][-actual_n:]
                row["coverage_per_path_mean"] = float(np.mean(last_vals))
                row["coverage_per_path_std"] = float(np.mean(last_stds))
            else:
                row["coverage_per_path_mean"] = np.nan
                row["coverage_per_path_std"] = np.nan

        # Move Success Rate (only for training, when action data is available)
        if event_type == "training":
            move_success_data = aggregate_move_success_rate(run_dirs, event_type)
            if move_success_data is not None and len(move_success_data["mean"]) > 0:
                actual_n = min(last_n, len(move_success_data["mean"]))
                last_vals = move_success_data["mean"][-actual_n:]
                last_stds = move_success_data["std"][-actual_n:]
                row["move_success_rate_mean"] = float(np.mean(last_vals))
                row["move_success_rate_std"] = float(np.mean(last_stds))
            else:
                row["move_success_rate_mean"] = np.nan
                row["move_success_rate_std"] = np.nan

        results.append(row)

    df = pd.DataFrame(results)

    # Formatted output
    print(f"\n{'='*80}")
    print(f"METRICS SUMMARY (last {last_n} blocks, {event_type})")
    print(f"{'='*80}\n")

    # One separate table per metric (clearer)
    for metric_name in tags.keys():
        cols = ["Scenario", "Seeds"]
        metric_cols = [c for c in df.columns if c.startswith(f"{metric_name}_")]
        if not metric_cols:
            continue

        # Rename columns for display
        display_df = df[cols + metric_cols].copy()
        display_df.columns = ["Scenario", "Seeds"] + [c.replace(f"{metric_name}_", "") for c in metric_cols]

        # Sort by mean (descending for score/reward, ascending for steps/collision)
        sort_ascending = metric_name in ["steps", "collision_rate"]
        display_df = display_df.sort_values(by="mean", ascending=sort_ascending)

        print(f"\n--- {metric_name.upper()} ---")
        # Format numbers
        pd.set_option("display.float_format", lambda x: f"{x:.4f}" if abs(x) < 100 else f"{x:.1f}")
        print(display_df.to_string(index=False))

    # Derived metrics (efficiency)
    if "score_per_path_mean" in df.columns:
        cols = ["Scenario", "Seeds"]
        metric_cols = [c for c in df.columns if c.startswith("score_per_path_")]
        if metric_cols:
            display_df = df[cols + metric_cols].copy()
            display_df.columns = ["Scenario", "Seeds"] + [c.replace("score_per_path_", "") for c in metric_cols]
            display_df = display_df.sort_values(by="mean", ascending=False)

            print(f"\n--- SCORE PER PATH (EFFICIENCY) ---")
            pd.set_option("display.float_format", lambda x: f"{x:.4f}")
            print(display_df.to_string(index=False))

    if "coverage_per_path_mean" in df.columns:
        cols = ["Scenario", "Seeds"]
        metric_cols = [c for c in df.columns if c.startswith("coverage_per_path_")]
        if metric_cols:
            display_df = df[cols + metric_cols].copy()
            display_df.columns = ["Scenario", "Seeds"] + [c.replace("coverage_per_path_", "") for c in metric_cols]
            display_df = display_df.sort_values(by="mean", ascending=False)

            print(f"\n--- COVERAGE PER PATH (EFFICIENCY) ---")
            pd.set_option("display.float_format", lambda x: f"{x:.4f}")
            print(display_df.to_string(index=False))

    # Move Success Rate (fairer than overall collision_rate)
    if "move_success_rate_mean" in df.columns:
        cols = ["Scenario", "Seeds"]
        metric_cols = [c for c in df.columns if c.startswith("move_success_rate_")]
        if metric_cols:
            display_df = df[cols + metric_cols].copy()
            display_df.columns = ["Scenario", "Seeds"] + [c.replace("move_success_rate_", "") for c in metric_cols]
            display_df = display_df.sort_values(by="mean", ascending=False)

            print(f"\n--- MOVE SUCCESS RATE (FAIRER THAN COLLISION RATE) ---")
            pd.set_option("display.float_format", lambda x: f"{x:.4f}")
            print(display_df.to_string(index=False))

    print(f"\n{'='*80}\n")

    return df


# ---- CLI --------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Aggregates and visualizes RL train/eval runs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python aggregate_runs.py                           # Everything with defaults (plots + summary at end)
  python aggregate_runs.py --mode eval               # Only eval plots + summary
  python aggregate_runs.py --smooth 50               # Block data with custom smoothing
  python aggregate_runs.py --no_raw                  # Rollout/* tags (already smoothed)
  python aggregate_runs.py --metrics score,steps     # Only specific metrics
  python aggregate_runs.py --efficiency              # Enable efficiency plots
  python aggregate_runs.py --no_scatter              # Disable scatter plots
  python aggregate_runs.py --summary_only            # Only summary table, no plots
  python aggregate_runs.py --no_summary              # Plots without summary table
        """,
    )

    parser.add_argument("--base_dir", default="RL_training/runs", help="Directory with runs (default: RL_training/runs)")
    parser.add_argument(
        "--output_dir", default="RL_training/runs/abb", help="Output directory for plots/CSV (default: RL_training/runs/abb)"
    )
    parser.add_argument("--mode", choices=["training", "eval", "both"], default="eval", help="Which metrics to plot (default: both)")
    parser.add_argument(
        "--metrics", default="score,reward,move_success_rate", help="Comma-separated metrics (default: score,reward,move_success_rate)"
    )
    parser.add_argument("--max_blocks", type=int, default=1000, help="Maximum for X-axis (default: 1000)")
    parser.add_argument("--raw", dest="raw", action="store_true", default=True, help="Use Block/* tags (default: True)")
    parser.add_argument("--no-raw", dest="raw", action="store_false", help="Use Rollout/* tags instead of Block/*")
    parser.add_argument("--smooth", type=int, default=100, help="Smoothing window, only active with --raw (default: 100)")
    parser.add_argument("--tables", action="store_true", help="Also export CSV tables")
    parser.add_argument("--last_n", type=int, default=200, help="Number of last values for tables (default: 200)")
    parser.add_argument("--no_show", action="store_true", help="Do not show plots, only save")
    parser.add_argument("--summary", action="store_true", default=True, help="Output metrics summary table (default: True)")
    parser.add_argument("--no_summary", dest="summary", action="store_false", help="Do not output summary table")
    parser.add_argument("--summary_only", action="store_true", help="Only output metrics summary table (no plots)")
    parser.add_argument("--summary_n", type=int, default=100, help="Number of last blocks for summary (default: 100)")
    parser.add_argument("--no_validate_configs", action="store_true", help="Skip config validation between runs")
    parser.add_argument("--efficiency", action="store_true", help="Enable efficiency plots (Score/Path)")
    parser.add_argument("--no_scatter", action="store_true", help="Disable scatter plots")

    args = parser.parse_args()

    metrics = [m.strip() for m in args.metrics.split(",")]
    show = not args.no_show
    validate_configs = not args.no_validate_configs
    use_block_tags = args.raw

    # Apply smoothing only with --raw
    smooth = args.smooth if args.raw else 1

    _ensure_dir(args.output_dir)

    print(f"Base Dir: {args.base_dir}")
    print(f"Output Dir: {args.output_dir}")
    print(f"Mode: {args.mode}")
    print(f"Metrics: {metrics}")
    print(f"Raw (Block Tags): {args.raw}")
    print(f"Smooth: {smooth}")
    print()

    # Summary-only mode: output only tables, no plots
    if args.summary_only:
        if args.mode in ["training", "both"]:
            print_metrics_summary(args.base_dir, last_n=args.summary_n, event_type="training", use_block_tags=args.raw)
        if args.mode in ["eval", "both"]:
            # For eval, only use the last block (agent is most trained)
            print_metrics_summary(args.base_dir, last_n=1, event_type="eval")

        return

    # Training plots
    if args.mode in ["training", "both"]:
        print("=== Training Plots ===")
        plot_training_summary(
            args.base_dir,
            metrics=metrics,
            max_blocks=args.max_blocks,
            smooth=smooth,
            save_path=os.path.join(args.output_dir, "training_summary.pdf"),
            show=show,
            use_block_tags=args.raw,
        )

        if args.tables:
            print("\n=== Training Tables ===")
            tables = export_training_tables(args.base_dir, last_n=args.last_n, output_dir=args.output_dir, use_block_tags=args.raw)
            for metric, df in tables.items():
                if not df.empty:
                    print(f"\n{metric}:")
                    print(df.to_string(index=False))

    # Eval plots
    if args.mode in ["eval", "both"]:
        print("\n=== Eval Plots ===")
        plot_eval_summary(
            args.base_dir,
            metrics=metrics,  # Use same metrics as training
            max_blocks=args.max_blocks,
            save_path=os.path.join(args.output_dir, "eval_summary.pdf"),
            show=show,
            use_block_tags=args.raw,
            smooth=smooth,
        )

        if args.tables:
            print("\n=== Eval Tables ===")
            # For eval, only use the last block (agent is most trained)
            tables = export_eval_tables(args.base_dir, last_n=1, output_dir=args.output_dir)
            for metric, df in tables.items():
                if not df.empty:
                    print(f"\n{metric}:")
                    print(df.to_string(index=False))

            print("\n=== Post-Training Eval Tables ===")
            post_tables = export_post_train_eval_tables(args.base_dir, output_dir=args.output_dir)
            for metric, df in post_tables.items():
                if not df.empty:
                    print(f"\n{metric}:")
                    print(df.to_string(index=False))

    # Training vs eval comparison (reward)
    if args.mode == "both":
        print("\n=== Train vs Eval Comparison ===")
        plot_training_vs_eval(
            args.base_dir,
            metric="reward",
            max_blocks=args.max_blocks,
            smooth=smooth,
            save_path=os.path.join(args.output_dir, "training_vs_eval_reward.pdf"),
            show=show,
            use_block_tags=use_block_tags,
        )

    # Efficiency plots (Score/Path, Coverage/Path)
    if args.efficiency:
        print("\n=== Efficiency Plots ===")

        # Training: Score / path length
        if args.mode in ["training", "both"]:
            tags = BLOCK_TAGS if use_block_tags else ROLLOUT_TAGS
            plot_derived_metric(
                args.base_dir,
                numerator_tag=tags["score"],
                denominator_tag=tags["path_length"],
                ylabel="Score / Path Length",
                title="Training Efficiency: Score per Path Length",
                max_blocks=args.max_blocks,
                smooth=smooth,
                save_path=os.path.join(args.output_dir, "efficiency_score_per_path.pdf"),
                show=show,
                event_type="training",
                validate_configs=False,  # Already validated earlier
            )

        # Eval: Score / path length
        if args.mode in ["eval", "both"]:
            plot_derived_metric(
                args.base_dir,
                numerator_tag=EVAL_TAGS["score"],
                denominator_tag=EVAL_TAGS["path_length"],
                ylabel="Score / Path Length",
                title="Eval Efficiency: Score per Path Length",
                max_blocks=args.max_blocks,
                smooth=1,
                save_path=os.path.join(args.output_dir, "eval_efficiency_score_per_path.pdf"),
                show=show,
                event_type="eval",
                validate_configs=False,
            )

            # Coverage / path length (eval only)
            plot_derived_metric(
                args.base_dir,
                numerator_tag=EVAL_TAGS["coverage"],
                denominator_tag=EVAL_TAGS["path_length"],
                ylabel="Coverage / Path Length",
                title="Eval Efficiency: Coverage per Path Length",
                max_blocks=args.max_blocks,
                smooth=1,
                save_path=os.path.join(args.output_dir, "eval_efficiency_coverage_per_path.pdf"),
                show=show,
                event_type="eval",
                validate_configs=False,
            )

    # Scatter plots
    if not args.no_scatter:
        print("\n=== Scatter Plots ===")

        # Training: Node Recall & Episode Length vs path length (Baseline, S2, S3, S4, S7, S8)
        if args.mode in ["training", "both"]:
            tags = BLOCK_TAGS if use_block_tags else ROLLOUT_TAGS
            plot_scatter_pair(
                args.base_dir,
                x_tag=tags["path_length"],
                y1_tag=tags["score"],
                y2_tag=tags["steps"],
                xlabel="Path Length",
                y1_label="Node Recall",
                y2_label="Episode Length",
                title_prefix="Train",
                save_path=os.path.join(args.output_dir, "training_scatter.pdf"),
                show=show,
                event_type="training",
                use_block_tags=use_block_tags,
                scenario_filter=[
                    "baseline_scenario_reinforce_nodepth_il",  # Baseline
                    "scenario2_1_ppo_legacy_no_depth",  # S2 (PPO+SH16 without depth)
                    "scenario6_ppo_singlehead_no_curriculum",  # S4
                    "scenario4_ppo_multihead_curriculum",  # S9
                ],
            )

        # Eval: Node Recall & Episode Length vs path length (same scenarios as training)
        if args.mode in ["eval", "both"]:
            plot_scatter_pair(
                args.base_dir,
                x_tag=EVAL_TAGS["path_length"],
                y1_tag=EVAL_TAGS["score"],
                y2_tag=EVAL_TAGS["steps"],
                xlabel="Path Length",
                y1_label="Node Recall",
                y2_label="Episode Length",
                title_prefix="",
                save_path=os.path.join(args.output_dir, "eval_scatter.pdf"),
                show=show,
                event_type="eval",
                scenario_filter=[
                    "baseline_scenario_reinforce_nodepth_il",  # Baseline
                    "scenario2_1_ppo_legacy_no_depth",  # S2 (PPO+SH16 without depth)
                    "scenario6_ppo_singlehead_no_curriculum",  # S4
                    "scenario4_ppo_multihead_curriculum",  # S9
                ],
            )

    # Output summary tables at the end (after all plots)
    if args.summary:
        print("\n" + "=" * 80)
        print("METRICS SUMMARY")
        print("=" * 80)

        if args.mode in ["training", "both"]:
            print_metrics_summary(args.base_dir, last_n=args.summary_n, event_type="training", use_block_tags=args.raw)

        if args.mode in ["eval", "both"]:
            # For eval, only use the last block (agent is most trained)
            print_metrics_summary(args.base_dir, last_n=1, event_type="eval")


def set_working_directory():
    """Set working directory to project root."""
    desired_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
    current_directory = os.getcwd()

    if current_directory != desired_directory:
        os.chdir(desired_directory)
        sys.path.append(desired_directory)
        print(f"Working directory changed from '{current_directory}' to '{desired_directory}'")
        return

    print("Current working directory:", os.getcwd())


if __name__ == "__main__":
    set_working_directory()
    main()
