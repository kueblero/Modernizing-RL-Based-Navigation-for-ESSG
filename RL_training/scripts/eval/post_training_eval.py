#!/usr/bin/env python3
"""
Post-Training Evaluation Script

Re-evaluates trained agents using their final checkpoints and reports detailed metrics.
Tracks movement success rates, path lengths, and other metrics after training is complete.
Results are logged to TensorBoard in a separate 'post_train_eval' subdirectory for each run.

Usage:
    python post_training_eval.py                                    # Evaluate all runs
    python post_training_eval.py --base_dir RL_training/runs       # Custom base dir
    python post_training_eval.py --scenarios scenario2,scenario3   # Specific scenarios
    python post_training_eval.py --episodes 50                     # More episodes per scene
"""

import argparse
import gc
import json
import os
import re
import sys

import numpy as np
import pandas as pd
import torch
from tensorboard.backend.event_processing import event_accumulator
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def set_working_directory():
    """Set working directory to project root."""
    desired_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
    current_directory = os.getcwd()

    if current_directory != desired_directory:
        os.chdir(desired_directory)
        sys.path.append(desired_directory)
        print(f"Working directory changed to: {desired_directory}\n")


set_working_directory()

# Import after setting working directory
from components.agents.reinforce_agent import ReinforceAgent
from components.agents.ppo_agent import PPOAgent
from components.environments.precomputed_thor_env import PrecomputedThorEnv


def discover_completed_runs(base_dir: str) -> dict[str, list[str]]:
    """
    Finds completed training runs with checkpoints.

    Args:
        base_dir: Base directory with runs

    Returns:
        dict: {scenario_name: [run_dir1, run_dir2, ...]}
    """
    runs = {}

    if not os.path.isdir(base_dir):
        print(f"[ERROR] Directory not found: {base_dir}")
        return runs

    for entry in os.listdir(base_dir):
        run_dir = os.path.join(base_dir, entry)
        if not os.path.isdir(run_dir):
            continue

        # Parse "{scenario}_seed_{seed}_{timestamp}" pattern
        match = re.match(r"(.+)_seed_\d+_\d{8}_\d{6}$", entry)
        if not match:
            continue

        scenario_name = match.group(1)

        # Check if checkpoint exists
        checkpoint_dir = os.path.join(run_dir, "checkpoints")
        if not os.path.isdir(checkpoint_dir):
            continue

        # Check for final_model.pth or agent_block_*.pt
        final_model_path = os.path.join(checkpoint_dir, "final_model.pth")
        checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith("agent_block_") and f.endswith(".pt")]

        if not os.path.exists(final_model_path) and not checkpoints:
            continue

        if scenario_name not in runs:
            runs[scenario_name] = []
        runs[scenario_name].append(run_dir)

    # Sort run directories per scenario
    for scenario in runs:
        runs[scenario] = sorted(runs[scenario])

    return runs


def find_last_checkpoint(run_dir: str) -> str | None:
    """Find the last checkpoint in a run directory."""
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    if not os.path.isdir(checkpoint_dir):
        return None

    # Check for final_model.pth first
    final_model_path = os.path.join(checkpoint_dir, "final_model.pth")
    if os.path.exists(final_model_path):
        return final_model_path

    # Fall back to agent_block_*.pt pattern
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith("agent_block_") and f.endswith(".pt")]
    if not checkpoints:
        return None

    # Extract block numbers and sort
    checkpoint_blocks = []
    for ckpt in checkpoints:
        match = re.match(r"agent_block_(\d+)\.pt", ckpt)
        if match:
            block_num = int(match.group(1))
            checkpoint_blocks.append((block_num, ckpt))

    checkpoint_blocks.sort(reverse=True)  # Descending
    last_checkpoint = checkpoint_blocks[0][1]

    return os.path.join(checkpoint_dir, last_checkpoint)


def find_event_file(run_dir: str) -> str | None:
    """Find the training event file in run directory (contains config)."""
    # Look in main run directory for event file with config
    for f in os.listdir(run_dir):
        if f.startswith("events.out.tfevents."):
            return os.path.join(run_dir, f)
    return None


def get_config_from_event(event_path: str) -> dict | None:
    """Extract config from TensorBoard event file."""
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


def load_config_from_run(run_dir: str) -> dict | None:
    """Load config from run directory (from event file)."""
    event_file = find_event_file(run_dir)
    if event_file is None:
        return None

    return get_config_from_event(event_file)


def create_agent_from_config(config: dict, device: torch.device):
    """Create agent from config."""
    agent_config = config["agent"]
    navigation_config = config["navigation"]
    # Support both "env" and "environment" keys for backwards compatibility
    env_config = config.get("env") or config.get("environment")
    if env_config is None:
        raise ValueError("Config must contain either 'env' or 'environment' key")

    # Determine correct curriculum stage
    use_legacy_actions = env_config.get("use_legacy_actions", False)
    training_config = config.get("training", {})
    curriculum_config = training_config.get("curriculum", {})

    if curriculum_config.get("enabled", False) and not use_legacy_actions:
        # If curriculum was enabled during training, use the maximum stage
        curriculum_stage = curriculum_config.get("n_stages", 4)
    else:
        # No curriculum or legacy actions: use config value
        curriculum_stage = env_config.get("curriculum_stage", 1)

    # Create dummy environment for agent initialization
    # IMPORTANT: Use the same curriculum stage that will be used for evaluation!
    # Otherwise the agent will have wrong action masking.
    dummy_env = PrecomputedThorEnv(
        scene_number=1,  # Dummy scene
        rho=env_config["rho"],
        max_actions=env_config["max_actions"],
        use_lmdb=True,
        render=False,
        use_legacy_actions=use_legacy_actions,
        action_space_mode=env_config.get("action_space_mode", "multi_head"),
        curriculum_stage=curriculum_stage,  # Use computed stage!
        stop_stagnation_steps=env_config.get("stop_stagnation_steps", 5),
        stop_stagnation_bonus=env_config.get("stop_stagnation_bonus", 0.02),
    )

    # Create agent based on algorithm
    agent_name = agent_config.get("name", "").lower()

    if agent_name == "ppo":
        agent = PPOAgent(env=dummy_env, navigation_config=navigation_config, agent_config=agent_config, device=device)
    elif agent_name == "reinforce":
        agent = ReinforceAgent(env=dummy_env, navigation_config=navigation_config, agent_config=agent_config, device=device)
    else:
        raise ValueError(f"Unknown agent type: {agent_name}. Supported: 'ppo', 'reinforce'")

    # Close dummy environment
    dummy_env.close()

    return agent


def is_movement_action(action, use_legacy_actions: bool, action_space_mode: str) -> bool:
    """
    Determine if an action is a movement (not rotation or stop).

    Args:
        action: Action (int or tuple)
        use_legacy_actions: Whether using legacy action space
        action_space_mode: Action space mode ('multi_head' or 'single_head_large')

    Returns:
        bool: True if action is a movement
    """
    if use_legacy_actions:
        action_idx = action[0] if isinstance(action, (tuple, list)) else action
        len_idx = int(action_idx) % 2
        return len_idx != 0  # len_idx != 0 means movement

    elif action_space_mode == "single_head_large":
        action_idx = action[0] if isinstance(action, (tuple, list)) else action
        len_idx = int(action_idx) % 21
        return len_idx != 0  # len_idx != 0 means movement

    else:  # multi_head
        if isinstance(action, int):
            # Single-head-large fallback
            len_idx = action % 21
            return len_idx != 0
        elif isinstance(action, (tuple, list)) and len(action) >= 2:
            dir_idx, len_idx = int(action[0]), int(action[1])
            stop_flag = int(action[2]) if len(action) >= 3 else 0
            # Movement if len_idx != 0 and not stopping
            return len_idx != 0 and stop_flag == 0
    return False


def evaluate_checkpoint(
    checkpoint_path: str,
    config: dict,
    eval_scenes: list[int],
    episodes_per_scene: int,
    device: torch.device,
    writer: SummaryWriter = None,
    episode_offset: int = 0,
    run_name: str = None,
) -> dict:
    """
    Evaluate a checkpoint on held-out scenes.

    Args:
        checkpoint_path: Path to checkpoint file
        config: Run configuration
        eval_scenes: List of scene numbers for evaluation
        episodes_per_scene: Number of episodes per scene
        device: torch.device
        writer: TensorBoard writer (optional, shared across seeds)
        episode_offset: Starting episode index for logging (for multi-seed scenarios)
        run_name: Name of the run for tagging (optional)

    Returns:
        dict: Evaluation statistics including 'final_episode_idx'
    """
    # Create agent from config
    agent = create_agent_from_config(config, device)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Handle _orig_mod prefix from torch.compile()
    # Remove _orig_mod from keys if present
    state_dict = {}
    for key, value in checkpoint.items():
        new_key = key.replace("_orig_mod.", "")
        state_dict[new_key] = value

    agent.load_state_dict(state_dict)
    agent.eval()

    # Create environment
    # Support both "env" and "environment" keys for backwards compatibility
    env_config = config.get("env") or config.get("environment")
    use_legacy_actions = env_config.get("use_legacy_actions", False)
    action_space_mode = env_config.get("action_space_mode", "multi_head")

    # IMPORTANT: Use final/max curriculum stage for evaluation!
    # During training, curriculum progresses to full action space.
    # For evaluation, we must use the FINAL stage, not the initial one.
    training_config = config.get("training", {})
    curriculum_config = training_config.get("curriculum", {})

    if curriculum_config.get("enabled", False) and not use_legacy_actions:
        # If curriculum was enabled during training, use the maximum stage
        curriculum_stage = curriculum_config.get("n_stages", 4)  # Final stage = full action space
    else:
        # No curriculum or legacy actions: use config value
        curriculum_stage = env_config.get("curriculum_stage", 1)

    env = PrecomputedThorEnv(
        scene_number=None,  # Will be set dynamically
        rho=env_config["rho"],
        max_actions=env_config["max_actions"],
        use_lmdb=True,
        render=False,
        use_legacy_actions=use_legacy_actions,
        action_space_mode=action_space_mode,
        curriculum_stage=curriculum_stage,
        stop_stagnation_steps=env_config.get("stop_stagnation_steps", 5),
        stop_stagnation_bonus=env_config.get("stop_stagnation_bonus", 0.02),
    )

    print(f"  Using curriculum stage: {curriculum_stage}")

    # Statistics accumulators
    all_scores = []
    all_steps = []
    all_rewards = []
    all_path_lengths = []
    all_coverages = []

    # Movement statistics (cumulative across all episodes)
    total_movements = 0
    total_successful_movements = 0

    # Evaluate each scene
    total_episodes = len(eval_scenes) * episodes_per_scene
    pbar = tqdm(total=total_episodes, desc="Evaluating", ncols=100, leave=False)

    episode_idx = episode_offset
    try:
        for scene_num in eval_scenes:
            for ep_idx in range(episodes_per_scene):
                # Reset environment
                obs = env.reset(scene_number=scene_num, random_start=True)
                agent.reset()

                episode_reward = 0.0
                episode_steps = 0
                movement_count = 0
                successful_movements = 0

                # Run episode
                while not (obs.terminated or obs.truncated):
                    with torch.no_grad():
                        action, *_ = agent.get_action(obs, deterministic=False)

                    # Check if this is a movement action
                    is_movement = is_movement_action(action, use_legacy_actions, action_space_mode)

                    obs = env.step(action)
                    episode_reward += obs.reward
                    episode_steps += 1

                    # Track movement success
                    if is_movement:
                        movement_count += 1
                        if obs.info.get("move_action_success", True):
                            successful_movements += 1

                # Collect episode statistics
                score = obs.info.get("score", 0.0)
                path_length = obs.info.get("total_path_length", 0.0)
                coverage = obs.info.get("exploration_coverage", 0.0)

                all_scores.append(score)
                all_steps.append(episode_steps)
                all_rewards.append(episode_reward)
                all_path_lengths.append(path_length)
                all_coverages.append(coverage)

                # Accumulate movement statistics (will compute rate at the end)
                total_movements += movement_count
                total_successful_movements += successful_movements

                # Log to TensorBoard
                if writer:
                    writer.add_scalar("eval/score", score, episode_idx)
                    writer.add_scalar("eval/steps", episode_steps, episode_idx)
                    writer.add_scalar("eval/reward", episode_reward, episode_idx)
                    writer.add_scalar("eval/path_length", path_length, episode_idx)
                    writer.add_scalar("eval/coverage", coverage, episode_idx)
                    writer.add_scalar("eval/movement_count", movement_count, episode_idx)
                    writer.add_scalar("eval/successful_movements", successful_movements, episode_idx)
                    # Calculate and log per-episode success rate (for visualization)
                    ep_movement_success_rate = successful_movements / movement_count if movement_count > 0 else float("nan")
                    if not np.isnan(ep_movement_success_rate):
                        writer.add_scalar("eval/movement_success_rate", ep_movement_success_rate, episode_idx)
                    writer.add_scalar("eval/scene", scene_num, episode_idx)
                    if run_name:
                        writer.add_text("eval/run_name", run_name, episode_idx)

                episode_idx += 1
                pbar.update(1)

    finally:
        pbar.close()
        try:
            env.close()
        except Exception:
            pass  # Ignore errors during cleanup

        # Note: Don't close writer here - it's shared across seeds and will be closed by the caller

        # Clear the transition table cache to ensure fresh LMDB connections
        # for the next evaluation (avoids reusing closed connections)
        PrecomputedThorEnv._transition_cache.clear()

        # Force garbage collection to clean up LMDB connections
        gc.collect()

    # Compute statistics
    # Calculate overall movement success rate from cumulative counts
    movement_success_rate = total_successful_movements / total_movements if total_movements > 0 else 0.0

    stats = {
        "score_mean": np.mean(all_scores),
        "score_std": np.std(all_scores),
        "steps_mean": np.mean(all_steps),
        "steps_std": np.std(all_steps),
        "reward_mean": np.mean(all_rewards),
        "reward_std": np.std(all_rewards),
        "path_length_mean": np.mean(all_path_lengths),
        "path_length_std": np.std(all_path_lengths),
        "coverage_mean": np.mean(all_coverages),
        "coverage_std": np.std(all_coverages),
        "movement_success_rate": movement_success_rate,  # Cumulative rate over all episodes
        "total_movements": total_movements,
        "total_successful_movements": total_successful_movements,
        "num_episodes": len(all_scores),
        "final_episode_idx": episode_idx,  # For continuing episode numbering across seeds
    }

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Post-Training Evaluation: Re-evaluate trained agents with final checkpoints",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--base_dir", default="RL_training/runs", help="Base directory with training runs")
    parser.add_argument("--scenarios", default=None, help="Comma-separated scenario names to evaluate (default: all)")
    parser.add_argument("--episodes", type=int, default=30, help="Episodes per scene (default: 30)")
    parser.add_argument("--eval_scenes", default="28,29,30", help="Comma-separated eval scene numbers (default: 28,29,30)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda/cpu)")
    parser.add_argument("--output", default=None, help="Output CSV file path (optional)")

    args = parser.parse_args()

    # Parse eval scenes
    eval_scenes = [int(s.strip()) for s in args.eval_scenes.split(",")]

    # Parse scenario filter
    scenario_filter = None
    if args.scenarios:
        scenario_filter = set(s.strip() for s in args.scenarios.split(","))

    device = torch.device(args.device)
    print(f"Using device: {device}")
    print(f"Eval scenes: {eval_scenes}")
    print(f"Episodes per scene: {args.episodes}")
    print()

    # Discover runs
    print(f"Discovering runs in: {args.base_dir}")
    runs = discover_completed_runs(args.base_dir)

    if not runs:
        print("[ERROR] No completed runs found!")
        return

    print(f"Found {len(runs)} scenarios with completed runs\n")

    # Evaluate each run
    results = []

    for scenario_name, run_dirs in sorted(runs.items()):
        # Apply scenario filter
        if scenario_filter and scenario_name not in scenario_filter:
            continue

        print(f"\n{'='*80}")
        print(f"Scenario: {scenario_name} ({len(run_dirs)} runs)")
        print(f"{'='*80}")

        # Create TensorBoard writer for this scenario (shared across all seeds)
        scenario_log_dir = os.path.join(args.base_dir, "post_train_eval", scenario_name)

        # Remove old event files if directory exists
        if os.path.exists(scenario_log_dir):
            for f in os.listdir(scenario_log_dir):
                if f.startswith("events.out.tfevents."):
                    os.remove(os.path.join(scenario_log_dir, f))

        writer = SummaryWriter(log_dir=scenario_log_dir)
        episode_offset = 0  # Track episode index across all seeds

        # Accumulate movement statistics across all seeds for this scenario
        scenario_total_movements = 0
        scenario_total_successful_movements = 0

        try:
            for run_dir in run_dirs:
                run_name = os.path.basename(run_dir)
                print(f"\n{run_name}")

                # Find checkpoint
                checkpoint_path = find_last_checkpoint(run_dir)
                if checkpoint_path is None:
                    print("  [SKIP] No checkpoint found")
                    continue

                checkpoint_name = os.path.basename(checkpoint_path)
                print(f"  Checkpoint: {checkpoint_name}")

                # Load config
                config = load_config_from_run(run_dir)
                if config is None:
                    print("  [SKIP] No config found")
                    continue

                # Evaluate
                try:
                    stats = evaluate_checkpoint(
                        checkpoint_path=checkpoint_path,
                        config=config,
                        eval_scenes=eval_scenes,
                        episodes_per_scene=args.episodes,
                        device=device,
                        writer=writer,
                        episode_offset=episode_offset,
                        run_name=run_name,
                    )

                    # Update episode offset for next run
                    episode_offset = stats["final_episode_idx"]

                    # Accumulate movement statistics for the scenario
                    scenario_total_movements += stats["total_movements"]
                    scenario_total_successful_movements += stats["total_successful_movements"]

                    # Print results
                    print(f"  Score:                 {stats['score_mean']:.4f} ± {stats['score_std']:.4f}")
                    print(f"  Steps:                 {stats['steps_mean']:.1f} ± {stats['steps_std']:.1f}")
                    print(f"  Path Length:           {stats['path_length_mean']:.2f} ± {stats['path_length_std']:.2f}")
                    print(
                        f"  Movement Success Rate: {stats['movement_success_rate']:.4f} ({stats['total_successful_movements']}/{stats['total_movements']})"
                    )
                    print(f"  Coverage:              {stats['coverage_mean']:.4f} ± {stats['coverage_std']:.4f}")

                    # Collect for summary table
                    results.append(
                        {
                            "Scenario": scenario_name,
                            "Run": run_name,
                            "Checkpoint": checkpoint_name,
                            "Score": stats["score_mean"],
                            "Score_Std": stats["score_std"],
                            "Steps": stats["steps_mean"],
                            "Steps_Std": stats["steps_std"],
                            "Path_Length": stats["path_length_mean"],
                            "Path_Length_Std": stats["path_length_std"],
                            "Movement_Success_Rate": stats["movement_success_rate"],
                            "Total_Movements": stats["total_movements"],
                            "Total_Successful_Movements": stats["total_successful_movements"],
                            "Coverage": stats["coverage_mean"],
                            "Coverage_Std": stats["coverage_std"],
                            "Num_Episodes": stats["num_episodes"],
                        }
                    )

                except Exception as e:
                    print(f"  [ERROR] {e}")
                    continue
                finally:
                    # Clear transition table cache and force cleanup after each checkpoint
                    # This ensures LMDB connections are properly closed between evaluations
                    try:
                        PrecomputedThorEnv._transition_cache.clear()
                    except Exception:
                        pass
                    gc.collect()

        finally:
            # Print overall scenario statistics
            if scenario_total_movements > 0:
                scenario_movement_success_rate = scenario_total_successful_movements / scenario_total_movements
                print(
                    f"\n  Overall Movement Success Rate (all seeds): {scenario_movement_success_rate:.4f} ({scenario_total_successful_movements}/{scenario_total_movements})"
                )

            # Close the scenario writer after all seeds are evaluated
            try:
                writer.close()
            except Exception:
                pass

    # Summary table
    if results:
        print("\n" + "=" * 80)
        print("SUMMARY TABLE")
        print("=" * 80 + "\n")

        df = pd.DataFrame(results)

        # Group by scenario and compute mean/std across seeds
        scenario_stats = []
        for scenario_name in df["Scenario"].unique():
            scenario_df = df[df["Scenario"] == scenario_name]

            # Calculate overall movement success rate from cumulative totals
            total_movements = scenario_df["Total_Movements"].sum()
            total_successful = scenario_df["Total_Successful_Movements"].sum()
            overall_movement_success_rate = total_successful / total_movements if total_movements > 0 else 0.0

            scenario_stats.append(
                {
                    "Scenario": scenario_name,
                    "Num_Seeds": len(scenario_df),
                    "Score": f"{scenario_df['Score'].mean():.4f} ± {scenario_df['Score'].std():.4f}",
                    "Steps": f"{scenario_df['Steps'].mean():.1f} ± {scenario_df['Steps'].std():.1f}",
                    "Path_Length": f"{scenario_df['Path_Length'].mean():.2f} ± {scenario_df['Path_Length'].std():.2f}",
                    "Movement_Success_Rate": f"{overall_movement_success_rate:.4f} ({total_successful}/{total_movements})",
                    "Coverage": f"{scenario_df['Coverage'].mean():.4f} ± {scenario_df['Coverage'].std():.4f}",
                }
            )

        summary_df = pd.DataFrame(scenario_stats)
        print(summary_df.to_string(index=False))

        # Save to CSV if requested
        if args.output:
            df.to_csv(args.output, index=False)
            print(f"\n[INFO] Detailed results saved to: {args.output}")

            summary_output = args.output.replace(".csv", "_summary.csv")
            summary_df.to_csv(summary_output, index=False)
            print(f"[INFO] Summary saved to: {summary_output}")

    else:
        print("\n[WARN] No results to display")


if __name__ == "__main__":
    main()
