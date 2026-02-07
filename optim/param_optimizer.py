"""
Parameter Optimization for REINFORCE and PPO Agents

Optimizes hyperparameters for four scenarios:
1. REINFORCE + IL pre-training (Scenario 0)
2. PPO + Legacy Actions (Scenario 2)
3. PPO + Multi-head Actions - No Curriculum (Scenario 3)
4. PPO + Single-Head Actions - No Curriculum (Scenario 6)

Uses Optuna with MedianPruner or PercentilePruner for early stopping.
Scenario 6 (singlehead_large) uses less aggressive pruning due to slower learning.
"""

import gc
import json
import os
import sys
from argparse import ArgumentParser
from functools import partial
from pathlib import Path

import optuna
import torch
from optuna.pruners import MedianPruner, PercentilePruner
from optuna.storages import RDBStorage

# Enable TensorFloat32 for faster matmuls on Ampere+ GPUs (~20% speedup)
# Slightly reduced precision, but safe for deep learning
torch.set_float32_matmul_precision("high")


def get_param_search_space(agent_name):
    """
    Define parameter search spaces for different agent configurations.

    Uses categorical (discrete) values for faster convergence and interpretability.

    Args:
        agent_name: "reinforce" or "ppo"

    Returns:
        dict: Parameter choices for Optuna (categorical)
    """
    if agent_name == "reinforce":
        search_space = {"gamma": [0.90, 0.95, 0.97, 0.99]}  # Discount factor
    elif agent_name == "ppo":
        search_space = {
            "clip_epsilon": [0.1, 0.15, 0.2, 0.25, 0.3],  # PPO clip parameter
            "gae_lambda": [0.90, 0.93, 0.95, 0.97, 0.98],  # GAE lambda
        }
    else:
        raise ValueError(f"Unknown agent: {agent_name}")

    search_space["entropy_coef"] = [0.01, 0.05, 0.10, 0.15]
    search_space["alpha"] = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3]
    search_space["collision_loss_coef"] = [0.1, 0.2, 0.3, 0.4]  # Collision loss weight

    return search_space


def objective(trial, scenario_name, scenario_config_path, checkpoint_path=None):
    """
    Optuna objective function for hyperparameter optimization.

    Args:
        trial: Optuna trial object
        scenario_name: Name of the scenario being optimized
        scenario_config_path: Path to scenario config JSON
        checkpoint_path: Optional path to pretrained weights (e.g., IL checkpoint)

    Returns:
        float: Objective value (mean score with penalties for collisions, truncations, and long episodes)
    """
    set_seeds(42)

    # Load base configuration
    config = read_config(scenario_config_path, use_print=False)
    agent_config = config["agent"]
    navigation_config = config["navigation"]
    env_config = config["env"]
    training_config = config["training"]

    agent_name = agent_config["name"]
    use_legacy_actions = env_config.get("use_legacy_actions", False)

    # Get parameter search space (categorical values)
    search_space = get_param_search_space(agent_name)

    # Sample hyperparameters from trial using categorical choices
    params = {}
    for param_name, choices in search_space.items():
        params[param_name] = trial.suggest_categorical(param_name, choices)

    # Update agent config with sampled parameters
    agent_config.update(params)

    # Determine training blocks based on scenario
    # Different scenarios require different training lengths due to varying learning speeds
    if agent_name == "reinforce" or (agent_name == "ppo" and use_legacy_actions):
        # REINFORCE + IL or PPO Legacy: 400 blocks (simpler action spaces learn faster)
        total_blocks = 400
    elif agent_name == "ppo":
        if scenario_name == "ppo_singlehead_large":
            # Singlehead large (504 actions): 800 blocks
            # Learns slower than multihead due to single output head, needs more training
            total_blocks = 800
        else:
            # PPO Multihead (504 actions): 500 blocks
            total_blocks = 500
    else:
        total_blocks = agent_config.get("blocks", 200)

    # Override blocks in config for this optimization trial
    agent_config["blocks"] = total_blocks

    # Rebuild full config
    config["agent"] = agent_config
    config["navigation"] = navigation_config
    config["env"] = env_config
    config["training"] = training_config

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create training manager
    from RL_training.runner.training_manager import TrainingManager

    n_envs = training_config.get("num_agents", 32)
    manager = TrainingManager(config, n_envs, device)

    # Create vectorized environment
    vec_env = manager._create_vec_env()

    # Create agent (includes IL loading if specified in config)
    agent = manager._create_agent()

    # Create UnifiedOptimRunner
    from optim.runner.unified_optim_runner import UnifiedOptimRunner

    runner = UnifiedOptimRunner(vec_env=vec_env, agent=agent, device=device, config=config, trial=trial, scenario_name=scenario_name)

    # Run optimization and return objective
    try:
        objective_value = runner.run(save_model=False)
        return objective_value
    except optuna.TrialPruned:
        # Trial was pruned - propagate exception
        raise
    except Exception as e:
        print(f"[ERROR] Trial failed: {e}")
        import traceback

        traceback.print_exc()
        # Return very bad objective so this trial is discarded
        return -1000.0
    finally:
        # Cleanup - order matters for GPU memory release!
        vec_env.close()

        # Clear agent caches explicitly before deletion
        if hasattr(agent, "_cached_state_features"):
            del agent._cached_state_features
        if hasattr(agent, "rollout_buffers"):
            agent.rollout_buffers.clear()

        # Clear runner episode history to free memory
        if hasattr(runner, "episode_history"):
            runner.episode_history.clear()
        if hasattr(runner, "_obs_list"):
            del runner._obs_list
        if hasattr(runner, "_hiddens_list"):
            del runner._hiddens_list

        del runner, agent, vec_env, manager

        # IMPORTANT: gc.collect() BEFORE empty_cache() to release Python references first
        gc.collect()

        try:
            if hasattr(torch, "_dynamo"):
                torch._dynamo.reset()
            inductor = getattr(torch, "_inductor", None)
            if inductor is not None:
                codecache = getattr(inductor, "codecache", None)
                if codecache is not None and hasattr(codecache, "clear_cache"):
                    codecache.clear_cache()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()  # Ensure all CUDA ops complete
        except Exception as e:
            print(f"[WARNING] Cache reset failed: {e}")

        # Second gc pass to catch anything released by empty_cache
        gc.collect()


def set_working_directory():
    """Set working directory to project root."""
    desired_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    current_directory = os.getcwd()

    if current_directory != desired_directory:
        os.chdir(desired_directory)
        sys.path.append(desired_directory)
        print(f"Changed working directory from '{current_directory}' to '{desired_directory}'")
        return

    print("Current working directory:", os.getcwd())


def save_progress_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial):
    """Persist progress after each completed trial."""
    # Skip pruned/failed trials
    if trial.state not in [optuna.trial.TrialState.COMPLETE]:
        return

    # Make sure output dirs exist
    out_dir = Path("optim")
    out_dir.mkdir(exist_ok=True, parents=True)

    # Get study name from storage
    study_name = study.study_name

    # Save current best params with metrics
    try:
        best = study.best_trial
        best_path = out_dir / f"best_params_{study_name}.json"

        best_data = {
            "params": best.params,
            "value": best.value,
            "mean_score": best.user_attrs.get("mean_score", None),
            "mean_steps": best.user_attrs.get("mean_steps", None),
            "mean_collision_rate": best.user_attrs.get("mean_collision_rate", None),
            "truncation_rate": best.user_attrs.get("truncation_rate", None),
            "success_rate": best.user_attrs.get("success_rate", None),
            "total_episodes": best.user_attrs.get("total_episodes", None),
            "trial_number": best.number,
        }

        with open(best_path, "w") as f:
            json.dump(best_data, f, indent=2)

        print(f"[INFO] Best params saved to: {best_path}")
    except Exception as e:
        print(f"[WARNING] Could not save best params: {e}")

    # Dump full trials table for auditing/analysis
    try:
        df = study.trials_dataframe(attrs=("number", "state", "value", "params", "user_attrs"))
        df.to_csv(out_dir / f"trials_{study_name}.csv", index=False)
    except Exception as e:
        print(f"[WARNING] Could not save trials dataframe: {e}")


if __name__ == "__main__":
    set_working_directory()

    from components.utils.utility_functions import read_config, set_seeds

    # Validate GT graphs before starting optimization
    print("\n" + "=" * 80)
    print("[VALIDATION] Checking GT graph files...")
    print("=" * 80)
    from components.scripts.validate_gt_graphs import validate_gt_graphs

    if not validate_gt_graphs(delete_corrupted=True, regenerate=True):
        print("[ERROR] GT graph validation/regeneration failed. Exiting.")
        sys.exit(1)
    print("=" * 80 + "\n")

    parser = ArgumentParser(description="Optimize hyperparameters for REINFORCE and PPO agents")
    parser.add_argument(
        "--scenario",
        type=str,
        required=True,
        choices=["reinforce_il", "ppo_legacy", "ppo_multihead", "ppo_singlehead_large"],
        help=(
            "Which scenario to optimize:\n"
            "  reinforce_il (Scenario 0): REINFORCE + IL pre-training\n"
            "  ppo_legacy (Scenario 2): PPO + Legacy Actions (16 discrete)\n"
            "  ppo_multihead (Scenario 3): PPO + Multi-head Actions (504) - NO curriculum\n"
            "  ppo_singlehead_large (Scenario 6): PPO + Single-Head Actions (504) - NO curriculum (FOCUSED tuning)\n"
        ),
    )
    parser.add_argument("--n_trials", type=int, default=50, help="Number of trials to run (default: 50)")
    parser.add_argument("--n_jobs", type=int, default=1, help="Number of parallel jobs (default: 1). WARNING: >1 requires significant RAM.")
    parser.add_argument("--no_shared_storage", action="store_true", help="If set, do not use shared RDB storage (for single-process runs)")
    parser.add_argument(
        "--export_files",
        action="store_true",
        help="If set, export JSON/CSV files after each trial (not recommended for distributed optimization)",
    )
    parser.add_argument(
        "--cleanup_running", action="store_true", help="If set, automatically mark old RUNNING trials as FAIL (useful after aborted jobs)"
    )
    args = parser.parse_args()

    # Map scenario to config file and pruning settings
    # warmup_steps: Number of blocks before pruning is enabled (allows learning to stabilize)
    scenario_mapping = {
        "reinforce_il": {
            "config_path": "configs/scenario0_reinforce_il.json",
            "checkpoint_path": "components/data/model_weights/il/il_pretrained_model",
            "study_name": "reinforce_il",
            "warmup_steps": 200,  # Total: 400 blocks, prune in last 200
        },
        "ppo_legacy": {
            "config_path": "configs/scenario2_ppo_legacy.json",
            "checkpoint_path": None,
            "study_name": "ppo_legacy",
            "warmup_steps": 200,  # Total: 400 blocks, prune in last 200
        },
        "ppo_multihead": {
            "config_path": "configs/scenario3_ppo_multihead_no_curriculum.json",
            "checkpoint_path": None,
            "study_name": "ppo_multihead",
            "warmup_steps": 300,  # Total: 500 blocks, prune in last 200
        },
        "ppo_singlehead_large": {
            "config_path": "configs/scenario6_ppo_singlehead_no_curriculum.json",
            "checkpoint_path": None,
            "study_name": "ppo_singlehead_large",
            "warmup_steps": 600,  # Total: 800 blocks, prune in last 200 (learns slower, needs patience)
        },
    }

    scenario_info = scenario_mapping[args.scenario]
    config_path = scenario_info["config_path"]
    checkpoint_path = scenario_info["checkpoint_path"]
    study_name = scenario_info["study_name"]
    warmup_steps = scenario_info["warmup_steps"]

    print(f"\n{'=' * 80}")
    print(f"[INFO] Starting Hyperparameter Optimization: {args.scenario}")
    print(f"[INFO] Config: {config_path}")
    print(f"[INFO] Checkpoint: {checkpoint_path if checkpoint_path else 'None (train from scratch)'}")
    print(f"[INFO] Number of trials: {args.n_trials}")
    print(f"[INFO] Parallel jobs: {args.n_jobs}")
    print(f"[INFO] Pruning warmup: {warmup_steps} blocks")
    print(f"[INFO] Storage: {'SQLite (local)' if args.no_shared_storage else 'MySQL (shared)'}")
    print(f"[INFO] File export: {'ENABLED' if args.export_files else 'DISABLED'}")
    print(f"{'=' * 80}\n")

    # Setup Optuna storage and study
    if args.no_shared_storage:
        storage = RDBStorage(url=f"sqlite:///optim/optuna_params_{study_name}.db", heartbeat_interval=60, grace_period=120)
    else:
        storage = RDBStorage(
            url="mysql+pymysql://optuna_user:my_optuna_password@137.250.121.164:3306/optuna_db", heartbeat_interval=60, grace_period=120
        )

    sampler = optuna.samplers.TPESampler(seed=42, n_startup_trials=5)

    # Use less aggressive pruner for slow-learning scenarios
    if args.scenario == "ppo_singlehead_large":
        # PercentilePruner: only prune bottom 25% (less aggressive than median)
        pruner = PercentilePruner(
            percentile=25.0,  # Prune trials in bottom 25%
            n_startup_trials=3,  # Start pruning after 3 trials
            n_warmup_steps=warmup_steps,  # Don't prune before warmup
            interval_steps=1,  # Check for pruning every block
        )
        print(f"[INFO] Using PercentilePruner (bottom 25%) due to slow learning")
    else:
        # MedianPruner: prune trials below median (default)
        pruner = MedianPruner(
            n_startup_trials=2,  # Start pruning after 2 trials
            n_warmup_steps=warmup_steps,  # Don't prune before this many blocks
            interval_steps=1,  # Check for pruning every block
        )
        print(f"[INFO] Using MedianPruner (prune below median)")

    study = optuna.create_study(
        direction="maximize", sampler=sampler, pruner=pruner, study_name=study_name, storage=storage, load_if_exists=True
    )

    # Check for zombie RUNNING trials (from aborted jobs)
    running_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.RUNNING]
    if running_trials:
        print(f"\n[WARNING] Found {len(running_trials)} RUNNING trials (possibly from aborted jobs):")
        for t in running_trials[:5]:  # Show first 5
            print(f"  - Trial {t.number}: started {t.datetime_start}")
        if len(running_trials) > 5:
            print(f"  ... and {len(running_trials) - 5} more")

        if args.cleanup_running:
            print(f"[INFO] Cleaning up {len(running_trials)} RUNNING trials (setting to FAIL)...")
            for trial in running_trials:
                try:
                    study._storage.set_trial_state_values(trial._trial_id, state=optuna.trial.TrialState.FAIL)
                except Exception as e:
                    print(f"[WARNING] Could not cleanup trial {trial.number}: {e}")
            print(f"[INFO] Cleanup complete.")
        else:
            print("[INFO] Use --cleanup_running flag to automatically mark them as FAIL")
        print()

    # Calculate how many trials we still need
    # Count existing successful trials (COMPLETE + PRUNED)
    existing_successful = len([t for t in study.trials if t.state in [optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED]])
    existing_failed = len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])

    target_trials = args.n_trials
    remaining_trials = max(0, target_trials - existing_successful)

    print(f"\n[INFO] Trial Summary:")
    print(f"  - Existing successful trials: {existing_successful}")
    print(f"  - Existing failed trials: {existing_failed}")
    print(f"  - Target total: {target_trials}")
    print(f"  - Remaining to run: {remaining_trials}")
    print()

    if remaining_trials == 0:
        print(f"[INFO] Target of {target_trials} successful trials already reached. Nothing to do.")
        print(f"[INFO] Use a higher --n_trials value to run more trials.")
    else:
        # Run optimization with optional callback for file export
        callbacks = [save_progress_callback] if args.export_files else []

        # Count only successful trials (COMPLETE + PRUNED), ignore FAIL
        objective_fn = partial(objective, scenario_name=args.scenario, scenario_config_path=config_path, checkpoint_path=checkpoint_path)

        if args.n_jobs > 1:
            # Parallel optimization: use standard n_trials (counts all states)
            print("[WARNING] Parallel jobs (n_jobs > 1): Failed trials will count towards n_trials limit")
            study.optimize(objective_fn, n_trials=remaining_trials, n_jobs=args.n_jobs, callbacks=callbacks)
        else:
            # Sequential optimization: retry failed trials, check target after each trial
            new_successful = 0
            max_consecutive_failures = 5  # Stop if 5 consecutive failures
            consecutive_failures = 0

            while True:
                # Reload study to get latest trials from DB (important for distributed optimization!)
                study = optuna.load_study(study_name=study_name, storage=storage)

                # Check current successful trial count (may have been updated by other workers)
                current_successful = len(
                    [t for t in study.trials if t.state in [optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED]]
                )

                # Check if target already reached (possibly by other worker)
                if current_successful >= target_trials:
                    print(f"\n[INFO] Target of {target_trials} successful trials reached (current: {current_successful}).")
                    if current_successful > existing_successful:
                        new_by_this_worker = new_successful
                        new_by_others = current_successful - existing_successful - new_by_this_worker
                        print(f"[INFO] This worker contributed {new_by_this_worker} new trials.")
                        if new_by_others > 0:
                            print(f"[INFO] Other workers contributed {new_by_others} new trials.")
                    break

                try:
                    study.optimize(objective_fn, n_trials=1, callbacks=callbacks)

                    # Reload again to get the trial we just created
                    study = optuna.load_study(study_name=study_name, storage=storage)
                    last_trial = study.trials[-1]

                    if last_trial.state in [optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED]:
                        new_successful += 1
                        consecutive_failures = 0

                        # Get updated count from DB
                        current_successful = len(
                            [t for t in study.trials if t.state in [optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED]]
                        )

                        print(f"[INFO] Progress: {current_successful}/{target_trials} successful trials (this worker: +{new_successful})")
                    else:
                        # FAIL state
                        consecutive_failures += 1
                        print(f"[WARNING] Trial {last_trial.number} FAILED (not counted). {consecutive_failures} consecutive failures.")

                        if consecutive_failures >= max_consecutive_failures:
                            print(f"[ERROR] {max_consecutive_failures} consecutive failures. Stopping optimization.")
                            break

                except KeyboardInterrupt:
                    print("\n[INFO] Optimization interrupted by user.")
                    break
                except Exception as e:
                    print(f"[ERROR] Unexpected error during optimization: {e}")
                    consecutive_failures += 1
                    if consecutive_failures >= max_consecutive_failures:
                        print(f"[ERROR] {max_consecutive_failures} consecutive failures. Stopping optimization.")
                        break

    # Print final results
    best_trial = study.best_trial
    print("\n" + "=" * 80)
    print("[RESULTS] Best Trial:")
    print(f"  Trial Number: {best_trial.number}")
    print(f"  Objective Value: {best_trial.value:.3f}")

    baseline_params = {}

    for param, value in best_trial.params.items():
        baseline_params[param] = value

    print("\n[BASELINE PARAMETERS]")
    for param, value in baseline_params.items():
        if isinstance(value, float):
            print(f"  {param}: {value:.6f}")
        else:
            print(f"  {param}: {value}")

    print("\n[METRICS]")
    print(f"  Mean Score: {best_trial.user_attrs.get('mean_score', 'N/A')}")
    print(f"  Mean Steps: {best_trial.user_attrs.get('mean_steps', 'N/A')}")
    print(f"  Mean Collision Rate: {best_trial.user_attrs.get('mean_collision_rate', 'N/A')}")
    print(f"  Truncation Rate: {best_trial.user_attrs.get('truncation_rate', 'N/A')}")
    print(f"  Success Rate: {best_trial.user_attrs.get('success_rate', 'N/A')}")
    print(f"  Total Episodes: {best_trial.user_attrs.get('total_episodes', 'N/A')}")
    print("=" * 80 + "\n")

    # Save final results (optional)
    if args.export_files:
        final_results = {
            "params": best_trial.params,
            "value": best_trial.value,
            "mean_score": best_trial.user_attrs.get("mean_score", None),
            "mean_steps": best_trial.user_attrs.get("mean_steps", None),
            "mean_collision_rate": best_trial.user_attrs.get("mean_collision_rate", None),
            "truncation_rate": best_trial.user_attrs.get("truncation_rate", None),
            "success_rate": best_trial.user_attrs.get("success_rate", None),
            "total_episodes": best_trial.user_attrs.get("total_episodes", None),
            "trial_number": best_trial.number,
        }

        final_path = f"optim/best_params_{study_name}_final.json"
        with open(final_path, "w") as f:
            json.dump(final_results, f, indent=2)

        print(f"[INFO] Final results saved to: {final_path}")
