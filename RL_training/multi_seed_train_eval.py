"""
Multi-Seed Training with Held-Out Evaluation

Orchestrates sequential training across multiple seeds with periodic evaluation
on held-out scenes (FloorPlan 28, 29, 30).

Features:
- Automatic seed generation (reproducible via master seed)
- Periodic evaluation on unseen kitchens
- Per-seed TensorBoard logging
- Efficient environment reuse

Usage:
    python RL_training/multi_seed_train_eval.py \
        --config configs/scenario4_ppo_multihead_curriculum.json \
        --num_seeds 5 \
        --start_seed 0 \
        --n_envs 8 \
        --save_model

Seeds are automatically generated using a fixed master seed (42) for reproducibility.
Same num_seeds always produces same seeds.

For distributed training across multiple nodes:
    Node 1: --num_seeds 3 --start_seed 0  # Trains seeds 0-2
    Node 2: --num_seeds 2 --start_seed 3  # Trains seeds 3-4
"""

import multiprocessing as mp
import os
import sys
import time
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

# Enable TensorFloat32 for faster matmuls on Ampere+ GPUs
torch.set_float32_matmul_precision("high")


def set_working_directory():
    """Set working directory to project root."""
    desired_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    current_directory = os.getcwd()

    if current_directory != desired_directory:
        os.chdir(desired_directory)
        sys.path.append(desired_directory)
        print(f"Working directory changed from '{current_directory}' to '{desired_directory}'")
        return

    print("Working directory:", os.getcwd())


def generate_seeds(total_seeds, start_seed=0, master_seed=42):
    """
    Generate seeds deterministically using a master seed.

    Args:
        total_seeds: Number of seeds to train in this run
        start_seed: Starting index in the global seed pool (for distributed training)
        master_seed: Fixed seed for reproducibility (default: 42)

    Returns:
        List of random seed values starting from start_seed index

    Note:
        The function generates a consistent global pool and returns a slice.
        Different nodes with different start_seed get non-overlapping seeds
        from the same deterministic sequence.

    Example:
        Master seed 42 produces sequence: [7271, 861, 5391, 5192, 5735, ...]

        Node 1: generate_seeds(3, start_seed=0) -> [7271, 861, 5391]
        Node 2: generate_seeds(2, start_seed=3) -> [5192, 5735]

        All nodes get different seeds from the same reproducible sequence.
    """
    np.random.seed(master_seed)
    # Generate enough seeds to cover start_seed + total_seeds
    all_seeds = np.random.randint(1, 10000, size=total_seeds + start_seed).tolist()
    # Return only the requested slice
    return all_seeds[start_seed:start_seed + total_seeds]


def main(config, args):
    """
    Main multi-seed training function with held-out evaluation.

    Args:
        config: Configuration dict
        args: Command-line arguments
    """
    from components.utils.utility_functions import set_seeds
    from RL_training.runner.held_out_eval_runner import HeldOutEvalRunner
    from RL_training.runner.training_manager import TrainingManager
    from RL_training.runner.unified_train_runner import UnifiedTrainRunner

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # Validate arguments
    if args.start_seed < 0:
        raise ValueError(f"start_seed must be >= 0, got {args.start_seed}")
    if args.num_seeds < 1:
        raise ValueError(f"num_seeds must be >= 1, got {args.num_seeds}")

    # Generate seeds deterministically
    seeds = generate_seeds(args.num_seeds, start_seed=args.start_seed)
    print(f"\n{'='*70}")
    print(f"[INFO] Multi-Seed Training Configuration")
    print(f"{'='*70}")
    print(f"  Config: {args.config}")
    print(f"  Number of seeds: {args.num_seeds}")
    print(f"  Start seed index: {args.start_seed}")
    print(f"  Seed range: {args.start_seed} to {args.start_seed + args.num_seeds - 1}")
    print(f"  Generated seeds: {seeds}")
    print(f"  Parallel envs: {args.n_envs or 'auto'}")
    print(f"  Save checkpoints: {args.save_model}")

    # Extract evaluation configuration
    eval_config = config.get("evaluation", {})
    eval_enabled = eval_config.get("enabled", False)  # Default to False if not configured
    eval_scenes = eval_config.get("eval_scene_numbers", [28, 29, 30])
    episodes_per_scene = eval_config.get("episodes_per_scene", 10)

    if eval_enabled:
        print(f"\n  Evaluation enabled:")
        print(f"    Eval scenes: {eval_scenes}")
        print(f"    Episodes per scene: {episodes_per_scene}")
        print(f"    Total eval episodes: {len(eval_scenes) * episodes_per_scene}")
    else:
        print(f"\n  Evaluation disabled")

    print(f"{'='*70}\n")

    # Track timing
    total_start_time = time.time()

    # Run training for each seed sequentially
    for local_idx, seed in enumerate(seeds):
        global_idx = args.start_seed + local_idx
        print(f"\n{'='*70}")
        print(f"[INFO] Starting Seed {local_idx+1}/{len(seeds)} (Global Index: {global_idx}): {seed}")
        print(f"{'='*70}\n")

        seed_start_time = time.time()

        # Update config with current seed
        config["seed"] = seed
        set_seeds(seed, strict_determinism=config.get("strict_determinism", False))

        # Create timestamp for this seed's run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Extract short scenario name from config path (e.g., "scenario2_ppo_legacy")
        # This avoids overly long directory names
        config_stem = Path(args.config).stem  # e.g., "scenario2_ppo_legacy"
        scenario_name = config_stem

        # Use custom log_dir if provided (for SLURM scratch), otherwise default
        base_log_dir = args.log_dir if args.log_dir else "RL_training/runs"
        log_dir = f"{base_log_dir}/{scenario_name}_seed_{seed}_{timestamp}"

        print(f"[INFO] Log directory: {log_dir}")

        # Extract config sections
        training_config = config.get("training", {})

        # Parallel environment setup
        n_envs = args.n_envs if args.n_envs else training_config.get("num_agents", mp.cpu_count())
        print(f"[INFO] Creating {n_envs} parallel environments")

        # Create training manager
        manager = TrainingManager(config, n_envs, device)

        # Create vectorized environment
        vec_env = manager._create_vec_env()

        # Create agent
        agent = manager._create_agent()

        # Setup evaluation if enabled
        eval_runner = None
        eval_writer = None

        try:
            if eval_enabled:
                print(f"[INFO] Setting up held-out evaluation on FloorPlan {eval_scenes}")

                # Create evaluation runner
                eval_runner = HeldOutEvalRunner(
                    scene_numbers=eval_scenes,
                    episodes_per_scene=episodes_per_scene,
                    env_config=config["env"],
                    device=device,
                )

                # Create TensorBoard writer for evaluation
                eval_writer = SummaryWriter(f"{log_dir}/eval")

                # Define evaluation callback
                def eval_callback(block_num, agent):
                    print(f"\n[EVAL] Running held-out evaluation at block {block_num}...")
                    eval_start = time.time()

                    # Run evaluation
                    eval_stats = eval_runner.evaluate(agent, block_num, eval_writer)

                    eval_time = time.time() - eval_start
                    print(f"[EVAL] Completed in {eval_time:.1f}s")
                    print(f"[EVAL] Mean Score: {eval_stats['mean_score']:.3f} ± {eval_stats['std_score']:.3f}")
                    print(f"[EVAL] Mean Steps: {eval_stats['mean_steps']:.1f} ± {eval_stats['std_steps']:.1f}")
                    print(f"[EVAL] Mean Reward: {eval_stats['mean_reward']:.2f} ± {eval_stats['std_reward']:.2f}\n")

            # Create unified runner (it creates its own TensorBoard writer)
            runner = UnifiedTrainRunner(
                vec_env=vec_env, agent=agent, device=device, config=config, log_dir=log_dir
            )

            # Attach evaluation callback if enabled
            if eval_enabled:
                runner.eval_callback = eval_callback

                # IMPORTANT: Run evaluation at block 0 (initial weights)
                print(f"[EVAL] Running initial evaluation at block 0...")
                eval_callback(0, agent)

            # Run training
            print(f"\n[INFO] Starting training for seed {seed}...\n")
            runner.run(save_model=args.save_model)

        finally:
            # Cleanup resources (always executed, even if error occurs)
            if eval_runner is not None:
                try:
                    eval_runner.close()
                except Exception as e:
                    print(f"[WARNING] Error closing eval_runner: {e}")

            if eval_writer is not None:
                try:
                    eval_writer.close()
                except Exception as e:
                    print(f"[WARNING] Error closing eval_writer: {e}")

            # Note: UnifiedTrainRunner closes its own writer internally
            try:
                vec_env.close()
            except Exception as e:
                print(f"[WARNING] Error closing vec_env: {e}")

            # CRITICAL: Clear LMDB cache to avoid reusing closed database handles
            # between seeds. Without this, the second seed will crash with
            # "lmdb.Error: Attempt to operate on closed object"
            try:
                from components.environments.precomputed_thor_env import PrecomputedThorEnv
                PrecomputedThorEnv.clear_cache()
                print(f"[INFO] Cleared LMDB cache for next seed")
            except Exception as e:
                print(f"[WARNING] Error clearing LMDB cache: {e}")

        seed_time = time.time() - seed_start_time
        print(f"\n[INFO] Seed {seed} completed in {seed_time/3600:.2f} hours\n")

    # Final summary
    total_time = time.time() - total_start_time
    print(f"\n{'='*70}")
    print(f"[INFO] All seeds completed successfully!")
    print(f"{'='*70}")
    print(f"  Total seeds: {len(seeds)}")
    print(f"  Total time: {total_time/3600:.2f} hours")
    print(f"  Avg time per seed: {(total_time/len(seeds))/3600:.2f} hours")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    set_working_directory()

    from components.utils.utility_functions import read_config

    parser = ArgumentParser(description="Multi-Seed RL Training with Held-Out Evaluation")
    parser.add_argument("--config", type=str, required=True, help="Path to configuration file")
    parser.add_argument(
        "--num_seeds",
        type=int,
        default=5,
        help="Number of seeds to run (default: 5). Seeds are auto-generated reproducibly.",
    )
    parser.add_argument(
        "--start_seed",
        type=int,
        default=0,
        help="Starting seed index for distributed training (default: 0). "
        "Use with --num_seeds to train specific seed ranges across multiple nodes.",
    )
    parser.add_argument(
        "--n_envs",
        type=int,
        default=None,
        help="Number of parallel environments (default: config or CPU count)",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default=None,
        help="Base directory for logs (default: RL_training/runs). "
        "Used for writing to fast local scratch storage on SLURM nodes.",
    )
    parser.add_argument("--save_model", action="store_true", help="Save model checkpoints")
    args = parser.parse_args()

    config_path = Path(args.config)
    print(f"[INFO] Loading configuration: {config_path}")
    config = read_config(config_path)

    # Handle memory fraction if set
    frac = os.environ.get("TORCH_MEM_FRACTION")
    if frac:
        torch.cuda.set_per_process_memory_fraction(float(frac))

    main(config, args)
