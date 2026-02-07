"""
Unified Training Script for all RL Agents (REINFORCE, A2C, PPO)

Single entry point for all training configurations.
Supports:
- All agents (REINFORCE, A2C, PPO)
- Parallel VecEnv training for all agents
- IL pre-training
- Curriculum learning
- Centralized logging

Usage:
    python RL_training/train.py --config path/to/config.json --save_model

Config format:
    {
        "name": "Experiment Name",
        "seed": 42,
        "agent": {...},
        "navigation": {...},
        "env": {...},
        "training": {...}
    }
"""

import multiprocessing as mp
import os
import sys
from argparse import ArgumentParser
from pathlib import Path

import torch

# Enable TensorFloat32 for faster matmuls on Ampere+ GPUs (~20% speedup)
# Slightly reduced precision, but safe for deep learning
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


def main(config, args):
    """Main training function."""
    from RL_training.runner.training_manager import TrainingManager
    from RL_training.runner.unified_train_runner import UnifiedTrainRunner

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # Extract config sections for display
    agent_config = config["agent"]
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

    # Create unified runner
    runner = UnifiedTrainRunner(vec_env=vec_env, agent=agent, device=device, config=config, log_dir=args.log_dir)

    # Run training
    runner.run(save_model=args.save_model)


if __name__ == "__main__":
    set_working_directory()

    from components.utils.utility_functions import read_config, set_seeds

    parser = ArgumentParser(description="Unified RL Training")
    parser.add_argument("--config", type=str, required=True, help="Path to configuration file")
    parser.add_argument("--n_envs", type=int, default=None, help="Number of parallel environments (default: config or CPU count)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (overrides config)")
    parser.add_argument("--save_model", action="store_true", help="Save model checkpoints")
    parser.add_argument("--log_dir", type=str, default=None, help="Directory for logging outputs")
    args = parser.parse_args()

    config_path = Path(args.config)
    print(f"[INFO] Loading configuration: {config_path}")
    config = read_config(config_path)

    # Override seed if provided via CLI
    if args.seed is not None:
        config["seed"] = args.seed
        print(f"[INFO] Using seed from CLI: {args.seed}")

    set_seeds(config["seed"], strict_determinism=config.get("strict_determinism", False))

    frac = os.environ.get("TORCH_MEM_FRACTION")
    if frac:
        torch.cuda.set_per_process_memory_fraction(float(frac))

    main(config, args)
