"""
Held-Out Evaluation Runner for RL Training

Lightweight evaluation runner for periodic evaluation on held-out scenes during training.
Unlike RLEvalRunner, this:
- Does not create its own TensorBoard writer (uses parent's)
- Reuses a single environment across scenes (efficient)
- Uses stochastic action selection (avoids deadlocks)
- Returns statistics dict (doesn't handle logging directly)
"""

import numpy as np
import torch
from tqdm import tqdm


class HeldOutEvalRunner:
    def __init__(self, scene_numbers, episodes_per_scene, env_config, device):
        """
        Initialize the held-out evaluation runner.

        Args:
            scene_numbers: List of scene numbers for evaluation (e.g., [28, 29, 30])
            episodes_per_scene: Number of episodes to run per scene (e.g., 10)
            env_config: Environment configuration dict
            device: torch.device for agent
        """
        self.scene_numbers = scene_numbers
        self.episodes_per_scene = episodes_per_scene
        self.device = device

        # Create a single environment that will be reused across all scenes
        # Use PrecomputedTHOREnv (not RandomSceneEnv) to support scene_number in reset()
        from components.environments.precomputed_thor_env import PrecomputedThorEnv

        self.env = PrecomputedThorEnv(
            scene_number=None,  # Will be set dynamically in reset()
            rho=env_config["rho"],
            max_actions=env_config["max_actions"],
            use_lmdb=True,
            render=False,
            use_legacy_actions=env_config.get("use_legacy_actions", False),
            action_space_mode=env_config.get("action_space_mode", "multi_head"),
            curriculum_stage=env_config.get("curriculum_stage", 1),
            stop_stagnation_steps=env_config.get("stop_stagnation_steps", 5),
            stop_stagnation_bonus=env_config.get("stop_stagnation_bonus", 0.02),
        )

    def evaluate(self, agent, block_num, writer):
        """
        Run evaluation on held-out scenes.

        Args:
            agent: The RL agent to evaluate
            block_num: Current training block number (for logging)
            writer: TensorBoard SummaryWriter for logging

        Returns:
            dict: Evaluation statistics (mean_score, mean_steps, mean_reward, etc.)
        """
        # Set agent to eval mode (affects batch norm, dropout)
        agent.eval()

        # Statistics accumulators
        all_scores = []
        all_steps = []
        all_rewards = []
        all_path_lengths = []
        all_coverages = []
        all_collision_rates = []

        # Per-scene statistics
        per_scene_stats = {scene: [] for scene in self.scene_numbers}

        # Progress bar for evaluation
        total_episodes = len(self.scene_numbers) * self.episodes_per_scene
        pbar = tqdm(total=total_episodes, desc=f"Eval Block {block_num}", ncols=100, leave=False)

        # Evaluate each scene
        for scene_num in self.scene_numbers:
            for ep_idx in range(self.episodes_per_scene):
                # Reset environment with specific scene (reuse environment!)
                obs = self.env.reset(scene_number=scene_num, random_start=True)

                # Reset agent hidden states (LSTM/Transformer)
                agent.reset()

                episode_reward = 0.0
                episode_steps = 0
                collision_count = 0

                # Run episode
                while not (obs.terminated or obs.truncated):
                    # Get action stochastically (dist.sample, NOT argmax!)
                    # This avoids deadlocks when multiple actions are similarly good
                    with torch.no_grad():
                        action, *_ = agent.get_action(obs, deterministic=False)

                    # Step environment
                    obs = self.env.step(action)
                    episode_reward += obs.reward
                    episode_steps += 1

                    # Track collisions (failed move actions)
                    if not obs.info.get("move_action_success", True):
                        collision_count += 1

                # Collect episode statistics from environment info
                score = obs.info.get("score", 0.0)
                path_length = obs.info.get("total_path_length", 0.0)
                coverage = obs.info.get("exploration_coverage", 0.0)

                # Calculate collision rate for this episode
                collision_rate = collision_count / max(1, episode_steps)

                # Accumulate statistics
                all_scores.append(score)
                all_steps.append(episode_steps)
                all_rewards.append(episode_reward)
                all_path_lengths.append(path_length)
                all_coverages.append(coverage)
                all_collision_rates.append(collision_rate)

                # Per-scene tracking
                per_scene_stats[scene_num].append(score)

                pbar.update(1)

        pbar.close()

        # Compute aggregated statistics
        mean_stats = {
            "mean_score": np.mean(all_scores),
            "std_score": np.std(all_scores),
            "mean_steps": np.mean(all_steps),
            "std_steps": np.std(all_steps),
            "mean_reward": np.mean(all_rewards),
            "std_reward": np.std(all_rewards),
            "mean_path_length": np.mean(all_path_lengths),
            "mean_coverage": np.mean(all_coverages),
            "mean_collision_rate": np.mean(all_collision_rates),
        }

        # Log to TensorBoard with "Eval/" prefix
        writer.add_scalar("Eval/Mean_Score", mean_stats["mean_score"], block_num)
        writer.add_scalar("Eval/Std_Score", mean_stats["std_score"], block_num)
        writer.add_scalar("Eval/Mean_Steps", mean_stats["mean_steps"], block_num)
        writer.add_scalar("Eval/Std_Steps", mean_stats["std_steps"], block_num)
        writer.add_scalar("Eval/Mean_Reward", mean_stats["mean_reward"], block_num)
        writer.add_scalar("Eval/Std_Reward", mean_stats["std_reward"], block_num)
        writer.add_scalar("Eval/Mean_Path_Length", mean_stats["mean_path_length"], block_num)
        writer.add_scalar("Eval/Mean_Coverage", mean_stats["mean_coverage"], block_num)
        writer.add_scalar("Eval/Mean_Collision_Rate", mean_stats["mean_collision_rate"], block_num)

        # Log per-scene breakdown
        for scene_num in self.scene_numbers:
            scene_scores = per_scene_stats[scene_num]
            scene_mean = np.mean(scene_scores)
            scene_std = np.std(scene_scores)
            writer.add_scalar(f"Eval/FP{scene_num}_Mean_Score", scene_mean, block_num)
            writer.add_scalar(f"Eval/FP{scene_num}_Std_Score", scene_std, block_num)

        # Set agent back to train mode
        agent.train()

        return mean_stats

    def close(self):
        """Close the evaluation environment."""
        if hasattr(self, "env"):
            self.env.close()
