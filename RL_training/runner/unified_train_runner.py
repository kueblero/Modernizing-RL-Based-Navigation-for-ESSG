"""
Unified Training Runner for all RL Agents (REINFORCE, A2C, PPO)

Uses VecEnv for parallel environment sampling for all agents.
Centralizes all logging logic in one place.
"""

import json
import os
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from components.utils.curriculum_utils import (
    compute_geometric_stages,
    parse_curriculum_stage,
    print_curriculum_stages,
)


class UnifiedTrainRunner:
    """
    Unified training runner that works with all agents (REINFORCE, A2C, PPO).

    Key features:
    - VecEnv-based parallel training for all agents
    - Centralized TensorBoard logging
    - IL pre-training support
    - Curriculum learning support
    - Consistent checkpointing
    """

    def __init__(self, vec_env, agent, device, config, log_dir=None):
        """
        Args:
            vec_env: VecEnv instance with parallel environments
            agent: Agent instance (REINFORCE, A2C, or PPO)
            device: torch device
            config: Full configuration dict with agent, navigation, env, training sections
        """
        self.vec_env = vec_env
        self.agent = agent
        self.device = device
        self.config = config

        # Extract config sections
        self.agent_config = config["agent"]
        self.navigation_config = config["navigation"]
        self.env_config = config["env"]
        self.training_config = config.get("training", {})
        self.use_legacy_actions = self.env_config.get("use_legacy_actions", False)
        self.use_single_head_large = self.env_config.get("action_space_mode") == "single_head_large"

        # Training parameters
        self.n_episodes = self.agent_config.get("episodes", 1000)
        self.n_blocks = self.agent_config.get("blocks", self.training_config.get("blocks"))
        self.progress_unit = "episodes"
        if self.n_blocks is not None:
            self.n_blocks = int(self.n_blocks)
            self.progress_unit = "blocks"
        self.save_interval = self.training_config.get("save_interval", 100)
        self.log_interval = self.training_config.get("log_interval", 10)
        self.log_buffer_size = 40

        # Evaluation configuration (distinct from console logging)
        eval_config = self.config.get("evaluation", {})
        self.eval_interval = eval_config.get("eval_interval", None)  # None = disabled
        self.eval_callback = None  # Set externally for held-out evaluation

        # Adaptive curriculum window size (only increase if adaptive mode is enabled)
        curriculum_cfg = self.training_config.get("curriculum", {})
        curriculum_enabled = curriculum_cfg.get("enabled", False)
        curriculum_mode = curriculum_cfg.get("mode", "block")

        if curriculum_enabled and curriculum_mode == "adaptive":
            eval_window = curriculum_cfg.get("evaluation_window", 75)
            self.rollout_window_blocks = max(int(self.training_config.get("rollout_window_blocks", 20)), eval_window)
        else:
            self.rollout_window_blocks = int(self.training_config.get("rollout_window_blocks", 20))

        self.rollout_window = deque(maxlen=self.rollout_window_blocks)

        # Curriculum learning
        self.curriculum_config = self.training_config.get("curriculum", {})
        self.curriculum_enabled = self.curriculum_config.get("enabled", False)
        self.curriculum_mode = self.curriculum_config.get("mode", "block")  # block (default), adaptive, hard
        self.min_stage_blocks = self.curriculum_config.get("min_stage_blocks", 0)
        self.force_promotion_blocks = self.curriculum_config.get("force_promotion_blocks")
        self.force_promotion_episodes = self.curriculum_config.get("force_promotion_episodes")
        self._stage_start_block = 0
        self._stage_start_episode = 0
        self._window_history = deque(maxlen=20)  # rolling curriculum windows
        self._last_plateau_debug = None  # stores last adaptive plateau check details

        # Compute curriculum stages dynamically using geometric progression
        self.n_stages = self.curriculum_config.get("n_stages", 6)  # Default: 6 stages
        self.curriculum_stages = compute_geometric_stages(n_stages=self.n_stages)
        # curriculum_stages is list of (stage_num, n_dirs, n_lens) tuples

        # Parse initial curriculum stage (only for multi-head actions)
        if self.use_legacy_actions:
            # Legacy actions: no curriculum, always "stage 1" (but not used)
            self.current_curriculum_stage = 1
        else:
            # Multi-head actions: parse curriculum stage (supports int or "last"/"max"/"full")
            initial_stage_value = self.env_config.get("curriculum_stage", 1)
            # Handle None explicitly (in case config has curriculum_stage: null)
            # null = full action space (last stage)
            if initial_stage_value is None:
                initial_stage_value = self.n_stages
            self.current_curriculum_stage = parse_curriculum_stage(initial_stage_value, self.n_stages)

        if self.curriculum_enabled:
            # Compute block schedule for curriculum stages
            # If explicit stage blocks are provided, use them; otherwise compute geometrically
            if "stage_blocks" in self.curriculum_config:
                # Explicit block schedule provided
                self.stage_blocks = self.curriculum_config["stage_blocks"]
                if len(self.stage_blocks) != self.n_stages - 1:
                    raise ValueError(
                        f"stage_blocks must have {self.n_stages - 1} entries (boundaries between {self.n_stages} stages), "
                        f"got {len(self.stage_blocks)}"
                    )
            else:
                # Compute geometric block schedule
                # Default: stage transitions at 120, 180, 270, 360, 450 for 6 stages
                # For n_stages, we compute: total_blocks / n_stages * [1, 1.5, 2.25, ...]
                base_blocks_per_stage = 120  # Default blocks for first stage
                ratio = 1.5  # Geometric ratio for block growth
                self.stage_blocks = []
                cumulative_blocks = 0
                for i in range(self.n_stages - 1):
                    cumulative_blocks += int(base_blocks_per_stage * (ratio ** i))
                    self.stage_blocks.append(cumulative_blocks)

            # Entropy boost settings for stage transitions
            self.entropy_boost_factor = self.curriculum_config.get("entropy_boost_factor", 1.5)
            self.entropy_decay_blocks = self.curriculum_config.get("entropy_decay_blocks", 12)
            self.original_entropy_coef = self.agent_config.get("entropy_coef", 0.01)
            self.entropy_boost_counter = 0  # Counter for decaying entropy back to normal

        # Tracking
        self.episode_count = 0
        self.update_count = 0
        self.ep_info_buffer = deque(maxlen=self.log_buffer_size)
        self.best_mean_score = -float("inf")
        self.max_score = -float("inf")

        # Action distribution tracking for curriculum learning
        # Use deques with maxlen to prevent unbounded memory growth
        self.action_counts_dir = {}  # {dir_idx: count} - Reset periodically
        self.action_counts_len = {}  # {len_idx: count} - Reset periodically
        self.action_counts_combined = {}  # {(dir_idx, len_idx): count} - Reset periodically
        self.actions_since_last_log = deque(maxlen=2000)  # FIXED: Use deque with limit to prevent memory leak
        self.new_actions_at_stage_transition = set()  # Track newly unlocked actions
        self.new_action_coverage = 0  # Percentage of new actions tried
        self._action_count_reset_interval = 100  # Reset action counts every N blocks

        # Weight interpolation tracking for curriculum transitions
        self.prev_valid_dir_indices = None
        self.prev_valid_len_indices = None

        # Reward component tracking for TensorBoard
        self.reward_components = {
            "recall_node": deque(maxlen=100),
            "recall_edge": deque(maxlen=100),
            "collision_rate": deque(maxlen=100),
            "time_penalty": deque(maxlen=100),
            "total_reward": deque(maxlen=100),
        }

        # TensorBoard setup
        if log_dir != "no_logging":
            self._setup_tensorboard(log_dir)

    def _setup_tensorboard(self, log_dir):
        """Setup TensorBoard writer and log directory."""
        if log_dir is not None:
            # Use provided log_dir directly (already includes scenario name, seed, timestamp)
            self.log_dir = log_dir
        else:
            # Create default log directory with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            scenario_name = self.config.get("name", "RL_Training").replace(" ", "_").replace(":", "")
            self.log_dir = f"RL_training/runs/{scenario_name}_{timestamp}"

        self.writer = SummaryWriter(self.log_dir)
        print(f"[INFO] TensorBoard logs will be saved to: {self.log_dir}")

        # Log full configuration
        self.writer.add_text("full_config", json.dumps(self.config, indent=2), 0)

    def run(self, save_model=False):
        """
        Main training loop.

        Args:
            save_model: Whether to save model checkpoints
        """
        print(f"\n{'='*70}")
        print(f"[INFO] Starting Unified Training: {self.config.get('name', 'RL Training')}")
        print(f"[INFO] Agent: {self.agent_config['name'].upper()}")
        # Determine action space mode for logging
        if self.env_config.get('use_legacy_actions'):
            action_space_str = 'Legacy (16)'
        elif self.env_config.get('action_space_mode') == 'single_head_large':
            action_space_str = 'Single-Head-Large (504)'
        else:
            action_space_str = 'Multi-Head (24×21 + STOP)'
        print(f"[INFO] Action Space: {action_space_str}")
        if self.progress_unit == "blocks":
            n_steps_per_env = int(self.agent_config.get("num_steps", 0))
            print(f"[INFO] Target Blocks: {self.n_blocks} (each block: {n_steps_per_env} steps/env)")
        else:
            print(f"[INFO] Target Episodes: {self.n_episodes}")
        print(f"[INFO] Parallel Envs: {self.vec_env.num_envs}")
        print(f"[INFO] RHO: {self.env_config['rho']}")

        if self.curriculum_enabled:
            print(f"[INFO] Curriculum Learning: Enabled ({self.n_stages} stages)")
            print(f"[INFO] Curriculum Mode: {self.curriculum_mode.capitalize()}")

            # Print stage definitions (action space progression)
            print(f"[INFO] Stage Definitions:")
            for stage_num, n_dirs, n_lens in self.curriculum_stages:
                n_actions = n_dirs * n_lens
                angle_step = 360 / n_dirs
                print(f"  - Stage {stage_num}: {n_actions:3d} actions ({n_dirs:2d} dirs × {n_lens:2d} lens) - {angle_step:.0f}° steps")

            # Print stage transitions (block schedule) for block/hard modes
            if self.curriculum_mode != "adaptive":
                print(f"[INFO] Stage Transitions (Block Schedule):")
                for i, (stage_num, _, _) in enumerate(self.curriculum_stages):
                    if i == 0:
                        block_range = f"blocks 0-{self.stage_blocks[0]}"
                    elif i == self.n_stages - 1:
                        block_range = f"blocks {self.stage_blocks[-1]}+"
                    else:
                        block_range = f"blocks {self.stage_blocks[i-1]}-{self.stage_blocks[i]}"
                    print(f"  - Stage {stage_num}: {block_range}")
            else:
                print(f"[INFO] Stage Transitions: Adaptive (based on plateau detection)")
                print(f"  - Min blocks per stage: {self.min_stage_blocks}")
                print(f"  - Force promotion after: {self.force_promotion_blocks} blocks")

        print(f"{'='*70}\n")

        # Create buffer for the agent
        from components.utils.vec_rollout_buffer import VecEnvRolloutBuffer

        n_steps_per_env = self.agent_config["num_steps"]
        buffer = VecEnvRolloutBuffer(n_envs=self.vec_env.num_envs, n_steps_per_env=n_steps_per_env)

        update_idx = 0
        progress_total = self.n_blocks if self.progress_unit == "blocks" else self.n_episodes
        progress_desc = "Training Blocks" if self.progress_unit == "blocks" else "Training Episodes"
        pbar = tqdm(total=progress_total, desc=progress_desc, ncols=160)

        while (update_idx < self.n_blocks) if self.progress_unit == "blocks" else (self.episode_count < self.n_episodes):
            update_idx += 1
            self.update_count = update_idx

            # Update curriculum stage if needed
            self._maybe_update_curriculum()

            # Collect rollouts from all parallel environments
            block_start = time.time()
            rollout_stats = self._collect_rollouts(buffer, n_steps=n_steps_per_env)
            self._update_rollout_window(rollout_stats)

            # Track episodes
            episodes_done = rollout_stats.get("episodes_done", 0)
            self.episode_count += episodes_done

            # Decay entropy coefficient if in boost phase (per block)
            if self.curriculum_enabled:
                self._decay_entropy_boost()

            # Get batch and update agent
            batch = buffer.get(gamma=self.agent.gamma, gae_lambda=getattr(self.agent, "gae_lambda", None))
            update_stats = self.agent.update(batch)

            # Clear buffer
            buffer.clear()

            # Optional CUDA memory debug (writes to TensorBoard under debug/)
            if torch.cuda.is_available() and hasattr(self, "writer") and self.writer is not None:
                alloc = torch.cuda.memory_allocated() / 1e9
                reserved = torch.cuda.memory_reserved() / 1e9
                self.writer.add_scalar("debug/cuda_memory_allocated_gb", alloc, self.update_count)
                self.writer.add_scalar("debug/cuda_memory_reserved_gb", reserved, self.update_count)
            elif self.writer is not None:
                print("[WARNING] CUDA not available for memory debug logging.")

            # Collect episode statistics and update progress bar
            if self.progress_unit == "blocks":
                pbar.update(1)
            else:
                pbar.update(episodes_done)

            if "mean_episode_reward" in rollout_stats:
                block_time = time.time() - block_start
                self.ep_info_buffer.append(
                    {
                        "reward": rollout_stats.get("mean_episode_reward", 0),
                        "steps": rollout_stats.get("mean_episode_steps", 0),
                        "score": rollout_stats.get("mean_episode_score", 0),
                    }
                )

                # Update progress bar with current stats
                pbar.set_postfix(
                    {
                        "Episode": f"{self.episode_count}",
                        "Score": f"{rollout_stats.get('mean_episode_score', 0):.2f}",
                        "Steps": f"{rollout_stats.get('mean_episode_steps', 0):.1f}",
                        "Reward": f"{rollout_stats.get('mean_episode_reward', 0):.2f}",
                        "Loss": f"{update_stats.get('policy_loss', update_stats.get('loss', 0)):.4f}",
                        "Block_s": f"{block_time:5.1f}",
                    }
                )

            # Log to TensorBoard after every update
            self._log_training_stats(update_stats, rollout_stats, update_idx)

            # Console logging every log_interval episodes
            progress_step = update_idx if self.progress_unit == "blocks" else self.episode_count
            if progress_step > 0 and progress_step % self.log_interval == 0:
                pass  # Detailed console logging is already done in _log_training_stats

            # Save checkpoint
            if save_model and progress_step > 0 and progress_step % self.save_interval == 0:
                self._save_checkpoint(progress_step)

            # Periodic evaluation callback (if configured)
            if self.eval_callback and self.eval_interval and progress_step > 0 and progress_step % self.eval_interval == 0:
                self.eval_callback(progress_step, self.agent)

        pbar.close()

        # Cleanup and save final results
        self._finalize_training(save_model)

    def _maybe_update_curriculum(self):
        """Update curriculum stage based on configured strategy."""
        if not self.curriculum_enabled or self.env_config.get("use_legacy_actions", False):
            return

        mode = (self.curriculum_mode or "block").lower()
        blocks_in_stage = self.update_count - self._stage_start_block
        episodes_in_stage = self.episode_count - self._stage_start_episode

        # Hard backstop: force promotion after a budget (optional)
        if mode != "block":
            if self.force_promotion_blocks and blocks_in_stage >= self.force_promotion_blocks:
                next_stage = self.current_curriculum_stage + 1
                if next_stage <= self.n_stages:  # Don't exceed max stages
                    self._apply_curriculum_stage_change(next_stage, reason="force_blocks")
                return
            if self.force_promotion_episodes and episodes_in_stage >= self.force_promotion_episodes:
                next_stage = self.current_curriculum_stage + 1
                if next_stage <= self.n_stages:  # Don't exceed max stages
                    self._apply_curriculum_stage_change(next_stage, reason="force_episodes")
                return

        # Block-based schedule (uses dynamically computed stage_blocks)
        if mode == "block":
            # Determine current stage based on block count
            new_stage = 1
            for i, threshold in enumerate(self.stage_blocks):
                if self.update_count >= threshold:
                    new_stage = i + 2  # Stage is 1-indexed, and we're past threshold

            # Apply stage change if needed
            if new_stage != self.current_curriculum_stage:
                # Ensure we don't exceed max stages
                new_stage = min(new_stage, self.n_stages)
                self._apply_curriculum_stage_change(new_stage, reason="block_schedule")
            return

        # New adaptive plateau detection (uses rollout_window history)
        if mode == "adaptive":
            # Check if we're already at the final stage
            if self.current_curriculum_stage >= self.n_stages:
                return  # Already at max stage, no more promotion possible

            promote = self._check_adaptive_plateau()
            if promote and blocks_in_stage >= self.min_stage_blocks:
                next_stage = self.current_curriculum_stage + 1
                if next_stage <= self.n_stages:  # Safety check
                    self._apply_curriculum_stage_change(next_stage, reason="adaptive_plateau")
            return

        # Legacy hard threshold mode (kept for backwards compatibility)
        if mode == "hard":
            windows_needed = 1
            required_len = max(2, windows_needed + 1)
            if len(self._window_history) < required_len:
                return

            cur = self._window_history[-1]
            max_actions = max(1, self.env_config.get("max_actions", 1))

            ht = self.curriculum_config.get("hard_thresholds", {})
            success_ok = cur.get("success_rate", 0.0) >= ht.get("success_rate", 1.0)
            score_ok = cur.get("mean_score", 0.0) >= ht.get("mean_score", 1.0)
            reward_ok = cur.get("mean_reward", 0.0) >= ht.get("mean_reward", 0.0)
            steps_frac = cur.get("mean_steps", 0.0) / max_actions
            steps_ok = steps_frac <= ht.get("mean_steps_frac", 1.0)
            collision_ok = True
            if ht.get("collision_rate") is not None and cur.get("collision_rate") is not None:
                collision_ok = cur.get("collision_rate") <= ht.get("collision_rate")
            promote = success_ok and score_ok and reward_ok and steps_ok and collision_ok

            if promote and blocks_in_stage >= self.min_stage_blocks:
                next_stage = self.current_curriculum_stage + 1
                if next_stage <= self.n_stages:  # Don't exceed max stages
                    self._apply_curriculum_stage_change(next_stage, reason="hard_criteria")

    def _check_adaptive_plateau(self):
        """
        Check if training has plateaued using EMA-smoothed recent vs earlier window comparison.

        Config parameters:
            evaluation_window: Total window size to consider (default: 75)
            recent_window: Size of recent window to compare (default: 15)
            plateau_threshold: Relative improvement threshold (default: 0.02 = 2%)
            metrics: List of metrics to check (default: ["mean_score", "mean_reward", "mean_steps"])

        Returns:
            bool: True if all metrics have plateaued
        """
        # Get config with sensible defaults
        eval_window = self.curriculum_config.get("evaluation_window", 75)
        recent_n = self.curriculum_config.get("recent_window", 15)
        threshold = self.curriculum_config.get("plateau_threshold", 0.02)
        metrics = self.curriculum_config.get("metrics", ["mean_score", "mean_reward", "mean_steps"])

        # Need enough data
        if len(self.rollout_window) < eval_window:
            self._last_plateau_debug = {
                "block": self.update_count,
                "episode": self.episode_count,
                "reason": "insufficient_history",
                "history_len": len(self.rollout_window),
                "required_len": eval_window,
            }
            return False

        # Extract metric histories from rollout_window
        metric_histories = {}
        for metric in metrics:
            history = []
            for entry in self.rollout_window:
                # Compute metrics from rollout window entries
                episodes = entry["episodes_done"]
                if episodes > 0:
                    if metric == "mean_reward":
                        history.append(entry["episode_reward_sum"] / episodes)
                    elif metric == "mean_score":
                        history.append(entry["episode_score_sum"] / episodes)
                    elif metric == "mean_steps":
                        history.append(entry["episode_steps_sum"] / episodes)
            metric_histories[metric] = history

        # Check each metric for plateau
        plateaued = []
        debug_metrics = []
        for metric in metrics:
            history = metric_histories.get(metric, [])
            if len(history) < eval_window:
                plateaued.append(False)
                debug_metrics.append(
                    {
                        "metric": metric,
                        "plateau": False,
                        "reason": "short_history",
                        "history_len": len(history),
                        "required_len": eval_window,
                    }
                )
                continue

            # Take last eval_window blocks
            recent_history = history[-eval_window:]

            # Apply EMA smoothing (alpha=0.2 for strong smoothing)
            smoothed = self._compute_ema(recent_history, alpha=0.2)

            # Compare recent vs earlier
            recent_mean = np.mean(smoothed[-recent_n:])
            earlier_mean = np.mean(smoothed[:-recent_n])

            # Relative improvement (handle negative values and zero denominators)
            denom = max(abs(earlier_mean), 1e-6)
            rel_improvement = (recent_mean - earlier_mean) / denom

            # For mean_steps: lower is better, so we check abs() since plateau can be in either direction
            is_flat = abs(rel_improvement) < threshold

            plateaued.append(is_flat)
            debug_metrics.append(
                {
                    "metric": metric,
                    "plateau": bool(is_flat),
                    "earlier_mean": float(earlier_mean),
                    "recent_mean": float(recent_mean),
                    "rel_improvement": float(rel_improvement),
                    "threshold": float(threshold),
                    "recent_window": recent_n,
                    "eval_window": eval_window,
                }
            )
            # Log window means and relative improvement to TensorBoard for visibility
            if hasattr(self, "writer") and self.writer is not None:
                step = self.update_count
                self.writer.add_scalar(f"curriculum/metric/{metric}/earlier_mean", float(earlier_mean), step)
                self.writer.add_scalar(f"curriculum/metric/{metric}/recent_mean", float(recent_mean), step)
                self.writer.add_scalar(f"curriculum/metric/{metric}/rel_improvement", float(rel_improvement), step)

        self._last_plateau_debug = {
            "block": self.update_count,
            "episode": self.episode_count,
            "metrics": debug_metrics,
            "threshold": float(threshold),
            "recent_window": recent_n,
            "eval_window": eval_window,
        }

        # All metrics must plateau for promotion
        return all(plateaued)

    @staticmethod
    def _compute_ema(values, alpha=0.2):
        """
        Compute Exponential Moving Average for smoothing noisy signals.

        Args:
            values: List of values
            alpha: Smoothing factor (0 < alpha <= 1). Lower = more smoothing

        Returns:
            np.array: EMA-smoothed values
        """
        if not values:
            return np.array([])

        ema = np.zeros(len(values))
        ema[0] = values[0]

        for i in range(1, len(values)):
            ema[i] = alpha * values[i] + (1 - alpha) * ema[i - 1]

        return ema

    def _apply_curriculum_stage_change(self, new_stage, reason="criteria"):
        """Apply curriculum stage change with bookkeeping and env/agent updates."""
        new_stage = int(new_stage)
        new_stage = max(1, min(new_stage, 6))
        if new_stage == self.current_curriculum_stage:
            return

        # Save old valid indices before updating stage
        try:
            if self.prev_valid_dir_indices is None:
                self.prev_valid_dir_indices = list(self.vec_env.get_attr("valid_direction_indices", indices=[0])[0])
                self.prev_valid_len_indices = list(self.vec_env.get_attr("valid_length_indices", indices=[0])[0])
        except Exception as e:
            print(f"[WARNING] Could not get previous valid indices: {e}")

        self.current_curriculum_stage = new_stage
        self._stage_start_block = self.update_count
        self._stage_start_episode = self.episode_count
        print(f"\n[INFO] Curriculum stage updated to {self.current_curriculum_stage} (reason: {reason})")

        # Boost entropy to encourage exploration of new actions
        old_entropy = self.agent.entropy_coef
        self.agent.entropy_coef = self.original_entropy_coef * self.entropy_boost_factor
        self.entropy_boost_counter = self.entropy_decay_blocks
        print(f"[INFO] Entropy coefficient boosted: {old_entropy:.4f} -> {self.agent.entropy_coef:.4f}")
        print(f"[INFO] Will decay back to {self.original_entropy_coef:.4f} over {self.entropy_decay_blocks} blocks")

        # Update all environments using call_method
        try:
            self.vec_env.call_method("set_curriculum_stage", args=[self.current_curriculum_stage])

            # Also update agent's environment (used for action masking in get_action)
            if hasattr(self.agent.env, "set_curriculum_stage"):
                self.agent.env.set_curriculum_stage(self.current_curriculum_stage)

            # Track which actions are newly unlocked at this stage
            valid_dir_indices = self.vec_env.get_attr("valid_direction_indices", indices=[0])[0]
            valid_len_indices = self.vec_env.get_attr("valid_length_indices", indices=[0])[0]

            # Create set of all newly valid action combinations
            current_valid_actions = {(d, l) for d in valid_dir_indices for l in valid_len_indices}

            # Calculate new actions based on STAGE DEFINITIONS (not executed actions)
            if self.current_curriculum_stage == 1:
                # First stage: all actions are new
                self.new_actions_at_stage_transition = current_valid_actions
                num_new_actions = len(current_valid_actions)
            else:
                # Find previous stage definition and reconstruct previous valid actions
                prev_stage_idx = self.current_curriculum_stage - 2  # Stage 2 → index 0
                if prev_stage_idx >= 0 and prev_stage_idx < len(self.curriculum_stages):
                    from components.utils.curriculum_utils import get_valid_direction_indices, get_valid_length_indices
                    _, prev_n_dirs, prev_n_lens = self.curriculum_stages[prev_stage_idx]

                    # Reconstruct previous stage's valid actions
                    prev_dir_indices = get_valid_direction_indices(prev_n_dirs)
                    prev_len_indices = get_valid_length_indices(prev_n_lens)
                    prev_valid_actions = {(d, l) for d in prev_dir_indices for l in prev_len_indices}

                    # New actions = current - previous
                    self.new_actions_at_stage_transition = current_valid_actions - prev_valid_actions
                else:
                    # Fallback: all actions are new
                    self.new_actions_at_stage_transition = current_valid_actions

                num_new_actions = len(self.new_actions_at_stage_transition)

            print(f"[INFO] {num_new_actions} new actions unlocked at stage {self.current_curriculum_stage}")
            print(
                f"[INFO] Valid actions: {len(valid_dir_indices)} dirs × {len(valid_len_indices)} lens = {len(current_valid_actions)} total"
            )

            # Update previous valid indices for next transition
            self.prev_valid_dir_indices = list(valid_dir_indices)
            self.prev_valid_len_indices = list(valid_len_indices)

        except Exception as e:
            print(f"[WARNING] Could not update curriculum stage: {e}")

    def _decay_entropy_boost(self):
        """
        Linearly decay entropy coefficient back to original value per block.
        Called once per training block/update.
        """
        if self.entropy_boost_counter <= 0:
            return

        # Decrease counter by 1 (one block)
        self.entropy_boost_counter = max(0, self.entropy_boost_counter - 1)

        # Linear interpolation between boosted and original entropy
        if self.entropy_boost_counter > 0:
            # Still in decay phase
            progress = 1.0 - (self.entropy_boost_counter / self.entropy_decay_blocks)
            boosted_entropy = self.original_entropy_coef * self.entropy_boost_factor
            self.agent.entropy_coef = boosted_entropy * (1 - progress) + self.original_entropy_coef * progress
        else:
            # Decay complete
            self.agent.entropy_coef = self.original_entropy_coef

    def _collect_rollouts(self, buffer, n_steps):
        """
        Collect rollouts from vectorized environments.
        Based on vec_trial_runner.py implementation.

        Args:
            buffer: VecEnvRolloutBuffer instance
            n_steps: Number of steps to collect per environment

        Returns:
            dict: Statistics about rollout collection
        """
        import torch

        n_envs = self.vec_env.num_envs
        steps_collected = 0
        episodes_done = 0
        episode_rewards = []
        episode_steps = []
        episode_scores = []
        episode_terminated = []  # Track if episode ended successfully (terminated) or truncated
        episode_steps_success = []  # Steps for successful episodes only
        episode_steps_truncated = []  # Steps for truncated episodes only
        episode_path_lengths = []
        episode_coverages = []
        block_recall_node_sum = 0.0
        block_recall_edge_sum = 0.0
        block_collision_rate_sum = 0.0
        block_time_penalty_sum = 0.0
        block_total_reward_sum = 0.0
        block_reward_count = 0
        block_path_length_sum = 0.0
        block_path_length_count = 0
        block_coverage_sum = 0.0
        block_coverage_count = 0
        block_action_total = 0
        block_action_pure_rotations = 0
        block_action_movements = 0
        block_action_stops = 0
        block_action_idle = 0

        # Initialize if needed
        if not hasattr(self, "_obs_list"):
            self._obs_list, _ = self.vec_env.reset()
            self._hiddens_list = [(None, None, None, None) for _ in range(n_envs)]
            self._last_actions = [torch.tensor([-1, -1], dtype=torch.long, device=self.device) for _ in range(n_envs)]
            self._agent_positions = [None for _ in range(n_envs)]
            self._episode_rewards = [0.0 for _ in range(n_envs)]
            self._episode_steps = [0 for _ in range(n_envs)]
            # Episode-level reward component accumulators
            self._ep_reward_components = [
                {"recall_node_sum": 0.0, "recall_edge_sum": 0.0, "collision_count": 0, "step_count": 0} for _ in range(n_envs)
            ]

        for step in range(n_steps):
            # Get actions from all environments
            actions = []
            hiddens_new = []

            if hasattr(self.agent, "act_batch"):
                actions, hiddens_new, _, _ = self.agent.act_batch(
                    self._obs_list, self._hiddens_list, last_actions_list=self._last_actions, deterministic=False
                )
                if self.use_legacy_actions or self.use_single_head_large:
                    # Convert single-head actions to tuples (action_idx, -1) for embedding compatibility
                    actions = [(a[0], -1) if isinstance(a, (tuple, list)) else (a, -1) for a in actions]
            else:
                for env_idx in range(n_envs):
                    obs = self._obs_list[env_idx]
                    # Set agent's hidden states (including stagnation tracker)
                    if len(self._hiddens_list[env_idx]) == 4:
                        self.agent.lssg_hidden, self.agent.gssg_hidden, self.agent.policy_hidden, self.agent.stagnation_state = (
                            self._hiddens_list[env_idx]
                        )
                    else:
                        self.agent.lssg_hidden, self.agent.gssg_hidden, self.agent.policy_hidden = self._hiddens_list[env_idx]
                        self.agent.stagnation_state = None
                    # Ensure per-env action history is respected (prevents cross-env contamination)
                    self.agent.last_action = self._last_actions[env_idx]

                    # Get action - returns (action, lssg_hidden, gssg_hidden, policy_hidden, prev_action, value)
                    result = self.agent.get_action(obs, deterministic=False)

                    if self.use_legacy_actions or self.use_single_head_large:
                        # Legacy/Single-head-large: returns (action_idx, lssg, gssg, policy, prev, val, stag)
                        action_idx, lssg_h, gssg_h, policy_h, _, _, stag_state = result
                        actions.append((action_idx, -1))  # Store as tuple for embedding compatibility
                    else:
                        # Multi-head: returns (action_tuple, lssg, gssg, policy, prev, val, stag)
                        action, lssg_h, gssg_h, policy_h, _, _, stag_state = result
                        actions.append(action)

                    hiddens_new.append((lssg_h, gssg_h, policy_h, stag_state))

            # Track action types per block (pure rotation vs movement vs stop)
            if self.use_legacy_actions:
                for action in actions:
                    action_idx = action[0] if isinstance(action, (tuple, list)) else action
                    # Legacy mapping: idx = angle_idx * 2 + len_idx, where len_idx 0=0.0m, 1=0.3m.
                    len_idx = int(action_idx) % 2
                    angle_idx = int(action_idx) // 2
                    if len_idx == 0:
                        if angle_idx == 0:
                            block_action_stops += 1
                        else:
                            block_action_pure_rotations += 1
                    else:
                        block_action_movements += 1
                    block_action_total += 1
            elif self.use_single_head_large:
                for action in actions:
                    action_idx = action[0] if isinstance(action, (tuple, list)) else action
                    # Single-head-large mapping: idx = dir_idx * 21 + len_idx (24 dirs x 21 lens = 504)
                    dir_idx = int(action_idx) // 21
                    len_idx = int(action_idx) % 21
                    if dir_idx == 0 and len_idx == 0:
                        block_action_stops += 1
                    elif len_idx == 0:
                        block_action_pure_rotations += 1
                    else:
                        block_action_movements += 1
                    block_action_total += 1
            else:
                for action in actions:
                    # Handle both int (single-head-large) and tuple (multi-head) actions
                    if isinstance(action, int):
                        # Single-head-large: decode flat index
                        dir_idx = action // 21
                        len_idx = action % 21
                        stop_flag = 0  # No separate STOP flag in single-head-large
                    elif isinstance(action, (tuple, list)) and len(action) >= 2:
                        # Multi-head: tuple format
                        dir_idx, len_idx = int(action[0]), int(action[1])
                        stop_flag = int(action[2]) if len(action) >= 3 else 0
                    else:
                        continue

                    if stop_flag == 1 or (dir_idx == 0 and len_idx == 0):
                        block_action_stops += 1
                    elif len_idx == 0:
                        if dir_idx == 0:
                            block_action_idle += 1
                        else:
                            block_action_pure_rotations += 1
                    else:
                        block_action_movements += 1
                    block_action_total += 1

            # Track actions for distribution logging (multi-head and single-head-large)
            if not self.use_legacy_actions:
                for action in actions:
                    if self.use_single_head_large:
                        # Single-head-large: action is tuple (flat_idx, -1), decode to (dir_idx, len_idx)
                        flat_idx = action[0] if isinstance(action, (tuple, list)) else action
                        dir_idx = flat_idx // 21
                        len_idx = flat_idx % 21
                    else:
                        # Multi-head: action is (dir_idx, len_idx) or (dir_idx, len_idx, stop_flag)
                        dir_idx, len_idx = action[0], action[1]
                    # Update action counters
                    self.action_counts_dir[dir_idx] = self.action_counts_dir.get(dir_idx, 0) + 1
                    self.action_counts_len[len_idx] = self.action_counts_len.get(len_idx, 0) + 1
                    self.action_counts_combined[(dir_idx, len_idx)] = self.action_counts_combined.get((dir_idx, len_idx), 0) + 1
                    self.actions_since_last_log.append((dir_idx, len_idx))

            # Step all environments
            next_obs_list, rewards, dones, truncated, infos = self.vec_env.step(actions)
            dones_combined = [d or t for d, t in zip(dones, truncated)]

            # Add transitions to buffer
            states = [obs.state for obs in self._obs_list]
            last_actions_buffer = [la.cpu().numpy().tolist() if isinstance(la, torch.Tensor) else la for la in self._last_actions]

            # CRITICAL: Detach hidden states to prevent gradient accumulation and memory leak
            # LSTM hidden states have requires_grad=True and accumulate computation graph
            hiddens_detached = []
            for hidden_tuple in self._hiddens_list:
                if len(hidden_tuple) == 4:
                    lssg_h, gssg_h, policy_h, stag_state = hidden_tuple
                else:
                    lssg_h, gssg_h, policy_h = hidden_tuple
                    stag_state = None

                # Detach LSTM hidden states (h, c) tuples
                lssg_h_det = (lssg_h[0].detach(), lssg_h[1].detach()) if lssg_h is not None else None
                gssg_h_det = (gssg_h[0].detach(), gssg_h[1].detach()) if gssg_h is not None else None
                policy_h_det = (policy_h[0].detach(), policy_h[1].detach()) if policy_h is not None else None

                # Detach stagnation state tensors (multi-head only)
                if stag_state is not None:
                    stag_state_det = {
                        "prev_gssg": stag_state["prev_gssg"].detach(),
                        "smoothed_change": stag_state["smoothed_change"].detach(),
                        "steps_since_discovery": stag_state["steps_since_discovery"].detach(),
                    }
                else:
                    stag_state_det = None

                hiddens_detached.append((lssg_h_det, gssg_h_det, policy_h_det, stag_state_det))

            buffer.add(
                states=states,
                actions=actions,
                rewards=rewards,
                dones=dones_combined,
                hiddens=hiddens_detached,
                last_actions=last_actions_buffer,
                agent_positions=self._agent_positions,
                move_successes=[info.get("move_action_success", True) for info in infos],
            )

            # Update per-environment tracking
            done_indices = []
            for env_idx in range(n_envs):
                self._episode_rewards[env_idx] += rewards[env_idx]
                self._episode_steps[env_idx] += 1

                # Track reward components from info dict
                info = infos[env_idx]
                self._ep_reward_components[env_idx]["recall_node_sum"] += info.get("recall_node", 0.0)
                self._ep_reward_components[env_idx]["recall_edge_sum"] += info.get("recall_edge", 0.0)
                self._ep_reward_components[env_idx]["step_count"] += 1
                if not info.get("move_action_success", True):
                    self._ep_reward_components[env_idx]["collision_count"] += 1

                if dones_combined[env_idx]:
                    episodes_done += 1
                    score = infos[env_idx].get("score", rewards[env_idx])

                    episode_rewards.append(self._episode_rewards[env_idx])
                    episode_steps.append(self._episode_steps[env_idx])
                    episode_scores.append(score)

                    path_length = info.get("total_path_length")
                    if path_length is not None:
                        episode_path_lengths.append(path_length)
                        block_path_length_sum += path_length
                        block_path_length_count += 1

                    coverage = info.get("exploration_coverage")
                    if coverage is not None:
                        episode_coverages.append(coverage)
                        block_coverage_sum += coverage
                        block_coverage_count += 1

                    # Track terminated vs truncated
                    is_terminated = dones[env_idx] and not truncated[env_idx]
                    episode_terminated.append(is_terminated)

                    if is_terminated:
                        episode_steps_success.append(self._episode_steps[env_idx])
                    else:
                        episode_steps_truncated.append(self._episode_steps[env_idx])

                    # Store reward components for this episode
                    ep_comps = self._ep_reward_components[env_idx]
                    n_steps_ep = max(1, ep_comps["step_count"])
                    recall_node_avg = ep_comps["recall_node_sum"] / n_steps_ep
                    recall_edge_avg = ep_comps["recall_edge_sum"] / n_steps_ep
                    collision_rate = ep_comps["collision_count"] / n_steps_ep
                    rho = self.env_config["rho"]
                    pure_rotations = min(int(infos[env_idx].get("num_pure_rotations", 0)), n_steps_ep)
                    time_penalty = rho * n_steps_ep - 0.5 * rho * pure_rotations
                    total_reward = self._episode_rewards[env_idx]

                    self.reward_components["recall_node"].append(recall_node_avg)
                    self.reward_components["recall_edge"].append(recall_edge_avg)
                    self.reward_components["collision_rate"].append(collision_rate)
                    self.reward_components["time_penalty"].append(time_penalty)
                    self.reward_components["total_reward"].append(total_reward)

                    block_recall_node_sum += recall_node_avg
                    block_recall_edge_sum += recall_edge_avg
                    block_collision_rate_sum += collision_rate
                    block_time_penalty_sum += time_penalty
                    block_total_reward_sum += total_reward
                    block_reward_count += 1

                    # Reset per-env trackers
                    self._episode_rewards[env_idx] = 0.0
                    self._episode_steps[env_idx] = 0
                    self._hiddens_list[env_idx] = (None, None, None, None)
                    self._last_actions[env_idx] = torch.tensor([-1, -1], dtype=torch.long, device=self.device)
                    self._agent_positions[env_idx] = None
                    self._ep_reward_components[env_idx] = {
                        "recall_node_sum": 0.0,
                        "recall_edge_sum": 0.0,
                        "collision_count": 0,
                        "step_count": 0,
                    }
                    done_indices.append(env_idx)
                else:
                    self._hiddens_list[env_idx] = hiddens_new[env_idx]
                    act = actions[env_idx]
                    if isinstance(act, (tuple, list)) and len(act) == 3:
                        act = (act[0], act[1])  # keep only (dir, len) for embedding/history
                    self._last_actions[env_idx] = torch.tensor(act, dtype=torch.long, device=self.device)
                    self._agent_positions[env_idx] = infos[env_idx].get("agent_pos", None)
                    self._obs_list[env_idx] = next_obs_list[env_idx]

            # Reset finished environments to keep batches full
            if done_indices:
                reset_obs_list, _ = self.vec_env.reset(done_indices)
                for idx in done_indices:
                    self._obs_list[idx] = reset_obs_list[idx]

            steps_collected += n_envs

        # Return statistics
        stats = {"steps_collected": steps_collected, "episodes_done": episodes_done}
        stats["actions_total"] = block_action_total
        stats["actions_pure_rotations"] = block_action_pure_rotations
        stats["actions_movements"] = block_action_movements
        stats["actions_stops"] = block_action_stops
        stats["actions_idle"] = block_action_idle
        stats["actions_pure_rotation_rate"] = block_action_pure_rotations / max(1, block_action_total)
        stats["actions_stop_rate"] = block_action_stops / max(1, block_action_total)
        stats["actions_pure_rotation_to_movement_ratio"] = block_action_pure_rotations / max(1, block_action_movements)

        if episode_rewards:
            reward_sum = float(sum(episode_rewards))
            steps_sum = float(sum(episode_steps))
            score_sum = float(sum(episode_scores))

            stats["episode_reward_sum"] = reward_sum
            stats["episode_steps_sum"] = steps_sum
            stats["episode_score_sum"] = score_sum
            stats["mean_episode_reward"] = sum(episode_rewards) / len(episode_rewards)
            stats["mean_episode_steps"] = sum(episode_steps) / len(episode_steps)
            stats["mean_episode_score"] = sum(episode_scores) / len(episode_scores)
            if episode_path_lengths:
                stats["episode_path_length_sum"] = float(sum(episode_path_lengths))
                stats["episode_path_length_count"] = len(episode_path_lengths)
                stats["mean_episode_path_length"] = stats["episode_path_length_sum"] / stats["episode_path_length_count"]
            if episode_coverages:
                stats["episode_coverage_sum"] = float(sum(episode_coverages))
                stats["episode_coverage_count"] = len(episode_coverages)
                stats["mean_episode_coverage"] = stats["episode_coverage_sum"] / stats["episode_coverage_count"]

            # Success rate metrics
            if episode_terminated:
                num_terminated = sum(episode_terminated)
                num_truncated = len(episode_terminated) - num_terminated
                stats["num_terminated"] = num_terminated
                stats["num_truncated"] = num_truncated
        else:
            stats["num_terminated"] = 0
            stats["num_truncated"] = 0

        if block_reward_count > 0:
            stats["reward_recall_node"] = block_recall_node_sum / block_reward_count
            stats["reward_recall_edge"] = block_recall_edge_sum / block_reward_count
            stats["reward_collision_rate"] = block_collision_rate_sum / block_reward_count
            stats["reward_time_penalty"] = block_time_penalty_sum / block_reward_count
            stats["reward_total_reward"] = block_total_reward_sum / block_reward_count

        return stats

    def _update_rollout_window(self, rollout_stats):
        entry = {
            "episodes_done": rollout_stats.get("episodes_done", 0),
            "episode_reward_sum": rollout_stats.get("episode_reward_sum", 0.0),
            "episode_steps_sum": rollout_stats.get("episode_steps_sum", 0.0),
            "episode_score_sum": rollout_stats.get("episode_score_sum", 0.0),
            "episode_path_length_sum": rollout_stats.get("episode_path_length_sum", 0.0),
            "episode_path_length_count": rollout_stats.get("episode_path_length_count", 0),
            "episode_coverage_sum": rollout_stats.get("episode_coverage_sum", 0.0),
            "episode_coverage_count": rollout_stats.get("episode_coverage_count", 0),
            "num_terminated": rollout_stats.get("num_terminated", 0),
            "num_truncated": rollout_stats.get("num_truncated", 0),
        }
        self.rollout_window.append(entry)
        # Track curriculum-facing window metrics (means over the latest rollout window)
        episodes_total = entry["episodes_done"]
        if episodes_total > 0:
            mean_reward = rollout_stats.get("mean_episode_reward", 0.0)
            mean_score = rollout_stats.get("mean_episode_score", 0.0)
            mean_steps = rollout_stats.get("mean_episode_steps", 0.0)
            success_rate = entry["num_terminated"] / max(1, entry["num_terminated"] + entry["num_truncated"])
            collision_rate = rollout_stats.get("reward_collision_rate", None)
            self._window_history.append(
                {
                    "mean_reward": mean_reward,
                    "mean_score": mean_score,
                    "mean_steps": mean_steps,
                    "success_rate": success_rate,
                    "collision_rate": collision_rate,
                }
            )

    def _get_rollout_window_stats(self):
        if not self.rollout_window:
            return None

        episodes_total = sum(entry["episodes_done"] for entry in self.rollout_window)
        if episodes_total == 0:
            return None

        reward_sum = sum(entry["episode_reward_sum"] for entry in self.rollout_window)
        steps_sum = sum(entry["episode_steps_sum"] for entry in self.rollout_window)
        score_sum = sum(entry["episode_score_sum"] for entry in self.rollout_window)
        num_terminated = sum(entry["num_terminated"] for entry in self.rollout_window)
        num_truncated = sum(entry["num_truncated"] for entry in self.rollout_window)
        path_length_sum = sum(entry["episode_path_length_sum"] for entry in self.rollout_window)
        path_length_count = sum(entry["episode_path_length_count"] for entry in self.rollout_window)
        coverage_sum = sum(entry["episode_coverage_sum"] for entry in self.rollout_window)
        coverage_count = sum(entry["episode_coverage_count"] for entry in self.rollout_window)

        return {
            "mean_episode_reward": reward_sum / episodes_total,
            "mean_episode_steps": steps_sum / episodes_total,
            "mean_episode_score": score_sum / episodes_total,
            "mean_episode_path_length": path_length_sum / max(1, path_length_count),
            "mean_episode_coverage": coverage_sum / max(1, coverage_count),
            "num_terminated": num_terminated,
            "num_truncated": num_truncated,
        }

    def _log_training_stats(self, update_stats, rollout_stats, update_idx):
        """Log training statistics to TensorBoard and console."""
        if not hasattr(self, "writer"):
            return

        log_step = update_idx if self.progress_unit == "blocks" else self.episode_count

        # Basic counters
        if self.progress_unit == "blocks":
            self.writer.add_scalar("Training/Episodes", self.episode_count, log_step)

        if "mean_episode_reward" in rollout_stats:
            self.writer.add_scalar("Block/Mean_Reward", rollout_stats["mean_episode_reward"], log_step)
            self.writer.add_scalar("Block/Mean_Steps", rollout_stats["mean_episode_steps"], log_step)
            self.writer.add_scalar("Block/Mean_Score", rollout_stats["mean_episode_score"], log_step)
            if "mean_episode_path_length" in rollout_stats:
                self.writer.add_scalar("Block/Mean_Path_Length", rollout_stats["mean_episode_path_length"], log_step)
            if "mean_episode_coverage" in rollout_stats:
                self.writer.add_scalar("Block/Mean_Exploration_Coverage", rollout_stats["mean_episode_coverage"], log_step)
            if rollout_stats["mean_episode_score"] > 0:
                self.writer.add_scalar(
                    "Block/Steps_for_score_1", rollout_stats["mean_episode_steps"] / rollout_stats["mean_episode_score"], log_step
                )
        num_terminated = rollout_stats.get("num_terminated", 0)
        num_truncated = rollout_stats.get("num_truncated", 0)
        self.writer.add_scalar("Block/Num_Terminated", num_terminated, log_step)
        self.writer.add_scalar("Block/Num_Truncated", num_truncated, log_step)
        self.writer.add_scalar("Block/Terminated_Truncated_Ratio", num_terminated / max(1, num_truncated), log_step)

        # Action type logging (per block)
        if "actions_total" in rollout_stats:
            self.writer.add_scalar("actions/total", rollout_stats["actions_total"], log_step)
        if "actions_pure_rotations" in rollout_stats:
            self.writer.add_scalar("actions/pure_rotations", rollout_stats["actions_pure_rotations"], log_step)
            self.writer.add_scalar("actions/movements", rollout_stats.get("actions_movements", 0), log_step)
            self.writer.add_scalar("actions/stops", rollout_stats.get("actions_stops", 0), log_step)
            self.writer.add_scalar("actions/idle", rollout_stats.get("actions_idle", 0), log_step)
            self.writer.add_scalar("actions/pure_rotation_rate", rollout_stats.get("actions_pure_rotation_rate", 0.0), log_step)
            self.writer.add_scalar(
                "actions/pure_rotation_to_movement_ratio", rollout_stats.get("actions_pure_rotation_to_movement_ratio", 0.0), log_step
            )
            self.writer.add_scalar("actions/stop_rate", rollout_stats.get("actions_stop_rate", 0.0), log_step)

        # Agent update stats (always log to TensorBoard)
        self.writer.add_scalar("Loss/policy_loss", update_stats.get("policy_loss", update_stats.get("loss", 0)), log_step)
        if "value_loss" in update_stats:
            self.writer.add_scalar("Loss/value_loss", update_stats.get("value_loss", 0), log_step)
        self.writer.add_scalar("Loss", update_stats.get("loss", 0), log_step)
        if "collision_loss" in update_stats:
            self.writer.add_scalar("Loss/collision_loss", update_stats.get("collision_loss", 0), log_step)
        if "collision_acc" in update_stats:
            self.writer.add_scalar("policy/collision_acc", update_stats.get("collision_acc", 0), log_step)
        self.writer.add_scalar("policy/entropy", update_stats.get("entropy", 0), log_step)

        # PPO-specific metrics
        if "clip_fraction" in update_stats:
            self.writer.add_scalar("policy/clip_fraction", update_stats.get("clip_fraction", 0), log_step)
        if "approx_kl" in update_stats:
            self.writer.add_scalar("policy/approx_kl", update_stats.get("approx_kl", 0), log_step)

        # REINFORCE-specific metrics
        if "ret_std" in update_stats:
            self.writer.add_scalar("policy/ret_std", update_stats.get("ret_std", 0), log_step)

        window_stats = self._get_rollout_window_stats()
        if window_stats:
            mean_score = window_stats["mean_episode_score"]
            mean_steps = window_stats["mean_episode_steps"]
            mean_reward = window_stats["mean_episode_reward"]

            self.writer.add_scalar("Rollout/Mean_Score", mean_score, log_step)
            self.writer.add_scalar("Rollout/Mean_Steps", mean_steps, log_step)
            self.writer.add_scalar("Rollout/Mean_Reward", mean_reward, log_step)
            self.writer.add_scalar("Rollout/Mean_Path_Length", window_stats.get("mean_episode_path_length", 0.0), log_step)
            self.writer.add_scalar("Rollout/Mean_Exploration_Coverage", window_stats.get("mean_episode_coverage", 0.0), log_step)
            self.writer.add_scalar("Rollout/Num_Terminated", window_stats["num_terminated"], log_step)
            self.writer.add_scalar("Rollout/Num_Truncated", window_stats["num_truncated"], log_step)
            if mean_score > 0:
                self.writer.add_scalar("Rollout/Steps_for_score_1", mean_steps / mean_score, log_step)
            self.writer.add_scalar(
                "Rollout/Terminated_Truncated_Ratio", window_stats["num_terminated"] / max(1, window_stats["num_truncated"]), log_step
            )

            # Track best scores
            if mean_score > self.max_score:
                self.max_score = mean_score
            if mean_score > self.best_mean_score:
                self.best_mean_score = mean_score

            # Console logging only every log_interval progress units
            if log_step % self.log_interval == 0:
                print()
                print(
                    f"{'Block' if self.progress_unit == 'blocks' else 'Ep'} {log_step:5d} | "
                    f"Episodes {self.episode_count:5d} | Update {update_idx:5d} | "
                    f"MA Score: {mean_score:5.2f} | Max: {self.max_score:5.2f} | "
                    f"MA Steps: {mean_steps:5.1f} | MA Reward: {mean_reward:6.2f} | "
                    f"Loss: {update_stats.get('policy_loss', update_stats.get('loss', 0)):.4f}"
                )

        # Environment-specific logging
        self.writer.add_scalar("env/rho", self.env_config["rho"], log_step)

        # Reward component logging
        if "reward_recall_node" in rollout_stats:
            self.writer.add_scalar("Reward/recall_node", rollout_stats["reward_recall_node"], log_step)
        if "reward_recall_edge" in rollout_stats:
            self.writer.add_scalar("Reward/recall_edge", rollout_stats["reward_recall_edge"], log_step)
        if "reward_collision_rate" in rollout_stats:
            self.writer.add_scalar("Reward/collision_rate", rollout_stats["reward_collision_rate"], log_step)
        if "reward_time_penalty" in rollout_stats:
            self.writer.add_scalar("Reward/time_penalty", rollout_stats["reward_time_penalty"], log_step)
        if "reward_total_reward" in rollout_stats:
            self.writer.add_scalar("Reward/total_reward", rollout_stats["reward_total_reward"], log_step)

        # Curriculum logging
        if self.curriculum_enabled or not self.env_config.get("use_legacy_actions", False):
            self.writer.add_scalar("curriculum/stage", self.current_curriculum_stage, log_step)
            self.writer.add_scalar("curriculum/entropy_coef", self.agent.entropy_coef, log_step)
            # Get number of valid actions for current stage
            try:
                valid_dir_indices = self.vec_env.get_attr("valid_direction_indices", indices=[0])[0]
                valid_len_indices = self.vec_env.get_attr("valid_length_indices", indices=[0])[0]
                num_valid_actions = len(valid_dir_indices) * len(valid_len_indices)
                self.writer.add_scalar("curriculum/num_valid_actions", num_valid_actions, log_step)
            except (AttributeError, IndexError):
                pass  # Environment doesn't have these attributes

        # Action distribution logging (multi-head only)
        if not self.use_legacy_actions and len(self.actions_since_last_log) > 0:
            # Compute action distribution from recent actions
            recent_actions = list(self.actions_since_last_log)[-1000:]

            # Direction distribution
            dir_counts = {}
            for action in recent_actions:
                dir_idx = action[0]
                dir_counts[dir_idx] = dir_counts.get(dir_idx, 0) + 1

            # Log top-5 most used directions
            if dir_counts:
                sorted_dirs = sorted(dir_counts.items(), key=lambda x: x[1], reverse=True)
                for rank, (dir_idx, count) in enumerate(sorted_dirs[:5]):
                    self.writer.add_scalar(f"actions/top_direction_{rank+1}_idx", dir_idx, log_step)
                    self.writer.add_scalar(f"actions/top_direction_{rank+1}_pct", 100 * count / len(recent_actions), log_step)

            # Length distribution
            len_counts = {}
            for action in recent_actions:
                len_idx = action[1]
                len_counts[len_idx] = len_counts.get(len_idx, 0) + 1

            # Log top-5 most used lengths
            if len_counts:
                sorted_lens = sorted(len_counts.items(), key=lambda x: x[1], reverse=True)
                for rank, (len_idx, count) in enumerate(sorted_lens[:5]):
                    self.writer.add_scalar(f"actions/top_length_{rank+1}_idx", len_idx, log_step)
                    self.writer.add_scalar(f"actions/top_length_{rank+1}_pct", 100 * count / len(recent_actions), log_step)

            # Action diversity: unique actions used
            unique_actions = len(set(recent_actions))
            self.writer.add_scalar("actions/unique_actions_used", unique_actions, log_step)
            self.writer.add_scalar("actions/unique_directions_used", len(dir_counts), log_step)
            self.writer.add_scalar("actions/unique_lengths_used", len(len_counts), log_step)

            # New action coverage tracking (only during entropy boost phase)
            if self.new_actions_at_stage_transition and self.entropy_boost_counter > 0:
                # Check how many of the newly unlocked actions have been tried
                tried_new_actions = set(self.action_counts_combined.keys()) & self.new_actions_at_stage_transition
                coverage_pct = (
                    100 * len(tried_new_actions) / len(self.new_actions_at_stage_transition) if self.new_actions_at_stage_transition else 0
                )

                self.writer.add_scalar("curriculum/new_action_coverage_pct", coverage_pct, log_step)
                self.writer.add_scalar("curriculum/new_actions_tried", len(tried_new_actions), log_step)
                self.writer.add_scalar("curriculum/new_actions_total", len(self.new_actions_at_stage_transition), log_step)

            # Action counts reset (prevent unbounded dict growth)
            # Reset every N blocks OR when dict grows too large (whichever comes first)
            if log_step % self._action_count_reset_interval == 0 or len(self.action_counts_combined) > 1000:
                # Reset all count dicts to prevent memory leak
                # For multi-head: max 504 possible actions, so 1000 is safe upper bound
                self.action_counts_dir = {}
                self.action_counts_len = {}
                self.action_counts_combined = {}

            # actions_since_last_log is now a deque with maxlen=2000, auto-managed

    def _save_checkpoint(self, progress_step: int):
        """Save model checkpoint."""
        save_folder = Path(self.log_dir) / "checkpoints"
        save_folder.mkdir(exist_ok=True, parents=True)
        unit = "block" if self.progress_unit == "blocks" else "ep"
        file_name = f"checkpoint_{unit}_{progress_step}.pth"
        torch.save(self.agent.state_dict(), save_folder / file_name)
        print(f"[INFO] Checkpoint saved: {save_folder / file_name}")

    def _finalize_training(self, save_model):
        """Cleanup and save final results."""
        num_envs = self.vec_env.num_envs
        steps_per_env_per_update = int(self.agent_config.get("num_steps", 0))
        env_steps_total = self.update_count * num_envs * steps_per_env_per_update
        self.vec_env.close()

        # Save training summary
        summary = {
            "best_mean_score": self.best_mean_score,
            "max_score": self.max_score,
            "episodes": self.episode_count,
            "updates": self.update_count,
            "env_steps": env_steps_total,
            "progress_unit": self.progress_unit,
        }
        if self.progress_unit == "blocks":
            summary["blocks"] = self.update_count

        try:
            summary_path = os.path.join(self.log_dir, "train_summary.json")
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2)
            print(f"[INFO] Saved train summary to: {summary_path}")
        except Exception as e:
            print(f"[WARNING] Could not save train summary: {e}")

        self.writer.close()
        print("\n[INFO] Training completed successfully!")
        print(f"[INFO] Best Mean Score: {self.best_mean_score:.2f}")
        print(f"[INFO] Max Score: {self.max_score:.2f}")

        # Save final model
        if save_model:
            save_folder = Path(self.log_dir) / "checkpoints"
            save_folder.mkdir(exist_ok=True, parents=True)
            file_name = "final_model.pth"
            torch.save(self.agent.state_dict(), save_folder / file_name)
            print(f"[INFO] Final model saved: {save_folder / file_name}")
