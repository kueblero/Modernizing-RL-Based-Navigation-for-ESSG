"""
Unified Optimization Runner for Hyperparameter Optimization

Extends UnifiedTrainRunner with Optuna integration for hyperparameter optimization.
Supports early stopping via MedianPruner and custom objective metrics.
"""

import time

import optuna
from tqdm import tqdm

from RL_training.runner.unified_train_runner import UnifiedTrainRunner


class UnifiedOptimRunner(UnifiedTrainRunner):
    """
    Optimization runner that extends UnifiedTrainRunner with Optuna integration.

    Key features:
    - Inherits all training logic from UnifiedTrainRunner
    - Reports intermediate objectives to Optuna for pruning
    - Computes custom objective: mean score for terminated episodes
    - Supports trial resumption via Optuna's built-in mechanisms
    """

    def __init__(self, vec_env, agent, device, config, trial, scenario_name):
        """
        Args:
            vec_env: VecEnv instance with parallel environments
            agent: Agent instance (REINFORCE or PPO)
            device: torch device
            config: Full configuration dict
            trial: Optuna trial object for reporting/pruning
            scenario_name: Name of the scenario being optimized
        """
        super().__init__(vec_env, agent, device, config, log_dir="no_logging")

        self.trial = trial
        self.scenario_name = scenario_name

        # Track all episode completions for objective calculation
        self.episode_history = []  # [(score, steps, terminated), ...]

        # Reporting interval (report to Optuna every N blocks)
        self.report_interval = 1  # Report every block for responsive pruning

        # Disable checkpoint saving for optimization
        self.save_interval = float("inf")  # Never save checkpoints during optimization

        # Disable detailed console output during optimization
        self.log_buffer_size = 20  # Smaller buffer for faster aggregation

        # Objective uses a block window; ensure we keep the last 50 blocks
        if hasattr(self, "rollout_window") and self.rollout_window.maxlen is not None:
            self.rollout_window = self.rollout_window.__class__(self.rollout_window, maxlen=50)

    def run(self, save_model=False):
        """
        Main optimization loop with Optuna integration.

        Args:
            save_model: Ignored (always False during optimization)

        Returns:
            float: Final objective value

        Raises:
            optuna.TrialPruned: If trial should be pruned
        """
        print(f"\n{'='*70}")
        print(f"[OPTUNA] Trial {self.trial.number} - {self.scenario_name}")
        print(f"[OPTUNA] Parameters:")
        for param, value in self.trial.params.items():
            print(f"  {param}: {value:.6f}")
        print(f"{'='*70}\n")

        print(f"[INFO] Starting Optimization Training")
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
        print(f"[INFO] Parallel Envs: {self.vec_env.num_envs}")
        print(f"[INFO] RHO: {self.env_config['rho']}")

        if self.curriculum_enabled:
            print(f"[INFO] Curriculum Learning: Enabled")
            print(f"  - Stage 1 (16 actions):  0-{self.stage1_blocks} blocks")
            print(f"  - Stage 2 (48 actions):  {self.stage1_blocks}-{self.stage2_blocks} blocks")
            print(f"  - Stage 3 (108 actions): {self.stage2_blocks}-{self.stage3_blocks} blocks")
            print(f"  - Stage 4 (208 actions): {self.stage3_blocks}-{self.stage4_blocks} blocks")
            print(f"  - Stage 5 (340 actions): {self.stage4_blocks}-{self.stage5_blocks} blocks")
            print(f"  - Stage 6 (504 actions): {self.stage5_blocks}+ blocks")

        print(f"{'='*70}\n")

        # Create buffer for the agent
        from components.utils.vec_rollout_buffer import VecEnvRolloutBuffer

        n_steps_per_env = self.agent_config["num_steps"]
        buffer = VecEnvRolloutBuffer(n_envs=self.vec_env.num_envs, n_steps_per_env=n_steps_per_env)

        update_idx = 0
        progress_total = self.n_blocks if self.progress_unit == "blocks" else self.n_episodes
        progress_desc = "Training Blocks" if self.progress_unit == "blocks" else "Training Episodes"
        pbar = tqdm(total=progress_total, desc=progress_desc, ncols=160)

        try:
            while (update_idx < self.n_blocks) if self.progress_unit == "blocks" else (self.episode_count < self.n_episodes):
                update_idx += 1
                self.update_count = update_idx

                # Update curriculum stage if needed
                self._maybe_update_curriculum()

                # Collect rollouts from all parallel environments
                block_start = time.time()
                rollout_stats = self._collect_rollouts(buffer, n_steps=n_steps_per_env)
                self._update_rollout_window(rollout_stats)

                # Track episodes for objective calculation
                # Note: Individual episodes are tracked in _collect_rollouts override
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

                # Update progress bar
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

                # Log stats (no-op when logging disabled)
                self._log_training_stats(update_stats, rollout_stats, update_idx)

                # Report to Optuna and check for pruning
                if update_idx % self.report_interval == 0:
                    current_objective = self._calculate_objective()

                    # Report intermediate value
                    self.trial.report(current_objective, update_idx)

                    # Check if trial should be pruned
                    if self.trial.should_prune():
                        pbar.close()

                        # Calculate metrics at pruning time
                        window_stats = self._get_rollout_window_stats()
                        if self.episode_history:
                            success_episodes = [1 for _, _, term in self.episode_history if term]
                            success_rate = len(success_episodes) / max(1, len(self.episode_history))
                            truncated_episodes = [1 for _, _, term in self.episode_history if not term]
                            truncation_rate = len(truncated_episodes) / max(1, len(self.episode_history))
                        else:
                            success_rate = 0.0
                            truncation_rate = 0.0

                        # Get collision rate
                        if hasattr(self, "reward_components") and len(self.reward_components["collision_rate"]) > 0:
                            collision_rates = list(self.reward_components["collision_rate"])
                            mean_collision_rate = sum(collision_rates) / len(collision_rates)
                        else:
                            mean_collision_rate = 0.0

                        # Print detailed pruning info
                        print(f"\n{'='*80}")
                        print(f"[OPTUNA] Trial {self.trial.number} PRUNED at block {update_idx}")
                        print(f"{'='*80}")
                        print(f"[OBJECTIVE AT PRUNING]")
                        print(f"  Current Objective: {current_objective:.3f}")
                        print(f"\n[METRICS AT PRUNING]")
                        if window_stats:
                            print(f"  Mean Score (Node Recall): {window_stats['mean_episode_score']:.3f}")
                            print(f"  Mean Steps: {window_stats['mean_episode_steps']:.1f}")
                        print(f"  Success Rate (terminated): {success_rate:.2%}")
                        print(f"  Truncation Rate: {truncation_rate:.2%}")
                        print(f"  Mean Collision Rate: {mean_collision_rate:.2%}")
                        print(f"  Total Episodes: {self.episode_count}")
                        print(f"\n[PARAMETERS]")
                        for param, value in self.trial.params.items():
                            if isinstance(value, float):
                                print(f"  {param}: {value:.6f}")
                            else:
                                print(f"  {param}: {value}")
                        print(f"{'='*80}\n")

                        raise optuna.TrialPruned()

            pbar.close()

            # Calculate final objective
            final_objective = self._calculate_objective()

            # Store metrics in trial user attributes
            window_stats = self._get_rollout_window_stats()
            if window_stats:
                self.trial.set_user_attr("mean_score", window_stats["mean_episode_score"])
                self.trial.set_user_attr("mean_steps", window_stats["mean_episode_steps"])

            self.trial.set_user_attr("total_episodes", self.episode_count)

            # Calculate success rate (terminated episodes)
            success_episodes = [1 for _, _, term in self.episode_history if term]
            success_rate = len(success_episodes) / max(1, len(self.episode_history))
            self.trial.set_user_attr("success_rate", success_rate)

            # Store collision rate
            if hasattr(self, "reward_components") and len(self.reward_components["collision_rate"]) > 0:
                collision_rates = list(self.reward_components["collision_rate"])
                mean_collision_rate = sum(collision_rates) / len(collision_rates)
                self.trial.set_user_attr("mean_collision_rate", mean_collision_rate)

            # Store truncation rate
            truncated_episodes = [1 for _, _, term in self.episode_history if not term]
            truncation_rate = len(truncated_episodes) / max(1, len(self.episode_history))
            self.trial.set_user_attr("truncation_rate", truncation_rate)

            # Cleanup
            self.vec_env.close()

            print(f"\n{'='*80}")
            print(f"[OPTUNA] Trial {self.trial.number} completed")
            print(f"{'='*80}")
            print(f"[OBJECTIVE]")
            print(f"  Final Objective: {final_objective:.3f}")
            print(f"\n[METRICS]")
            if window_stats:
                print(f"  Mean Score (Node Recall): {window_stats['mean_episode_score']:.3f}")
                print(f"  Mean Steps: {window_stats['mean_episode_steps']:.1f}")
            print(f"  Success Rate (terminated): {success_rate:.2%}")
            print(f"  Truncation Rate: {truncation_rate:.2%}")
            if hasattr(self, "reward_components") and len(self.reward_components["collision_rate"]) > 0:
                collision_rates = list(self.reward_components["collision_rate"])
                mean_collision_rate = sum(collision_rates) / len(collision_rates)
                print(f"  Mean Collision Rate: {mean_collision_rate:.2%}")
            print(f"  Total Episodes: {self.episode_count}")
            print(f"\n[PARAMETERS]")
            for param, value in self.trial.params.items():
                if isinstance(value, float):
                    print(f"  {param}: {value:.6f}")
                else:
                    print(f"  {param}: {value}")
            print(f"{'='*80}\n")

            return final_objective

        except optuna.TrialPruned:
            # Cleanup on pruning
            self.vec_env.close()
            raise
        except Exception as e:
            # Cleanup on error
            pbar.close()
            self.vec_env.close()
            print(f"\n[ERROR] Trial {self.trial.number} failed: {e}")
            raise

    def _calculate_objective(self):
        """
        Calculate objective value based on episode history.

        Objective: Maximize mean score and apply light secondary penalties for
        collisions, truncations, and overly long episodes, using the most recent
        rollout window (last ~50 blocks).

        Returns:
            float: Objective value
        """
        window_stats = self._get_rollout_window_stats()
        if not window_stats:
            return -100.0  # No episodes yet

        mean_score = window_stats["mean_episode_score"]
        mean_steps = window_stats["mean_episode_steps"]
        total_eps = window_stats["num_terminated"] + window_stats["num_truncated"]
        truncation_rate = window_stats["num_truncated"] / max(1, total_eps)

        # Collision rate from the same window if available, else fallback to recent reward components
        mean_collision_rate = window_stats.get("mean_collision_rate", None)
        if mean_collision_rate is None:
            if hasattr(self, "reward_components") and len(self.reward_components["collision_rate"]) > 0:
                collision_rates = list(self.reward_components["collision_rate"])
                recent_collision_rates = collision_rates[-50:] if len(collision_rates) > 50 else collision_rates
                mean_collision_rate = sum(recent_collision_rates) / len(recent_collision_rates)
            else:
                mean_collision_rate = 0.0

        # Light penalties to keep collisions/truncations low and discourage very long episodes
        collision_penalty = 0.20 * mean_collision_rate
        truncation_penalty = 0.10 * truncation_rate
        step_penalty = 0.01 * max(0.0, mean_steps - 25.0)  # small cost after 25 mean steps

        objective = mean_score - collision_penalty - truncation_penalty - step_penalty

        return max(-100.0, objective)

    def _collect_rollouts(self, buffer, n_steps):
        """
        Override to track individual episodes for objective calculation.

        We need to intercept episode completions to track:
        - Individual episode scores
        - Individual episode steps
        - Whether episode terminated vs truncated

        This data is used to compute the objective metric.
        """
        import torch

        n_envs = self.vec_env.num_envs
        steps_collected = 0
        episodes_done = 0
        episode_rewards = []
        episode_steps = []
        episode_scores = []
        episode_terminated = []
        episode_steps_success = []
        episode_steps_truncated = []
        block_recall_node_sum = 0.0
        block_recall_edge_sum = 0.0
        block_collision_rate_sum = 0.0
        block_time_penalty_sum = 0.0
        block_total_reward_sum = 0.0
        block_reward_count = 0
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
                    if len(self._hiddens_list[env_idx]) == 4:
                        self.agent.lssg_hidden, self.agent.gssg_hidden, self.agent.policy_hidden, self.agent.stagnation_state = (
                            self._hiddens_list[env_idx]
                        )
                    else:
                        self.agent.lssg_hidden, self.agent.gssg_hidden, self.agent.policy_hidden = self._hiddens_list[env_idx]
                        self.agent.stagnation_state = None
                    self.agent.last_action = self._last_actions[env_idx]

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
            if not self.use_legacy_actions and hasattr(self, "action_counts_dir"):
                for action in actions:
                    if self.use_single_head_large:
                        # Single-head-large: action is tuple (flat_idx, -1), decode to (dir_idx, len_idx)
                        flat_idx = action[0] if isinstance(action, (tuple, list)) else action
                        dir_idx = flat_idx // 21
                        len_idx = flat_idx % 21
                    else:
                        # Multi-head: action is (dir_idx, len_idx) or (dir_idx, len_idx, stop_flag)
                        dir_idx, len_idx = action[0], action[1]
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
            buffer.add(
                states=states,
                actions=actions,
                rewards=rewards,
                dones=dones_combined,
                hiddens=self._hiddens_list,
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

                    # Track terminated vs truncated
                    is_terminated = dones[env_idx] and not truncated[env_idx]
                    episode_terminated.append(is_terminated)

                    # OPTIMIZATION TRACKING: Store individual episode data
                    self.episode_history.append((score, self._episode_steps[env_idx], is_terminated))

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

                    if hasattr(self, "reward_components"):
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
                        act = (act[0], act[1])
                    self._last_actions[env_idx] = torch.tensor(act, dtype=torch.long, device=self.device)
                    self._agent_positions[env_idx] = infos[env_idx].get("agent_pos", None)
                    self._obs_list[env_idx] = next_obs_list[env_idx]

            # Reset finished environments
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
