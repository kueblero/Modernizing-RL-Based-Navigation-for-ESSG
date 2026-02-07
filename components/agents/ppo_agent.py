import math

import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from components.agents.abstract_agent import AbstractAgent
from components.utils.vec_rollout_buffer import VecEnvRolloutBuffer


class PPOAgent(AbstractAgent):
    """
    Proximal Policy Optimization (PPO) agent with clipped surrogate objective.
    More stable and sample-efficient than A2C.

    Key features:
    - Clipped surrogate objective for policy updates
    - Multiple epochs over same batch (K epochs)
    - GAE (Generalized Advantage Estimation) for better advantage estimates
    - Value function clipping (optional)
    - Entropy bonus for exploration
    """

    def __init__(
        self, env, navigation_config, agent_config, device=None, mapping_path=None, use_mixed_precision=False, use_torch_compile=False
    ):
        super().__init__(env, navigation_config, agent_config, device, mapping_path)

        # Action space mode (inherited from parent, but store for convenience)
        self.action_space_mode = getattr(env, "action_space_mode", "multi_head")

        # PPO-specific hyperparameters
        self.clip_epsilon = agent_config.get("clip_epsilon", 0.2)
        self.value_coef = agent_config.get("value_coef", 0.5)
        self.entropy_coef = agent_config.get("entropy_coef", 0.01)
        self.ppo_epochs = agent_config.get("ppo_epochs", 1)
        self.max_grad_norm = agent_config.get("max_grad_norm", 0.5)
        self.trajectory_chunk_size = agent_config.get("trajectory_chunk_size", 0)
        self.auto_trajectory_chunk = agent_config.get("auto_trajectory_chunk", False)
        self.trajectory_chunk_min = int(agent_config.get("trajectory_chunk_min", 100))
        self.trajectory_chunk_max = int(agent_config.get("trajectory_chunk_max", 1000))
        self.trajectory_chunk_backoff = float(agent_config.get("trajectory_chunk_backoff", 0.9))
        self._trajectory_chunk_calibrated = False

        # GAE parameters
        self.use_gae = agent_config.get("use_gae", True)
        self.gae_lambda = agent_config.get("gae_lambda", 0.95)

        # Value clipping
        self.clip_value = agent_config.get("clip_value", True)
        self.clip_value_epsilon = agent_config.get("clip_value_epsilon", 0.2)

        # Anti-collapse: Minimum entropy target and adaptive entropy coefficient
        # For 3 heads (dir=24, len=21, stop=2), max entropy ≈ log(24)+log(21)+log(2) ≈ 6.5
        self.target_entropy = agent_config.get("target_entropy", 3.0)  # ~50% of max
        self.entropy_coef_min = agent_config.get("entropy_coef_min", 0.01)
        self.entropy_coef_max = agent_config.get("entropy_coef_max", 0.5)
        self.adaptive_entropy = agent_config.get("adaptive_entropy", True)

        # KL early stopping to prevent catastrophic updates
        self.target_kl = agent_config.get("target_kl", 0.03)  # Stop epoch if KL > target

        # Cache curriculum masks for efficiency (invalidated on curriculum stage change)
        self._curriculum_masks_cache = None
        self._curriculum_stage_cache = getattr(self.env, "curriculum_stage", None)

        # Mixed Precision Training with GradScaler
        self.use_mixed_precision = use_mixed_precision and torch.cuda.is_available()
        if self.use_mixed_precision:
            # Use new API: torch.amp instead of torch.cuda.amp
            if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
                self.scaler = torch.amp.GradScaler("cuda")
            else:
                # Fallback for older PyTorch versions
                self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None

        # Selective torch.compile for hot paths only (PyTorch 2.0+)
        # Strategy: Compile small, frequently called modules instead of whole model
        # This avoids OOM from compiling huge models while still getting speedup
        self.use_torch_compile = use_torch_compile
        if self.use_torch_compile and hasattr(torch, "compile"):
            compiled_count = 0

            # 1. Compile Depth Encoder (small ~50k params, trainable, called for every observation)
            if hasattr(self.encoder, "depth_encoder") and not self.encoder.freeze_depth_backbone:
                try:
                    self.encoder.depth_encoder = torch.compile(self.encoder.depth_encoder, mode="default")
                    compiled_count += 1
                except Exception as e:
                    print(f"[WARNING] Depth encoder compile failed: {e}")

            # 2. Graph Encoder: SKIP compilation (variable input shapes cause recompiles)
            # Graph neural networks process scenes with varying numbers of objects (128, 132, 145, etc.)
            # torch.compile would recompile for each shape → overhead instead of speedup
            # Verified via profiling: graph caching is sufficient optimization

            # 3. Compile LSSG/GSSG LSTM Encoders (medium size, called for every observation)
            if hasattr(self.encoder, "lssg_encoder"):
                try:
                    self.encoder.lssg_encoder = torch.compile(self.encoder.lssg_encoder, mode="default")
                    compiled_count += 1
                except Exception as e:
                    print(f"[WARNING] LSSG encoder compile failed: {e}")

            if hasattr(self.encoder, "gssg_encoder"):
                try:
                    self.encoder.gssg_encoder = torch.compile(self.encoder.gssg_encoder, mode="default")
                    compiled_count += 1
                except Exception as e:
                    print(f"[WARNING] GSSG encoder compile failed: {e}")

            if compiled_count > 0:
                print(f"[INFO] Selectively compiled {compiled_count} hot-path modules (avoids OOM)")
            else:
                print("[INFO] torch.compile available but no modules compiled")
                self.use_torch_compile = False

    def compute_gae(self, rewards, values, dones, last_value=None):
        """
        Compute Generalized Advantage Estimation (GAE) - VECTORIZED.

        This is a more sophisticated advantage estimation than simple TD-error,
        balancing bias vs. variance through the lambda parameter.

        OPTIMIZED: Vectorized TD-error computation, only the GAE accumulation loop remains
        (which has sequential dependencies and can't be fully vectorized).

        Formula: A_t = sum_{l=0}^{inf} (gamma * lambda)^l * delta_{t+l}
        where delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)

        Args:
            rewards: Tensor of shape [T] containing rewards
            values: Tensor of shape [T] containing value estimates
            dones: Tensor of shape [T] (1 if episode ended, 0 otherwise)
            last_value: Optional value estimate for bootstrapping at rollout end.
                       If None and last step is not done, uses 0 (underestimates).

        Returns:
            advantages: Tensor of shape [T] with computed advantages
            returns: Tensor of shape [T] with computed returns (advantages + values)
        """
        T = len(rewards)
        if T == 0:
            return torch.zeros_like(rewards), torch.zeros_like(rewards)

        # VECTORIZED: Compute next_values for all timesteps at once
        next_values = torch.zeros_like(values)
        next_values[:-1] = values[1:]  # Shift values by 1

        # Handle last timestep bootstrap
        if last_value is not None and dones[-1] < 0.5:
            next_values[-1] = last_value
        else:
            next_values[-1] = 0.0

        # VECTORIZED: Compute all TD errors at once
        # delta_t = r_t + gamma * V(s_{t+1}) * (1-done_t) - V(s_t)
        deltas = rewards + self.gamma * next_values * (1.0 - dones) - values

        # GAE accumulation (sequential dependency - must use loop)
        # A_t = delta_t + (gamma * lambda) * (1-done_t) * A_{t+1}
        advantages = torch.zeros_like(rewards)
        gae = 0.0
        discount = self.gamma * self.gae_lambda

        for t in reversed(range(T)):
            gae = deltas[t] + discount * (1.0 - dones[t]) * gae
            advantages[t] = gae

        # Returns are advantages + values (for value function training)
        returns = advantages + values

        return advantages, returns

    def _get_curriculum_masks(self):
        """
        Get curriculum-based action masks - VECTORIZED + CACHED.
        Returns:
            - Multi-head: (dir_mask, len_mask, stop_mask)
            - Single-head-large: (flat_mask, None, None)
            - Legacy: (None, None, None)

        OPTIMIZED: Masks are cached and only recomputed on curriculum stage change.
        """
        if self.use_legacy_actions:
            return None, None, None

        if self.action_space_mode == "single_head_large":
            # Single-head-large: flat action mask
            if not hasattr(self.env, "valid_flat_action_indices"):
                return None, None, None

            curr_stage = getattr(self.env, "curriculum_stage", None)
            if self._curriculum_stage_cache != curr_stage:
                self._curriculum_masks_cache = None
                self._curriculum_stage_cache = curr_stage

            # Return cached masks if available
            if self._curriculum_masks_cache is not None:
                return self._curriculum_masks_cache

            # Create flat action mask (504 dims)
            flat_mask = torch.zeros(504, dtype=torch.bool, device=self.device)
            flat_indices = torch.tensor(self.env.valid_flat_action_indices, dtype=torch.long, device=self.device)
            flat_mask[flat_indices] = True

            # Cache for future calls
            self._curriculum_masks_cache = (flat_mask, None, None)
            return flat_mask, None, None
        else:
            # Multi-head: direction, length, stop masks
            if not hasattr(self.env, "valid_direction_indices") or not hasattr(self.env, "valid_length_indices"):
                return None, None, None

            curr_stage = getattr(self.env, "curriculum_stage", None)
            if self._curriculum_stage_cache != curr_stage:
                self._curriculum_masks_cache = None
                self._curriculum_stage_cache = curr_stage

            # Return cached masks if available
            if self._curriculum_masks_cache is not None:
                return self._curriculum_masks_cache

            # VECTORIZED: Create masks using advanced indexing (no Python loops)
            dir_mask = torch.zeros(24, dtype=torch.bool, device=self.device)  # 24 directions
            len_mask = torch.zeros(21, dtype=torch.bool, device=self.device)  # 21 lengths
            stop_mask = torch.ones(2, dtype=torch.bool, device=self.device)  # STOP always allowed

            # VECTORIZED: Set valid indices to True in one operation
            dir_indices = torch.tensor(self.env.valid_direction_indices, dtype=torch.long, device=self.device)
            len_indices = torch.tensor(self.env.valid_length_indices, dtype=torch.long, device=self.device)

            dir_mask[dir_indices] = True
            len_mask[len_indices] = True

            # Cache for future calls
            self._curriculum_masks_cache = (dir_mask, len_mask, stop_mask)

            return dir_mask, len_mask, stop_mask

    def _apply_action_mask(self, logits, mask):
        """Apply action mask by setting invalid action logits to -inf."""
        if mask is None:
            return logits
        # Expand mask to match logits shape
        while mask.dim() < logits.dim():
            mask = mask.unsqueeze(0)
        mask = mask.expand_as(logits)
        return logits.masked_fill(~mask, float("-inf"))

    def _compute_masked_entropy(self, logits, mask):
        """
        Compute entropy only over valid (masked) actions.
        This gives accurate exploration signal for curriculum learning.
        """
        if mask is None:
            dist = Categorical(logits=logits)
            return dist.entropy().mean()

        # Apply mask and compute entropy only over valid actions
        masked_logits = self._apply_action_mask(logits, mask)
        dist = Categorical(logits=masked_logits)
        return dist.entropy().mean()

    def update(self, obs=None):
        """
        PPO update with multiple epochs over the same batch.

        - Action masking applied consistently for curriculum learning
        - Entropy computed only over valid actions
        - Bootstrap value for GAE at rollout end

        Args:
            obs: Optional pre-collected batch (e.g., from VecEnvRolloutBuffer).
                 If None, falls back to the internal rollout buffer.

        Returns:
            dict with loss metrics: loss, policy_loss, value_loss, entropy, clip_fraction, approx_kl, collision_loss
        """
        # Ensure training mode for backward through RNNs
        self.train()

        if obs is None:
            try:
                raw_batch = self.rollout_buffers.get(self.gamma, getattr(self, "gae_lambda", None))
            except TypeError:
                raw_batch = self.rollout_buffers.get(self.gamma)
        else:
            raw_batch = obs

        batch = self._prepare_batch_for_update(raw_batch)

        # Get curriculum masks for action masking (None if legacy or no curriculum)
        dir_mask, len_mask, stop_mask = self._get_curriculum_masks()

        # Store old values for PPO ratio calculation
        with torch.no_grad():
            if self.use_mixed_precision:
                autocast_context = (
                    torch.amp.autocast("cuda") if hasattr(torch, "amp") and hasattr(torch.amp, "autocast") else torch.cuda.amp.autocast()
                )
                with autocast_context:
                    logits_or_dict_old, values_old = self.forward_update(batch)
            else:
                logits_or_dict_old, values_old = self.forward_update(batch)

            # Actions
            if not isinstance(batch["actions"], torch.Tensor):
                actions = torch.tensor(batch["actions"], device=self.device)
            else:
                actions = batch["actions"].to(self.device)

            # Initialize actions_dir and actions_len (used for multi-head only)
            actions_dir = None
            actions_len = None

            if self.use_legacy_actions or self.action_space_mode == "single_head_large":
                # Legacy: extract action logits from dict (now includes collision head)
                if isinstance(logits_or_dict_old, dict):
                    logits_old = logits_or_dict_old["action"].view(-1, logits_or_dict_old["action"].size(-1))
                else:
                    # Backward compatibility: old checkpoints return tensor directly
                    logits_old = logits_or_dict_old.view(-1, logits_or_dict_old.size(-1))
                dist_old = Categorical(logits=logits_old)

                if actions.ndim > 1:
                    actions = actions.view(-1)

                old_log_probs = dist_old.log_prob(actions)
            else:
                # Multi-head: dictionary with direction and length logits
                logits_dict_old = logits_or_dict_old
                logits_dir_old = logits_dict_old["direction"].view(-1, logits_dict_old["direction"].size(-1))
                logits_len_old = logits_dict_old["length"].view(-1, logits_dict_old["length"].size(-1))
                logits_stop_old = logits_dict_old["stop"].view(-1, logits_dict_old["stop"].size(-1))

                # Apply action masking to old logits for consistent log_probs
                logits_dir_old_masked = self._apply_action_mask(logits_dir_old, dir_mask)
                logits_len_old_masked = self._apply_action_mask(logits_len_old, len_mask)
                logits_stop_old_masked = self._apply_action_mask(logits_stop_old, stop_mask)

                dist_dir_old = Categorical(logits=logits_dir_old_masked)
                dist_len_old = Categorical(logits=logits_len_old_masked)
                dist_stop_old = Categorical(logits=logits_stop_old_masked)

                if actions.ndim == 3:
                    actions = actions.view(-1, actions.size(-1))
                elif actions.ndim == 1:
                    # Handle 1D actions - this can occur with buffer inconsistencies
                    # Reshape to [batch_size, 1] if single values, or raise error
                    raise ValueError(
                        f"Expected multi-head actions to be 2D or 3D tensor, but got 1D tensor with shape {actions.shape}. "
                        f"This likely means the buffer contains legacy-format actions while agent is configured for multi-head actions. "
                        f"Check that use_legacy_actions setting matches the data in the buffer."
                    )

                actions_dir = actions[:, 0]
                actions_len = actions[:, 1]
                if actions.size(-1) > 2:
                    actions_stop = actions[:, 2]
                else:
                    # If stop flag missing (older buffers), treat as "continue"
                    actions_stop = torch.zeros_like(actions_dir)

                old_log_probs_dir = dist_dir_old.log_prob(actions_dir)
                old_log_probs_len = dist_len_old.log_prob(actions_len)
                old_log_probs_stop = dist_stop_old.log_prob(actions_stop)
                old_log_probs = old_log_probs_dir + old_log_probs_len + old_log_probs_stop

            values_old = values_old.detach()
            old_log_probs = old_log_probs.float()
            values_old = values_old.float()

        # Get rewards and dones for GAE
        if not isinstance(batch["rewards"], torch.Tensor):
            rewards = torch.tensor(batch["rewards"], device=self.device).view(-1)
        else:
            rewards = batch["rewards"].to(self.device).view(-1)

        if not isinstance(batch["dones"], torch.Tensor):
            dones = torch.tensor(batch["dones"], device=self.device, dtype=torch.float32).view(-1)
        else:
            dones = batch["dones"].to(self.device).float().view(-1)

        trajectory_info = batch.get("trajectory_info") if self.use_trajectory_info else None

        # Compute advantages and returns
        if self.use_gae:
            if trajectory_info:
                # IMPORTANT: Compute GAE per trajectory to avoid leakage across episode/env boundaries.
                # For truncated (in-progress) trajectories at rollout cut, we don't have V(s_{T}) for bootstrapping,
                # so we conservatively use 0.0 (slight underestimation, but stable and boundary-correct).
                advantages = torch.zeros_like(rewards)
                returns = torch.zeros_like(rewards)

                for traj in trajectory_info:
                    start_idx = int(traj["start_idx"])
                    length = int(traj["length"])
                    end_idx = start_idx + length

                    r = rewards[start_idx:end_idx]
                    v = values_old[start_idx:end_idx]
                    d = dones[start_idx:end_idx]

                    last_value = 0.0 if (len(d) > 0 and d[-1] < 0.5) else None
                    adv_t, ret_t = self.compute_gae(r, v, d, last_value=last_value)
                    advantages[start_idx:end_idx] = adv_t
                    returns[start_idx:end_idx] = ret_t
            else:
                # Fallback: single contiguous trajectory (no trajectory metadata)
                last_value = None
                if len(dones) > 0 and dones[-1] < 0.5:  # Last step is not done
                    # Use last value estimate as bootstrap (best-effort, since V(s_{T}) is not available here)
                    last_value = values_old[-1].item() if values_old.numel() > 0 else 0.0
                advantages, returns = self.compute_gae(rewards, values_old, dones, last_value=last_value)
        else:
            if not isinstance(batch["returns"], torch.Tensor):
                returns = torch.tensor(batch["returns"], device=self.device).view(-1)
            else:
                returns = batch["returns"].to(self.device).view(-1)
            advantages = returns - values_old

        # Normalize advantages for stability
        if len(advantages) <= 1:
            advantages = advantages - advantages.mean()
        else:
            adv_mean = advantages.mean()
            adv_std = advantages.std(unbiased=False)

            if adv_std < 1e-6 or torch.isnan(adv_std) or torch.isinf(adv_std):
                advantages = advantages - adv_mean
            else:
                advantages = (advantages - adv_mean) / (adv_std + 1e-8)

        if trajectory_info and self.auto_trajectory_chunk and not self._trajectory_chunk_calibrated:
            self.trajectory_chunk_size = self._calibrate_trajectory_chunk_size(
                batch, trajectory_info, actions, old_log_probs, advantages, returns, values_old, dir_mask, len_mask, stop_mask
            )
            self._trajectory_chunk_calibrated = True

        if trajectory_info and isinstance(self.trajectory_chunk_size, int) and self.trajectory_chunk_size > 0:
            total_steps = int(actions.shape[0]) if actions.ndim > 0 else 0
            if total_steps <= 0:
                return {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

            chunk_batches = self._build_chunk_batches(batch, trajectory_info, self.trajectory_chunk_size)
            move_success = batch.get("move_success")
            if move_success is not None:
                if not isinstance(move_success, torch.Tensor):
                    move_success = torch.tensor(move_success, device=self.device, dtype=torch.float32)
                else:
                    move_success = move_success.to(self.device)
                move_success = move_success.view(-1)

            total_policy_loss = 0.0
            total_value_loss = 0.0
            total_entropy = 0.0
            total_clip_fraction = 0.0
            total_collision_loss = 0.0
            total_collision_acc = 0.0
            collision_acc_count = 0
            epochs_completed = 0
            early_stopped = False
            log_probs_final = old_log_probs

            for epoch in range(self.ppo_epochs):
                self.optimizer.zero_grad()
                log_probs_epoch = torch.zeros_like(old_log_probs)
                approx_kl_epoch = 0.0

                for chunk_batch, chunk_indices, chunk_steps in chunk_batches:
                    if chunk_steps <= 0:
                        continue

                    scale = chunk_steps / total_steps

                    actions_chunk = actions[chunk_indices]
                    old_log_probs_chunk = old_log_probs[chunk_indices]
                    advantages_chunk = advantages[chunk_indices]
                    returns_chunk = returns[chunk_indices]
                    values_old_chunk = values_old[chunk_indices]
                    move_success_chunk = move_success[chunk_indices] if move_success is not None else None

                    if self.use_legacy_actions or self.action_space_mode == "single_head_large":
                        actions_dir = None
                        actions_len = None
                        actions_stop = None
                    else:
                        if actions_chunk.ndim == 1:
                            raise ValueError("Multi-head actions require 2D actions tensor. Check use_legacy_actions and buffer format.")
                        actions_dir = actions_chunk[:, 0]
                        actions_len = actions_chunk[:, 1]
                        if actions_chunk.size(-1) > 2:
                            actions_stop = actions_chunk[:, 2]
                        else:
                            actions_stop = torch.zeros_like(actions_dir)

                    if self.use_mixed_precision:
                        autocast_context = (
                            torch.amp.autocast("cuda")
                            if hasattr(torch, "amp") and hasattr(torch.amp, "autocast")
                            else torch.cuda.amp.autocast()
                        )
                        with autocast_context:
                            logits_or_dict, values = self.forward_update(chunk_batch)
                            policy_loss, entropy, log_probs, clip_fraction = self._compute_policy_loss(
                                logits_or_dict,
                                actions_chunk,
                                actions_dir,
                                actions_len,
                                actions_stop,
                                old_log_probs_chunk,
                                advantages_chunk,
                                dir_mask,
                                len_mask,
                                stop_mask,
                            )
                            value_loss = self._compute_value_loss(values, values_old_chunk, returns_chunk)
                    else:
                        logits_or_dict, values = self.forward_update(chunk_batch)
                        policy_loss, entropy, log_probs, clip_fraction = self._compute_policy_loss(
                            logits_or_dict,
                            actions_chunk,
                            actions_dir,
                            actions_len,
                            actions_stop,
                            old_log_probs_chunk,
                            advantages_chunk,
                            dir_mask,
                            len_mask,
                            stop_mask,
                        )
                        value_loss = self._compute_value_loss(values, values_old_chunk, returns_chunk)

                    collision_loss, collision_acc = self._compute_collision_loss(
                        logits_or_dict, actions_len, move_success_chunk, actions_chunk
                    )

                    effective_entropy_coef = self.entropy_coef
                    base_entropy_coef = 0.1
                    curriculum_boost_active = self.entropy_coef > base_entropy_coef * 1.5
                    if self.adaptive_entropy and not curriculum_boost_active and entropy.item() < self.target_entropy:
                        entropy_ratio = self.target_entropy / max(entropy.item(), 0.1)
                        effective_entropy_coef = min(self.entropy_coef * entropy_ratio, self.entropy_coef_max)

                    loss = (
                        policy_loss
                        + self.value_coef * value_loss
                        - effective_entropy_coef * entropy
                        + self.collision_loss_coef * collision_loss
                    )
                    loss = loss * scale

                    if self.use_mixed_precision:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()

                    with torch.no_grad():
                        log_ratio_check = torch.clamp(log_probs - old_log_probs_chunk, -20, 20)
                        ratio_check = torch.exp(log_ratio_check)
                        approx_kl_chunk = ((ratio_check - 1.0) - log_ratio_check).mean().item()
                        approx_kl_epoch += approx_kl_chunk * scale

                    log_probs_epoch[chunk_indices] = log_probs.detach().to(log_probs_epoch.dtype)

                    total_policy_loss += policy_loss.item() * scale
                    total_value_loss += value_loss.item() * scale
                    total_entropy += entropy.item() * scale
                    total_clip_fraction += clip_fraction.item() * scale
                    total_collision_loss += collision_loss.item() * scale
                    if collision_acc is not None:
                        total_collision_acc += collision_acc.item() * scale
                        collision_acc_count += 1

                if self.use_mixed_precision:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
                    self.optimizer.step()

                epochs_completed += 1
                log_probs_final = log_probs_epoch

                if self.target_kl is not None and approx_kl_epoch > 1.5 * self.target_kl:
                    early_stopped = True
                    break

            self.reset()

            if epochs_completed > 0:
                with torch.no_grad():
                    ratio = torch.exp(log_probs_final - old_log_probs)
                    approx_kl = ((ratio - 1.0) - torch.log(ratio + 1e-8)).mean().item()
            else:
                approx_kl = 0.0

            num_epochs = max(epochs_completed, 1)
            result = {
                "loss": (total_policy_loss + self.value_coef * total_value_loss - self.entropy_coef * total_entropy) / num_epochs,
                "policy_loss": total_policy_loss / num_epochs,
                "value_loss": total_value_loss / num_epochs,
                "entropy": total_entropy / num_epochs,
                "clip_fraction": total_clip_fraction / num_epochs,
                "approx_kl": approx_kl,
                "epochs_completed": epochs_completed,
                "early_stopped": early_stopped,
                "collision_loss": total_collision_loss / num_epochs,
            }
            if collision_acc_count > 0:
                result["collision_acc"] = total_collision_acc / num_epochs

            return result

        # Multiple epochs over the same data
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        total_clip_fraction = 0
        total_collision_loss = 0
        total_collision_acc = 0
        collision_acc_count = 0
        epochs_completed = 0
        early_stopped = False
        log_probs_final = old_log_probs  # Initialize with old log probs (in case no update happens)

        for epoch in range(self.ppo_epochs):
            # Forward pass
            if self.use_mixed_precision:
                autocast_context = (
                    torch.amp.autocast("cuda") if hasattr(torch, "amp") and hasattr(torch.amp, "autocast") else torch.cuda.amp.autocast()
                )
                with autocast_context:
                    logits_or_dict, values = self.forward_update(batch)
                    policy_loss, entropy, log_probs, clip_fraction = self._compute_policy_loss(
                        logits_or_dict,
                        actions,
                        actions_dir if not self.use_legacy_actions else None,
                        actions_len if not self.use_legacy_actions else None,
                        None if self.use_legacy_actions else actions_stop,
                        old_log_probs,
                        advantages,
                        dir_mask,
                        len_mask,
                        stop_mask,
                    )
                    value_loss = self._compute_value_loss(values, values_old, returns)
            else:
                logits_or_dict, values = self.forward_update(batch)
                policy_loss, entropy, log_probs, clip_fraction = self._compute_policy_loss(
                    logits_or_dict,
                    actions,
                    actions_dir if not self.use_legacy_actions else None,
                    actions_len if not self.use_legacy_actions else None,
                    None if self.use_legacy_actions else actions_stop,
                    old_log_probs,
                    advantages,
                    dir_mask,
                    len_mask,
                    stop_mask,
                )
                value_loss = self._compute_value_loss(values, values_old, returns)

            collision_loss, collision_acc = self._compute_collision_loss(logits_or_dict, actions_len, batch.get("move_success"), actions)

            # KL Early Stopping: Check if policy changed too much
            with torch.no_grad():
                log_ratio_check = torch.clamp(log_probs - old_log_probs, -20, 20)
                ratio_check = torch.exp(log_ratio_check)
                approx_kl_epoch = ((ratio_check - 1.0) - log_ratio_check).mean().item()

            if self.target_kl is not None and approx_kl_epoch > 1.5 * self.target_kl:
                early_stopped = True
                break

            # Adaptive entropy coefficient: increase if entropy drops below target
            # IMPORTANT: Skip adaptive boost if curriculum boost is active (to avoid double-boosting)
            effective_entropy_coef = self.entropy_coef

            # Check if curriculum boost is active (entropy_coef significantly higher than base)
            base_entropy_coef = 0.1  # From config
            curriculum_boost_active = self.entropy_coef > base_entropy_coef * 1.5

            if self.adaptive_entropy and not curriculum_boost_active and entropy.item() < self.target_entropy:
                # Scale up entropy coef when entropy is low (fighting collapse)
                # Only active when NO curriculum boost is running
                entropy_ratio = self.target_entropy / max(entropy.item(), 0.1)
                effective_entropy_coef = min(self.entropy_coef * entropy_ratio, self.entropy_coef_max)

            loss = policy_loss + self.value_coef * value_loss - effective_entropy_coef * entropy + self.collision_loss_coef * collision_loss

            # Backward pass
            self.optimizer.zero_grad()

            if self.use_mixed_precision:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
                self.optimizer.step()

            # Accumulate stats
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.item()
            total_clip_fraction += clip_fraction.item()
            total_collision_loss += collision_loss.item()
            if collision_acc is not None:
                total_collision_acc += collision_acc.item()
                collision_acc_count += 1
            epochs_completed += 1

            # Store log_probs after successful backward pass (for final KL computation)
            log_probs_final = log_probs.detach()

        # Average over completed epochs
        num_epochs = max(epochs_completed, 1)

        self.reset()

        # Compute approximate KL divergence
        # Use log_probs_final which contains log_probs from the last SUCCESSFUL backward pass
        # This avoids logging KL from early-stopped epochs that were not applied
        if epochs_completed > 0:
            with torch.no_grad():
                ratio = torch.exp(log_probs_final - old_log_probs)
                approx_kl = ((ratio - 1.0) - torch.log(ratio + 1e-8)).mean().item()
        else:
            # No epochs completed - early stopped before any backward pass
            approx_kl = 0.0

        result = {
            "loss": (total_policy_loss + self.value_coef * total_value_loss - self.entropy_coef * total_entropy) / num_epochs,
            "policy_loss": total_policy_loss / num_epochs,
            "value_loss": total_value_loss / num_epochs,
            "entropy": total_entropy / num_epochs,
            "clip_fraction": total_clip_fraction / num_epochs,
            "approx_kl": approx_kl,
            "epochs_completed": epochs_completed,
            "early_stopped": early_stopped,
            "collision_loss": total_collision_loss / num_epochs,
        }
        if collision_acc_count > 0:
            result["collision_acc"] = total_collision_acc / collision_acc_count

        return result

    def _calibrate_trajectory_chunk_size(
        self, batch, trajectory_info, actions, old_log_probs, advantages, returns, values_old, dir_mask, len_mask, stop_mask
    ):
        if not torch.cuda.is_available():
            print("[WARNING] Auto chunk calibration requested but CUDA is unavailable; using current chunk size.")
            return self.trajectory_chunk_size

        total_steps = int(actions.shape[0]) if actions.ndim > 0 else 0
        if total_steps <= 0:
            print("[WARNING] Auto chunk calibration skipped (no steps).")
            return self.trajectory_chunk_size

        min_size = max(1, int(self.trajectory_chunk_min))
        max_size = max(min_size, int(self.trajectory_chunk_max))
        max_size = min(max_size, total_steps)
        backoff = self.trajectory_chunk_backoff
        if backoff <= 0.0 or backoff > 1.0:
            backoff = 0.9
        print()
        print(f"[INFO] Auto chunk calibration: probing {min_size}..{max_size} steps (backoff {backoff:.2f})")

        cache_state_features = getattr(self, "cache_state_features", False)

        def _try_chunk_size(chunk_size):
            nonlocal cache_state_features
            chunk_batches = self._build_chunk_batches(batch, trajectory_info, chunk_size)
            if not chunk_batches:
                return False

            chunk_batch, chunk_indices, chunk_steps = max(chunk_batches, key=lambda item: item[2])
            if chunk_steps <= 0:
                return False

            actions_chunk = actions[chunk_indices]
            old_log_probs_chunk = old_log_probs[chunk_indices]
            advantages_chunk = advantages[chunk_indices]
            returns_chunk = returns[chunk_indices]
            values_old_chunk = values_old[chunk_indices]

            move_success = batch.get("move_success")
            if move_success is not None:
                if not isinstance(move_success, torch.Tensor):
                    move_success = torch.tensor(move_success, device=self.device, dtype=torch.float32)
                else:
                    move_success = move_success.to(self.device)
                move_success = move_success.view(-1)
                move_success_chunk = move_success[chunk_indices]
            else:
                move_success_chunk = None

            if self.use_legacy_actions:
                actions_dir = None
                actions_len = None
                actions_stop = None
            else:
                if actions_chunk.ndim == 1:
                    return False
                actions_dir = actions_chunk[:, 0]
                actions_len = actions_chunk[:, 1]
                if actions_chunk.size(-1) > 2:
                    actions_stop = actions_chunk[:, 2]
                else:
                    actions_stop = torch.zeros_like(actions_dir)

            original_cache_state = cache_state_features
            if original_cache_state:
                self.cache_state_features = False
                if hasattr(self, "_cached_state_features"):
                    del self._cached_state_features

            self.optimizer.zero_grad(set_to_none=True)

            try:
                if self.use_mixed_precision:
                    autocast_context = (
                        torch.amp.autocast("cuda")
                        if hasattr(torch, "amp") and hasattr(torch.amp, "autocast")
                        else torch.cuda.amp.autocast()
                    )
                    with autocast_context:
                        logits_or_dict, values = self.forward_update(chunk_batch)
                        policy_loss, entropy, log_probs, _ = self._compute_policy_loss(
                            logits_or_dict,
                            actions_chunk,
                            actions_dir,
                            actions_len,
                            actions_stop,
                            old_log_probs_chunk,
                            advantages_chunk,
                            dir_mask,
                            len_mask,
                            stop_mask,
                        )
                        value_loss = self._compute_value_loss(values, values_old_chunk, returns_chunk)
                else:
                    logits_or_dict, values = self.forward_update(chunk_batch)
                    policy_loss, entropy, log_probs, _ = self._compute_policy_loss(
                        logits_or_dict,
                        actions_chunk,
                        actions_dir,
                        actions_len,
                        actions_stop,
                        old_log_probs_chunk,
                        advantages_chunk,
                        dir_mask,
                        len_mask,
                        stop_mask,
                    )
                    value_loss = self._compute_value_loss(values, values_old_chunk, returns_chunk)

                collision_loss, _ = self._compute_collision_loss(logits_or_dict, actions_len, move_success_chunk, actions_chunk)
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy + self.collision_loss_coef * collision_loss

                if self.use_mixed_precision and self.scaler is not None:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
            finally:
                if original_cache_state:
                    self.cache_state_features = True
                if hasattr(self, "_cached_state_features"):
                    del self._cached_state_features
                self.optimizer.zero_grad(set_to_none=True)

            del loss, value_loss, policy_loss, entropy, log_probs, logits_or_dict, values
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return True

        best = None
        size = min_size
        last_ok = None

        while size <= max_size:
            try:
                ok = _try_chunk_size(size)
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    ok = False
                else:
                    raise

            if ok:
                best = size
                last_ok = size
                size = min(size * 2, max_size + 1)
            else:
                break

        if best is None:
            print("[WARNING] Auto chunk calibration failed at min size; using minimum.")
            return min_size

        upper = min(size - 1, max_size)
        low = last_ok
        high = upper

        while low is not None and low < high:
            mid = (low + high + 1) // 2
            try:
                ok = _try_chunk_size(mid)
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    ok = False
                else:
                    raise

            if ok:
                best = mid
                low = mid
            else:
                high = mid - 1

        final_size = max(min_size, int(best * backoff))
        final_size = min(final_size, max_size)

        print(f"[INFO] Auto chunk calibration complete: selected {final_size} (max fit {best})")
        return final_size

    def _split_trajectory_info(self, trajectory_info, max_steps):
        segments = []
        if not trajectory_info:
            return segments

        max_steps = int(max_steps)
        for traj in trajectory_info:
            start_idx = int(traj["start_idx"])
            length = int(traj["length"])
            init_hidden = traj.get("initial_hidden", None)

            if max_steps <= 0 or length <= max_steps:
                segments.append({"start_idx": start_idx, "length": length, "initial_hidden": init_hidden})
                continue

            offset = 0
            first_segment = True
            while offset < length:
                seg_len = min(max_steps, length - offset)
                segments.append(
                    {"start_idx": start_idx + offset, "length": seg_len, "initial_hidden": init_hidden if first_segment else None}
                )
                offset += seg_len
                first_segment = False

        return segments

    def _build_trajectory_chunks(self, segments, max_steps):
        if not segments:
            return []

        total_steps = sum(int(seg["length"]) for seg in segments)
        if total_steps <= 0:
            return []

        max_steps = max(1, int(max_steps))
        num_chunks = max(1, math.ceil(total_steps / max_steps))
        target_steps = math.ceil(total_steps / num_chunks)

        chunks = []
        current = []
        current_steps = 0

        for seg in segments:
            seg_len = int(seg["length"])
            if current and current_steps + seg_len > target_steps and (len(chunks) + 1 < num_chunks):
                chunks.append(current)
                current = [seg]
                current_steps = seg_len
            else:
                current.append(seg)
                current_steps += seg_len

        if current:
            chunks.append(current)

        return chunks

    def _build_chunk_batches(self, batch, trajectory_info, max_steps):
        segments = self._split_trajectory_info(trajectory_info, max_steps)
        if not segments:
            return []

        chunks = self._build_trajectory_chunks(segments, max_steps)

        def _extract_seq(key):
            value = batch.get(key)
            if value is None:
                return None
            if isinstance(value, list):
                if len(value) == 1:
                    return value[0]
                raise ValueError("Trajectory-aware update expects batch size B=1 after flattening.")
            return value

        rgb_seq = _extract_seq("rgb")
        depth_seq = _extract_seq("depth")
        lssg_seq = _extract_seq("lssg")
        gssg_seq = _extract_seq("gssg")
        last_actions = batch.get("last_actions")

        chunk_batches = []
        for chunk in chunks:
            if not chunk:
                continue

            chunk_start = int(chunk[0]["start_idx"])
            chunk_end = int(chunk[-1]["start_idx"] + chunk[-1]["length"])
            chunk_steps = chunk_end - chunk_start
            if chunk_steps <= 0:
                continue

            chunk_batch = {}
            if rgb_seq is not None:
                chunk_rgb = rgb_seq[chunk_start:chunk_end]
                if not isinstance(chunk_rgb, list):
                    chunk_rgb = list(chunk_rgb)
                chunk_batch["rgb"] = [chunk_rgb]
            if depth_seq is not None:
                chunk_depth = depth_seq[chunk_start:chunk_end]
                if not isinstance(chunk_depth, list):
                    chunk_depth = list(chunk_depth)
                chunk_batch["depth"] = [chunk_depth]
            if lssg_seq is not None:
                chunk_lssg = lssg_seq[chunk_start:chunk_end]
                if not isinstance(chunk_lssg, list):
                    chunk_lssg = list(chunk_lssg)
                chunk_batch["lssg"] = [chunk_lssg]
            if gssg_seq is not None:
                chunk_gssg = gssg_seq[chunk_start:chunk_end]
                if not isinstance(chunk_gssg, list):
                    chunk_gssg = list(chunk_gssg)
                chunk_batch["gssg"] = [chunk_gssg]
            if last_actions is not None:
                if last_actions.dim() == 3:
                    chunk_batch["last_actions"] = last_actions[:, chunk_start:chunk_end, :]
                elif last_actions.dim() == 2:
                    chunk_batch["last_actions"] = last_actions[chunk_start:chunk_end, :].unsqueeze(0)
                else:
                    chunk_batch["last_actions"] = last_actions

            chunk_trajectory_info = []
            offset = 0
            for seg in chunk:
                seg_len = int(seg["length"])
                chunk_trajectory_info.append({"start_idx": offset, "length": seg_len, "initial_hidden": seg.get("initial_hidden", None)})
                offset += seg_len

            chunk_batch["trajectory_info"] = chunk_trajectory_info

            chunk_indices = torch.arange(chunk_start, chunk_end, device=self.device, dtype=torch.long)
            chunk_batches.append((chunk_batch, chunk_indices, chunk_steps))

        return chunk_batches

    def _compute_policy_loss(
        self, logits_or_dict, actions, actions_dir, actions_len, actions_stop, old_log_probs, advantages, dir_mask, len_mask, stop_mask
    ):
        """
        Compute policy loss with proper action masking for curriculum learning.

        Args:
            logits_or_dict: Model output (dict or tensor) with action logits
            actions: Action indices (multi-head format or flattened)
            actions_dir: Direction action indices (multi-head only)
            actions_len: Length action indices (multi-head only)
            actions_stop: Stop action indices (multi-head only)
            old_log_probs: Log probabilities from previous forward pass
            advantages: Advantage estimates for policy gradient
            dir_mask: Curriculum mask for direction actions (None if no curriculum)
            len_mask: Curriculum mask for length actions (None if no curriculum)
            stop_mask: Curriculum mask for stop actions (None if no curriculum)

        Returns:
            policy_loss: PPO clipped surrogate loss
            entropy: Mean entropy for exploration bonus
            log_probs: Log probabilities of current policy
            clip_fraction: Fraction of samples clipped (diagnostic)
        """
        if self.use_legacy_actions:
            # Extract action logits from dict (now includes collision head)
            if isinstance(logits_or_dict, dict):
                logits = logits_or_dict["action"].view(-1, logits_or_dict["action"].size(-1))
            else:
                # Backward compatibility
                logits = logits_or_dict.view(-1, logits_or_dict.size(-1))
            dist = Categorical(logits=logits)

            if actions.ndim > 1:
                actions = actions.view(-1)

            log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()
        elif self.action_space_mode == "single_head_large":
            # Single-head-large: flat action space with curriculum masking
            if isinstance(logits_or_dict, dict):
                logits = logits_or_dict["action"].view(-1, logits_or_dict["action"].size(-1))
            else:
                logits = logits_or_dict.view(-1, logits_or_dict.size(-1))

            # Apply flat action mask (dir_mask is repurposed as flat_mask)
            if dir_mask is not None:
                logits = self._apply_action_mask(logits, dir_mask)

            dist = Categorical(logits=logits)

            if actions.ndim > 1:
                actions = actions.view(-1)

            log_probs = dist.log_prob(actions)

            # Entropy computed only over valid actions
            entropy = self._compute_masked_entropy(logits, dir_mask) if dir_mask is not None else dist.entropy().mean()
        else:
            logits_dict = logits_or_dict
            logits_dir = logits_dict["direction"].view(-1, logits_dict["direction"].size(-1))
            logits_len = logits_dict["length"].view(-1, logits_dict["length"].size(-1))
            logits_stop = logits_dict["stop"].view(-1, logits_dict["stop"].size(-1))

            # Apply action masking for consistent log_prob computation
            logits_dir_masked = self._apply_action_mask(logits_dir, dir_mask)
            logits_len_masked = self._apply_action_mask(logits_len, len_mask)
            logits_stop_masked = self._apply_action_mask(logits_stop, stop_mask)

            dist_dir = Categorical(logits=logits_dir_masked)
            dist_len = Categorical(logits=logits_len_masked)
            dist_stop = Categorical(logits=logits_stop_masked)

            log_probs_dir = dist_dir.log_prob(actions_dir)
            log_probs_len = dist_len.log_prob(actions_len)
            log_probs_stop = dist_stop.log_prob(actions_stop)
            log_probs = log_probs_dir + log_probs_len + log_probs_stop

            # Entropy computed only over valid actions
            entropy_dir = self._compute_masked_entropy(logits_dir, dir_mask)
            entropy_len = self._compute_masked_entropy(logits_len, len_mask)
            entropy_stop = self._compute_masked_entropy(logits_stop, stop_mask)
            entropy = entropy_dir + entropy_len + entropy_stop

        # PPO ratio and clipping
        log_ratio = torch.clamp(log_probs - old_log_probs, -20, 20)
        ratio = torch.exp(log_ratio)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # Clip fraction (diagnostic)
        clip_fraction = ((ratio - 1.0).abs() > self.clip_epsilon).float().mean()

        return policy_loss, entropy, log_probs, clip_fraction

    def _compute_value_loss(self, values, values_old, returns):
        """Compute value loss with optional clipping."""
        if self.clip_value:
            values_clipped = values_old + torch.clamp(values - values_old, -self.clip_value_epsilon, self.clip_value_epsilon)
            value_loss_unclipped = F.mse_loss(values, returns)
            value_loss_clipped = F.mse_loss(values_clipped, returns)
            value_loss = torch.max(value_loss_unclipped, value_loss_clipped)
        else:
            value_loss = F.mse_loss(values, returns)
        return value_loss

    def get_agent_info(self):
        """
        Return basic information about the agent.
        """
        return {
            "Agent Name": "PPO Agent",
            "alpha": self.alpha,
            "gamma": self.gamma,
            "entropy_coef": self.entropy_coef,
            "clip_epsilon": self.clip_epsilon,
            "ppo_epochs": self.ppo_epochs,
            "use_gae": self.use_gae,
            "gae_lambda": self.gae_lambda if self.use_gae else None,
        }

    # ============================================================================
    # VECTORIZED ENVIRONMENT SUPPORT
    # ============================================================================

    def act_batch(self, obs_list, hiddens_list, last_actions_list=None, deterministic=False):
        """
        VECTORIZED: Select actions for a batch of observations from multiple environments.

        All environments are processed in a SINGLE forward pass for maximum efficiency.
        This eliminates the sequential loop bottleneck.

        Args:
            obs_list: List of observations, one per environment
            hiddens_list: List of (lssg_h, gssg_h, policy_h, stag_state) tuples, one per env
            last_actions_list: List of last actions (tensors or tuples), one per env
            deterministic: Whether to select actions deterministically

        Returns:
            actions: List of (dir_idx, len_idx, stop_flag) tuples
            hiddens_new: List of updated hidden state tuples
            log_probs: Tensor of log probabilities (for training)
            values: Tensor of value estimates (for training)
        """
        was_training = self.training
        self.eval()
        with torch.no_grad():
            n_envs = len(obs_list)

            # If no last_actions provided, use defaults
            if last_actions_list is None:
                last_actions_list = [torch.tensor([-1, -1], dtype=torch.long, device=self.device) for _ in range(n_envs)]

            # ========== VECTORIZED PREPARATION ==========
            # 1. Prepare batch_dict for encoder (B=n_envs, T=1)
            batch_dict = {
                "rgb": [[obs.state[0]] for obs in obs_list],
                "depth": [[obs.state[1]] for obs in obs_list],
                "lssg": [[obs.state[2]] for obs in obs_list],
                "gssg": [[obs.state[3]] for obs in obs_list],
            }

            # 2. Stack last actions [n_envs, 2]
            last_actions_tensor = torch.stack(
                [la if isinstance(la, torch.Tensor) else torch.tensor(la, dtype=torch.long, device=self.device) for la in last_actions_list]
            )  # [n_envs, 2]
            last_actions_tensor = last_actions_tensor.unsqueeze(1)  # [n_envs, 1, 2]

            # 3. Stack LSTM hidden states
            # For LSTM: hidden = (h, c) where h/c are [num_layers, batch, hidden_size]
            # We need to stack along batch dimension (dim=1)
            lssg_h_list, gssg_h_list, policy_h_list, stag_state_list = [], [], [], []

            for hiddens in hiddens_list:
                if len(hiddens) == 4:
                    lssg_h, gssg_h, policy_h, stag_state = hiddens
                else:
                    lssg_h, gssg_h, policy_h = hiddens
                    stag_state = None
                lssg_h_list.append(lssg_h)
                gssg_h_list.append(gssg_h)
                policy_h_list.append(policy_h)
                stag_state_list.append(stag_state)

            # Stack LSTM hidden states if they exist
            def stack_lstm_hidden(h_list):
                """Stack list of (h, c) tuples along batch dimension."""
                if all(h is None for h in h_list):
                    return None
                sample = next(h for h in h_list if h is not None)
                h_shape = sample[0].shape
                c_shape = sample[1].shape
                h_tensors, c_tensors = [], []
                for h in h_list:
                    if h is None:
                        h_tensors.append(torch.zeros(h_shape, device=sample[0].device, dtype=sample[0].dtype))
                        c_tensors.append(torch.zeros(c_shape, device=sample[1].device, dtype=sample[1].dtype))
                    else:
                        h_tensors.append(h[0])
                        c_tensors.append(h[1])
                h_stack = torch.cat(h_tensors, dim=1)  # [num_layers, n_envs, hidden_size]
                c_stack = torch.cat(c_tensors, dim=1)
                return (h_stack, c_stack)

            lssg_h_batch = stack_lstm_hidden(lssg_h_list)
            gssg_h_batch = stack_lstm_hidden(gssg_h_list)
            policy_h_batch = stack_lstm_hidden(policy_h_list)

            # Stack stagnation states (handle mixed None by zero-filling)
            if any(s is not None for s in stag_state_list):
                sample = next(s for s in stag_state_list if s is not None)
                prev_shape = sample["prev_gssg"].shape
                smoothed_shape = sample["smoothed_change"].shape
                steps_shape = sample["steps_since_discovery"].shape
                prev_list, smoothed_list, steps_list = [], [], []
                for s in stag_state_list:
                    if s is None:
                        prev_list.append(torch.zeros(prev_shape, device=sample["prev_gssg"].device, dtype=sample["prev_gssg"].dtype))
                        smoothed_list.append(
                            torch.zeros(smoothed_shape, device=sample["smoothed_change"].device, dtype=sample["smoothed_change"].dtype)
                        )
                        steps_list.append(
                            torch.zeros(
                                steps_shape, device=sample["steps_since_discovery"].device, dtype=sample["steps_since_discovery"].dtype
                            )
                        )
                    else:
                        prev_list.append(s["prev_gssg"])
                        smoothed_list.append(s["smoothed_change"])
                        steps_list.append(s["steps_since_discovery"])
                stag_state_batch = {
                    "prev_gssg": torch.cat(prev_list, dim=0),
                    "smoothed_change": torch.cat(smoothed_list, dim=0),
                    "steps_since_discovery": torch.cat(steps_list, dim=0),
                }
            else:
                stag_state_batch = None

            # ========== SINGLE BATCHED FORWARD PASS ==========
            features, lssg_h_new, gssg_h_new, stag_state_new = self.encoder.forward_seq(
                batch_dict, last_actions_tensor, lssg_hidden=lssg_h_batch, gssg_hidden=gssg_h_batch, stagnation_state=stag_state_batch
            )
            # features: [n_envs, 1, feature_dim]

            logits_or_dict, value, policy_h_new = self.policy(features, hidden=policy_h_batch)
            # logits: [n_envs, 1, ...] or dict with such tensors
            # value: [n_envs, 1] or [n_envs, 1, 1]

            # ========== UNPACK BATCHED RESULTS ==========
            # Squeeze batch dimension (B=1)
            values = None
            if value is not None:
                if value.dim() == 3:
                    values = value.squeeze(-1).squeeze(1)  # [n_envs]
                elif value.dim() == 2:
                    values = value.squeeze(1)  # [n_envs]
                else:
                    values = value

            # Unstack hidden states back into per-env tuples
            def unstack_lstm_hidden(h_batched, env_idx):
                """Extract hidden state for specific environment."""
                if h_batched is None:
                    return None
                return (h_batched[0][:, env_idx : env_idx + 1, :].contiguous(), h_batched[1][:, env_idx : env_idx + 1, :].contiguous())

            hiddens_new = []
            for env_idx in range(n_envs):
                lssg_h_env = unstack_lstm_hidden(lssg_h_new, env_idx)
                gssg_h_env = unstack_lstm_hidden(gssg_h_new, env_idx)
                policy_h_env = unstack_lstm_hidden(policy_h_new, env_idx)

                if stag_state_new is not None:
                    stag_state_env = {
                        "prev_gssg": stag_state_new["prev_gssg"][env_idx : env_idx + 1, :],
                        "smoothed_change": stag_state_new["smoothed_change"][env_idx : env_idx + 1, :],
                        "steps_since_discovery": stag_state_new["steps_since_discovery"][env_idx : env_idx + 1, :],
                    }
                else:
                    stag_state_env = None

                hiddens_new.append((lssg_h_env, gssg_h_env, policy_h_env, stag_state_env))

            # ========== SAMPLE ACTIONS (VECTORIZED) ==========
            actions = []

            if self.use_legacy_actions:
                # Legacy actions - extract action logits from dict
                if isinstance(logits_or_dict, dict):
                    logits = logits_or_dict["action"].squeeze(1)  # [n_envs, num_actions]
                else:
                    # Backward compatibility
                    logits = logits_or_dict.squeeze(1)  # [n_envs, num_actions]
                dist = Categorical(logits=logits)

                if deterministic:
                    action_indices = torch.argmax(logits, dim=-1)  # [n_envs]
                else:
                    action_indices = dist.sample()  # [n_envs]

                log_probs = dist.log_prob(action_indices)  # [n_envs]
                actions = action_indices.tolist()
            elif self.action_space_mode == "single_head_large":
                # Single-head-large: flat action space with curriculum masking
                if isinstance(logits_or_dict, dict):
                    logits = logits_or_dict["action"].squeeze(1)  # [n_envs, 504]
                else:
                    logits = logits_or_dict.squeeze(1)

                # Apply curriculum mask (broadcast over all envs)
                if hasattr(self.env, "valid_flat_action_indices"):
                    mask = torch.zeros(logits.shape[-1], dtype=torch.bool, device=self.device)
                    for idx in self.env.valid_flat_action_indices:
                        mask[idx] = True
                    logits = logits.masked_fill(~mask.unsqueeze(0), float("-inf"))

                dist = Categorical(logits=logits)

                if deterministic:
                    action_indices = torch.argmax(logits, dim=-1)  # [n_envs]
                else:
                    action_indices = dist.sample()  # [n_envs]

                log_probs = dist.log_prob(action_indices)  # [n_envs]
                actions = action_indices.tolist()
            else:
                # Multi-head actions
                logits_dict = logits_or_dict
                logits_dir = logits_dict["direction"].squeeze(1)  # [n_envs, num_dir]
                logits_len = logits_dict["length"].squeeze(1)  # [n_envs, num_len]
                logits_stop = logits_dict["stop"].squeeze(1)  # [n_envs, 2]

                # Apply action masking (broadcast across all envs)
                if hasattr(self.env, "get_action_masks"):
                    action_masks = self.env.get_action_masks()
                    dir_mask = torch.tensor(action_masks["direction"], dtype=torch.bool, device=self.device)
                    len_mask = torch.tensor(action_masks["length"], dtype=torch.bool, device=self.device)

                    # Broadcast mask: [num_actions] -> [n_envs, num_actions]
                    logits_dir_masked = logits_dir.masked_fill(~dir_mask.unsqueeze(0), float("-inf"))
                    logits_len_masked = logits_len.masked_fill(~len_mask.unsqueeze(0), float("-inf"))
                    logits_stop_masked = logits_stop
                else:
                    logits_dir_masked = logits_dir
                    logits_len_masked = logits_len
                    logits_stop_masked = logits_stop

                dist_dir = Categorical(logits=logits_dir_masked)
                dist_len = Categorical(logits=logits_len_masked)
                dist_stop = Categorical(logits=logits_stop_masked)

                if deterministic:
                    dir_indices = torch.argmax(logits_dir_masked, dim=-1)  # [n_envs]
                    len_indices = torch.argmax(logits_len_masked, dim=-1)
                    stop_indices = torch.argmax(logits_stop_masked, dim=-1)
                else:
                    dir_indices = dist_dir.sample()  # [n_envs]
                    len_indices = dist_len.sample()
                    stop_indices = dist_stop.sample()

                log_probs_dir = dist_dir.log_prob(dir_indices)  # [n_envs]
                log_probs_len = dist_len.log_prob(len_indices)
                log_probs_stop = dist_stop.log_prob(stop_indices)
                log_probs = log_probs_dir + log_probs_len + log_probs_stop

                # Convert to list of tuples
                actions = [(dir_indices[i].item(), len_indices[i].item(), stop_indices[i].item()) for i in range(n_envs)]

        if was_training:
            self.train()

        return actions, hiddens_new, log_probs, values

    def collect_rollouts_vec(self, vec_env, buffer: VecEnvRolloutBuffer, n_steps: int):
        """
        Collect rollouts from vectorized environment.

        Args:
            vec_env: VecEnv instance with n_envs parallel environments
            buffer: VecEnvRolloutBuffer to store transitions
            n_steps: Number of steps to collect per environment

        Returns:
            Dictionary with collection statistics
        """
        n_envs = vec_env.num_envs

        # Initialize hidden states for all envs (lssg_h, gssg_h, policy_h, stag_state)
        hiddens_list = [(None, None, None, None) for _ in range(n_envs)]

        # Reset all environments
        obs_list, info_list = vec_env.reset()

        # Initialize last actions
        last_actions = [torch.tensor([-1, -1], dtype=torch.long, device=self.device) for _ in range(n_envs)]
        agent_positions = [None for _ in range(n_envs)]

        total_reward = 0
        episodes_done = 0
        episode_rewards = []

        # Collect n_steps from each environment
        for step in range(n_steps):
            # Get actions for all envs
            actions, hiddens_new, log_probs, values = self.act_batch(
                obs_list, hiddens_list, last_actions_list=last_actions, deterministic=False
            )

            # Step all environments
            next_obs_list, rewards, dones, truncated, infos = vec_env.step(actions)

            # Add to buffer
            states = [obs.state for obs in obs_list]
            buffer.add(
                states=states,
                actions=actions,
                rewards=rewards,
                dones=[d or t for d, t in zip(dones, truncated)],
                hiddens=hiddens_list,
                last_actions=[la.cpu().numpy().tolist() if isinstance(la, torch.Tensor) else la for la in last_actions],
                agent_positions=agent_positions,
                move_successes=[info.get("move_action_success", True) for info in infos],
            )

            # Update for next iteration
            total_reward += sum(rewards)

            for env_idx in range(n_envs):
                if dones[env_idx] or truncated[env_idx]:
                    episodes_done += 1
                    episode_rewards.append(infos[env_idx].get("score", 0))
                    # Reset hidden states for this env (lssg_h, gssg_h, policy_h, stag_state)
                    hiddens_list[env_idx] = (None, None, None, None)
                    last_actions[env_idx] = torch.tensor([-1, -1], dtype=torch.long, device=self.device)
                    agent_positions[env_idx] = None
                else:
                    hiddens_list[env_idx] = hiddens_new[env_idx]
                    # Convert action to tensor: for legacy actions, store as [action_idx, -1] for compatibility
                    if self.use_legacy_actions:
                        last_actions[env_idx] = torch.tensor([actions[env_idx], -1], dtype=torch.long, device=self.device)
                    else:
                        act = actions[env_idx]
                        if isinstance(act, (tuple, list)) and len(act) == 3:
                            act = (act[0], act[1])  # keep only (dir, len) for embeddings
                        last_actions[env_idx] = torch.tensor(act, dtype=torch.long, device=self.device)
                    agent_positions[env_idx] = infos[env_idx].get("agent_pos", None)

            obs_list = next_obs_list

        return {
            "total_reward": total_reward,
            "episodes_done": episodes_done,
            "mean_episode_reward": sum(episode_rewards) / len(episode_rewards) if episode_rewards else 0,
            "steps_collected": n_steps * n_envs,
        }
