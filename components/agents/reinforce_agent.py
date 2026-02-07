import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from components.agents.abstract_agent import AbstractAgent


class ReinforceAgent(AbstractAgent):
    """
    REINFORCE policy-gradient agent using a pretrained feature encoder.
    """

    def __init__(self, env, navigation_config, agent_config, device=None, mapping_path=None):
        super().__init__(env, navigation_config, agent_config, device, mapping_path)
        self.entropy_coef = agent_config.get("entropy_coef", 0.0)

    def update(self, batch=None):
        """Update policy using REINFORCE algorithm.

        Args:
            batch: Optional external batch (for VecEnv training). If None, uses internal buffer.

        Returns:
            dict with loss components: loss, policy_loss, entropy, collision_loss, ret_std
        """
        if batch is None:
            batch = self._get_update_values()

        logits_or_dict = self.forward_update(batch)
        # Forward may return (logits, value) or a dict; unwrap and pick the correct head
        if isinstance(logits_or_dict, tuple):
            logits_or_dict = logits_or_dict[0]
        if isinstance(logits_or_dict, dict):
            # Legacy head is "action"; fall back to "direction" for multi-head-style dicts
            if "action" in logits_or_dict:
                logits = logits_or_dict["action"]
            elif "direction" in logits_or_dict:
                logits = logits_or_dict["direction"]
            else:
                raise ValueError("Expected 'action' or 'direction' logits for REINFORCE update.")
        else:
            logits = logits_or_dict

        # Returns
        if not isinstance(batch["returns"], torch.Tensor):
            returns = torch.tensor(batch["returns"], device=self.device).view(-1)
        else:
            returns = batch["returns"].to(self.device).view(-1)

        if self.use_legacy_actions:
            # Legacy: Single action logits tensor [B, T, num_actions]
            logits = logits.view(-1, logits.size(-1))  # [B*T, num_actions]

            # Actions are [B, T, 2] with format [action_idx, -1]
            if not isinstance(batch["actions"], torch.Tensor):
                actions = torch.tensor(batch["actions"], device=self.device)
            else:
                actions = batch["actions"].to(self.device)

            if actions.ndim == 3:
                actions = actions.view(-1, 2)

            # Extract action index - if already 1D, use as is; if 2D, extract first column
            if actions.ndim == 1:
                action_indices = actions
            else:
                action_indices = actions[:, 0]

            # Single distribution for legacy actions
            probs = F.softmax(logits, dim=-1)
            dist = Categorical(probs=probs)

            log_probs = dist.log_prob(action_indices)
            entropy = dist.entropy().mean()

            # Normalize returns
            if returns.numel() > 1:
                ret_std = returns.std().item()
            else:
                ret_std = 1.0
            ret_std = ret_std if ret_std > 1e-2 else 1e-2
            returns = (returns - returns.mean()) / ret_std

            policy_loss = -torch.mean(log_probs * returns)

            # Compute collision loss (auxiliary depth learning)
            collision_loss, collision_acc = self._compute_collision_loss(
                logits_or_dict, None, batch.get("move_success"), batch["actions"]
            )

            loss = policy_loss - self.entropy_coef * entropy + self.collision_loss_coef * collision_loss

            result = {
                "loss": loss.item(),
                "policy_loss": policy_loss.item(),
                "collision_loss": collision_loss.item(),
                "entropy": entropy.item(),
                "ret_std": ret_std
            }
            if collision_acc is not None:
                result["collision_acc"] = collision_acc.item()
        else:
            # Multi-Head Actions: Separate distributions for direction and length
            logits_dict = logits_or_dict
            logits_dir = logits_dict["direction"]  # [B, T, num_directions]
            logits_len = logits_dict["length"]     # [B, T, num_lengths]

            # Flatten to [B*T, num_directions/num_lengths]
            logits_dir = logits_dir.view(-1, logits_dir.size(-1))
            logits_len = logits_len.view(-1, logits_len.size(-1))

            probs_dir = F.softmax(logits_dir, dim=-1)
            probs_len = F.softmax(logits_len, dim=-1)

            dist_dir = Categorical(probs=probs_dir)
            dist_len = Categorical(probs=probs_len)

            # Actions are [B, T, 2] or already flattened
            if not isinstance(batch["actions"], torch.Tensor):
                actions = torch.tensor(batch["actions"], device=self.device)
            else:
                actions = batch["actions"].to(self.device)

            # Reshape to [B*T, 2] if needed
            if actions.ndim == 3:
                actions = actions.view(-1, 2)

            actions_dir = actions[:, 0]  # [B*T]
            actions_len = actions[:, 1]  # [B*T]

            # Compute log probs for both heads
            log_probs_dir = dist_dir.log_prob(actions_dir)
            log_probs_len = dist_len.log_prob(actions_len)
            log_probs = log_probs_dir + log_probs_len  # Combined log prob

            # Entropy for both heads
            entropy_dir = dist_dir.entropy().mean()
            entropy_len = dist_len.entropy().mean()
            entropy = (entropy_dir + entropy_len) / 2.0

            if returns.numel() > 1:
                ret_std = returns.std().item()
            else:
                ret_std = 1.0
            ret_std = ret_std if ret_std > 1e-2 else 1e-2
            returns = (returns - returns.mean()) / ret_std

            policy_loss = -torch.mean(log_probs * returns)

            # Compute collision loss (auxiliary depth learning)
            collision_loss, collision_acc = self._compute_collision_loss(
                logits_or_dict, actions_len, batch.get("move_success"), batch["actions"]
            )

            loss = policy_loss - self.entropy_coef * entropy + self.collision_loss_coef * collision_loss

            result = {
                "loss": loss.item(),
                "policy_loss": policy_loss.item(),
                "collision_loss": collision_loss.item(),
                "entropy": entropy.item(),
                "entropy_dir": entropy_dir.item(),
                "entropy_len": entropy_len.item(),
                "ret_std": ret_std
            }
            if collision_acc is not None:
                result["collision_acc"] = collision_acc.item()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)
        self.optimizer.step()

        self.reset()

        return result

    def get_agent_info(self):
        """
        Return basic information about the agent.
        :return: Dictionary with agent info
        """
        return {"Agent Name": "REINFORCE Agent", "alpha": self.alpha, "gamma": self.gamma}
