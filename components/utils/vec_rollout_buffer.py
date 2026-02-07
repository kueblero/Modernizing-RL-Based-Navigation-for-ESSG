"""
Vectorized Rollout Buffer for parallel PPO training.

Collects data from multiple parallel environments and handles:
- Asynchronous episode termination
- Per-environment trajectory tracking
- Correct GAE computation with done masks
- Batch flattening for PPO updates
"""

import torch
from typing import List, Dict, Any, Optional


class VecEnvRolloutBuffer:
    """
    Rollout buffer for vectorized environments.

    Key features:
    - Tracks separate trajectories for each environment
    - Handles asynchronous episode endings
    - Computes GAE per trajectory with proper done masking
    - Flattens all trajectories into single batch for PPO

    Example:
        buffer = VecEnvRolloutBuffer(n_envs=8, n_steps_per_env=128)

        # Collect data
        for step in range(n_steps):
            actions = agent.act(obs, hiddens)
            next_obs, rewards, dones, infos = vec_env.step(actions)
            buffer.add(obs, actions, rewards, dones, hiddens, ...)

        # Get batch for update
        batch = buffer.get(gamma=0.99, gae_lambda=0.95)
        agent.update(batch)
        buffer.clear()
    """

    def __init__(self, n_envs: int, n_steps_per_env: int):
        """
        Args:
            n_envs: Number of parallel environments
            n_steps_per_env: Number of steps to collect per environment before update
        """
        self.n_envs = n_envs
        self.n_steps_per_env = n_steps_per_env
        self.total_steps_needed = n_envs * n_steps_per_env

        # Per-environment trajectories (list of completed trajectories) and currently open trajectories
        # Each trajectory is a list of transitions (dicts)
        self.completed_trajectories = [[] for _ in range(n_envs)]
        self.current_trajectories = [[] for _ in range(n_envs)]

        # Track hidden states at the start of each trajectory (aligned with completed_trajectories)
        self.initial_hiddens_completed = [[] for _ in range(n_envs)]
        self.initial_hiddens_current = [None for _ in range(n_envs)]

        # Total step counter
        self.total_steps = 0

    def add(
        self,
        states: List[List],  # List of [rgb, depth, lssg, gssg] per env
        actions: List[tuple],  # List of (dir, len) per env
        rewards: List[float],
        dones: List[bool],
        hiddens: List[tuple],  # List of (lssg_h, gssg_h, policy_h) per env
        last_actions: List[tuple],
        agent_positions: List[tuple],
        move_successes: List[bool],
    ):
        """
        Add a step of experience from all environments.

        Args:
            states: List of states, one per environment
            actions: List of actions taken
            rewards: List of rewards received
            dones: List of done flags
            hiddens: List of hidden state tuples (before action)
            last_actions: List of previous actions
            agent_positions: List of agent positions
        """
        for env_idx in range(self.n_envs):
            # If no open trajectory, start one and store its initial hidden state
            if len(self.current_trajectories[env_idx]) == 0:
                self.initial_hiddens_current[env_idx] = hiddens[env_idx]

            # Add transition to the current trajectory for this env
            self.current_trajectories[env_idx].append({
                "state": states[env_idx],
                "action": actions[env_idx],
                "reward": rewards[env_idx],
                "done": dones[env_idx],
                "hidden": hiddens[env_idx],
                "last_action": last_actions[env_idx],
                "agent_position": agent_positions[env_idx],
                "move_success": move_successes[env_idx],
            })

            # If episode ended, move current trajectory to completed and reset current buffers
            if dones[env_idx]:
                self.completed_trajectories[env_idx].append(self.current_trajectories[env_idx])
                self.initial_hiddens_completed[env_idx].append(self.initial_hiddens_current[env_idx])
                self.current_trajectories[env_idx] = []
                self.initial_hiddens_current[env_idx] = None

        self.total_steps += self.n_envs

    def is_ready(self) -> bool:
        """Check if we have enough data for an update."""
        return self.total_steps >= self.total_steps_needed

    def get(self, gamma: float, gae_lambda: Optional[float] = None) -> Dict[str, Any]:
        """
        Compute returns/advantages and create batch for PPO update.

        Args:
            gamma: Discount factor
            gae_lambda: GAE lambda parameter

        Returns:
            Batch dictionary with flattened trajectories
        """
        # Collect all trajectories (completed + any in-progress)
        all_rgb = []
        all_depth = []
        all_lssg = []
        all_gssg = []
        all_actions = []
        all_rewards = []
        all_returns = []
        all_advantages = []
        all_dones = []
        all_last_actions = []
        all_agent_positions = []
        all_move_success = []
        all_initial_hiddens = []

        for env_idx in range(self.n_envs):
            # Merge completed trajectories with current in-progress trajectory (if any)
            env_trajectories = list(self.completed_trajectories[env_idx])
            env_hidden_seeds = list(self.initial_hiddens_completed[env_idx])

            if len(self.current_trajectories[env_idx]) > 0:
                env_trajectories.append(self.current_trajectories[env_idx])
                env_hidden_seeds.append(self.initial_hiddens_current[env_idx])

            if not env_trajectories:
                continue

            for traj, init_hidden in zip(env_trajectories, env_hidden_seeds):
                # Extract trajectory data
                states = [t["state"] for t in traj]
                raw_actions = [t["action"] for t in traj]
                rewards = [t["reward"] for t in traj]
                dones = [t["done"] for t in traj]
                last_actions = [t["last_action"] for t in traj]
                agent_positions = [t["agent_position"] for t in traj]
                move_success = [t.get("move_success", True) for t in traj]

                # Handle actions based on type
                actions = []
                for action in raw_actions:
                    if isinstance(action, tuple):
                        if len(action) >= 2 and action[1] == -1:
                            # Legacy actions: (action_idx, -1) -> just use action_idx
                            actions.append(action[0])
                        else:
                            # Multi-head actions: (dir_idx, len_idx[, stop_flag]) -> keep as tuple
                            actions.append(action)
                    else:
                        # Single integer action
                        actions.append(action)

                # Separate state components
                rgb_list = [s[0] for s in states]
                depth_list = [s[1] for s in states]
                lssg_list = [s[2] for s in states]
                gssg_list = [s[3] for s in states]

                # Compute returns for this trajectory
                # Note: Advantages are computed later in the agent using values
                returns = self._compute_returns(rewards, dones, gamma)

                # Add to batch
                all_rgb.extend(rgb_list)
                all_depth.extend(depth_list)
                all_lssg.extend(lssg_list)
                all_gssg.extend(gssg_list)
                all_actions.extend(actions)
                all_rewards.extend(rewards)
                all_returns.extend(returns)
                all_dones.extend(dones)
                all_last_actions.extend(last_actions)
                all_agent_positions.extend(agent_positions)
                all_move_success.extend(move_success)

                # Store initial hidden and trajectory span for agent-side slicing
                all_initial_hiddens.append({
                    "initial_hidden": init_hidden,
                    "start_idx": len(all_actions) - len(actions),
                    "length": len(actions),
                })

        # Convert to tensors
        batch = {
            "rgb": all_rgb,
            "depth": all_depth,
            "lssg": all_lssg,
            "gssg": all_gssg,
            "actions": torch.tensor(all_actions, dtype=torch.long) if all_actions else torch.tensor([]),
            "rewards": torch.tensor(all_rewards, dtype=torch.float32) if all_rewards else torch.tensor([]),
            "returns": torch.tensor(all_returns, dtype=torch.float32) if all_returns else torch.tensor([]),
            "dones": torch.tensor(all_dones, dtype=torch.bool) if all_dones else torch.tensor([]),
            "last_actions": torch.tensor(all_last_actions, dtype=torch.long) if all_last_actions else torch.tensor([]),
            "agent_positions": all_agent_positions,
            "move_success": torch.tensor(all_move_success, dtype=torch.float32) if all_move_success else torch.tensor([]),
            "trajectory_info": all_initial_hiddens,  # For LSTM hidden state handling
        }

        return batch

    def _compute_returns(self, rewards: List[float], dones: List[bool], gamma: float) -> List[float]:
        """
        Compute Monte Carlo returns for a single trajectory - VECTORIZED.

        OPTIMIZED: Uses tensor operations for efficient computation.

        Args:
            rewards: List of rewards in trajectory
            dones: List of done flags
            gamma: Discount factor

        Returns:
            List of returns (same length as rewards)
        """
        T = len(rewards)
        if T == 0:
            return []

        # Convert to tensors for vectorized operations
        rewards_t = torch.tensor(rewards, dtype=torch.float32)
        dones_t = torch.tensor(dones, dtype=torch.float32)

        returns = torch.zeros_like(rewards_t)
        running_return = 0.0

        # Backward iteration (still sequential due to temporal dependency)
        # G_t = r_t + gamma * (1 - done_t) * G_{t+1}
        for t in reversed(range(T)):
            running_return = rewards_t[t] + gamma * running_return * (1.0 - dones_t[t])
            returns[t] = running_return

        return returns.tolist()

    def clear(self):
        """Clear buffer for next rollout."""
        self.completed_trajectories = [[] for _ in range(self.n_envs)]
        self.current_trajectories = [[] for _ in range(self.n_envs)]
        self.initial_hiddens_completed = [[] for _ in range(self.n_envs)]
        self.initial_hiddens_current = [None for _ in range(self.n_envs)]
        self.total_steps = 0

    def __len__(self):
        """Return total number of transitions collected."""
        total = sum(len(traj) for trajs in self.completed_trajectories for traj in trajs)
        total += sum(len(traj) for traj in self.current_trajectories)
        return total
