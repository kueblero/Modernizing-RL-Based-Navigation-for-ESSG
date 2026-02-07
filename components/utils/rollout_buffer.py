# In utils/rollout_buffer.py

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RolloutBuffer:
    def __init__(self, n_steps):
        self.n_steps = n_steps
        # Per-step data
        self.state_rgb = []
        self.state_depth = []
        self.state_lssg = []
        self.state_gssg = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.last_actions = []
        self.agent_positions = []

        # Hidden states at the beginning of the rollout
        self.initial_lssg_hidden = None
        self.initial_gssg_hidden = None
        self.initial_policy_hidden = None

        self.is_first_add = True

    def clear(self):
        # Per-step data
        self.state_rgb = []
        self.state_depth = []
        self.state_lssg = []
        self.state_gssg = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.last_actions = []
        self.agent_positions = []

        # Hidden states at the beginning of the rollout
        self.initial_lssg_hidden = None
        self.initial_gssg_hidden = None
        self.initial_policy_hidden = None

        self.is_first_add = True

    def add(self, state, action, reward, done, hiddens, last_action, agent_position):
        """
        Add a single step transition to the buffer.

        Args:
            state: Tuple of (rgb, depth, lssg, gssg) observations
            action: Action taken
            reward: Reward received
            done: Whether episode is done
            hiddens: Tuple of (lssg_hidden, gssg_hidden, policy_hidden) from this step
            last_action: Previous action (for embedding)
            agent_position: Agent's position for visualization

        Returns:
            None
        """
        # hiddens is a tuple of (lssg_hidden, gssg_hidden, policy_hidden)
        if self.is_first_add:
            # Store initial hidden states on the very first add after a clear
            self.initial_lssg_hidden = hiddens[0]
            self.initial_gssg_hidden = hiddens[1]
            self.initial_policy_hidden = hiddens[2]
            self.is_first_add = False

        s_rgb, s_depth, s_lssg, s_gssg = state
        self.state_rgb.append(s_rgb)
        self.state_depth.append(s_depth)
        self.state_lssg.append(s_lssg)
        self.state_gssg.append(s_gssg)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.last_actions.append(last_action)
        self.agent_positions.append(agent_position)

    def add_batch(self, states, actions, rewards, dones, hiddens, last_actions, agent_pos):
        """
        Adds a whole batch of transitions to the buffer at once.

        Args:
            states: List of state tuples (rgb, depth, lssg, gssg)
            actions: List of actions
            rewards: List of rewards
            dones: List of done flags
            hiddens: List of hidden state tuples (one for first step)
            last_actions: List of previous actions
            agent_pos: List of agent positions

        Returns:
            None
        """
        if self.is_first_add and hiddens:
            self.initial_lssg_hidden = hiddens[0][0]
            self.initial_gssg_hidden = hiddens[0][1]
            self.initial_policy_hidden = hiddens[0][2]
            self.is_first_add = False

        if states:
            states_rgb, states_depth, states_lssg, states_gssg = zip(*states)
            self.state_rgb.extend(states_rgb)
            self.state_depth.extend(states_depth)
            self.state_lssg.extend(states_lssg)
            self.state_gssg.extend(states_gssg)

        self.actions.extend(actions)
        self.rewards.extend(rewards)
        self.dones.extend(dones)
        self.last_actions.extend(last_actions)
        self.agent_positions.extend(agent_pos)

    def is_ready(self):
        """Check if buffer has collected enough steps for update."""
        return len(self.rewards) >= self.n_steps

    def compute_returns(self, gamma):
        """
        Compute discounted cumulative returns (G_t = r_t + gamma * r_{t+1} + ...).

        Args:
            gamma: Discount factor

        Returns:
            List of returns for each step
        """
        returns = []
        G = 0
        for reward, done in zip(reversed(self.rewards), reversed(self.dones)):
            if done:
                G = 0
            G = reward + gamma * G
            returns.insert(0, G)
        return returns

    def get(self, gamma):
        """
        Get batch of collected transitions for training.

        Args:
            gamma: Discount factor for computing returns

        Returns:
            dict with keys: rgb, depth, lssg, gssg, actions, returns, rewards, dones, last_actions, agent_positions, and initial hidden states
        """
        returns = self.compute_returns(gamma)

        batch = {
            "rgb": self.state_rgb,
            "depth": self.state_depth,
            "lssg": self.state_lssg,
            "gssg": self.state_gssg,
            "actions": (
                torch.stack(self.actions) if isinstance(self.actions[0], torch.Tensor) else torch.tensor(self.actions, dtype=torch.long)
            ),
            "returns": torch.tensor(returns, dtype=torch.float32),
            "rewards": torch.tensor(self.rewards, dtype=torch.float32),  # Add raw rewards for GAE
            "dones": torch.tensor(self.dones, dtype=torch.bool),
            "last_actions": (
                torch.stack(self.last_actions)
                if isinstance(self.last_actions[0], torch.Tensor)
                else torch.tensor(self.last_actions, dtype=torch.long)
            ),
            "agent_positions": self.agent_positions,
            "initial_lssg_hidden": self.initial_lssg_hidden,
            "initial_gssg_hidden": self.initial_gssg_hidden,
            "initial_policy_hidden": self.initial_policy_hidden,
        }
        return batch
