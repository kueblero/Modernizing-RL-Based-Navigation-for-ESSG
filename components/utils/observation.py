class Observation:
    """
    Container for environment step results, mimicking Gym/Gymnasium API.

    Attributes:
        state: Tuple of (rgb, depth, lssg, gssg) observations
        reward: Scalar reward value
        terminated: Whether episode is done (goal reached or timeout)
        truncated: Whether episode was truncated (time limit)
        info: Dict with additional information
    """
    def __init__(self, state, reward=None, terminated=None, truncated=None, info=None):
        """
        Initialize Observation.

        Args:
            state: Tuple of (rgb, depth, lssg, gssg) observations
            reward: Reward value for this step
            terminated: Whether episode terminated naturally
            truncated: Whether episode was truncated
            info: Additional information dict
        """
        self.state = state
        self.info = info if info is not None else {}
        self.reward = reward
        self.terminated = terminated
        self.truncated = truncated
