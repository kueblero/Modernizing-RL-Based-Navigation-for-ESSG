"""
RandomSceneEnv: Environment that randomly selects a scene on each reset.

This environment is designed for efficient parallel PPO training where each
worker process can sample from a pool of scenes. Using LMDB (memory-mapped database)
allows multiple processes to share the same transition tables without duplicating RAM.
"""

import random
from typing import List

from components.environments.precomputed_thor_env import PrecomputedThorEnv


class RandomSceneEnv(PrecomputedThorEnv):
    """
    Environment that randomly selects a scene on each reset.

    This is the key component for parallel PPO training:
    - Each env in VecEnv is an instance of RandomSceneEnv
    - On reset, randomly picks a scene from the available pool
    - Uses LMDB for efficient memory-mapped transition tables
    - Multiple processes can read the same LMDB without RAM duplication

    Example:
        # Create vectorized environment with random scene selection
        from components.environments.vec_env import make_vec_env
        from components.environments.random_scene_env import RandomSceneEnv

        vec_env = make_vec_env(
            env_fn=RandomSceneEnv,
            n_envs=8,
            scene_numbers=[1, 2, 3, 4, 5],  # Pool of scenes to sample from
            use_lmdb=True,                   # Use memory-mapped DB
            rho=0.014,
            max_actions=40
        )
    """

    def __init__(
        self,
        scene_numbers: List[int],
        rho=0.014,
        render=False,
        include_event_in_info: bool = False,
        grid_size=0.1,
        transition_tables_path="components/data/transition_tables",
        max_actions=40,
        use_lmdb=True,
        curriculum_stage=1,
        use_legacy_actions=False,
        action_space_mode="multi_head",
        stop_stagnation_steps=5,
        stop_stagnation_bonus=0.02,
    ):
        """
        Args:
            scene_numbers: List of scene IDs to randomly sample from (e.g., [1, 2, 3, 4, 5])
            rho: Exploration parameter
            render: Whether to render (usually False for training)
            grid_size: Grid size for discretization
            transition_tables_path: Path to transition tables directory
            max_actions: Maximum steps per episode
            use_lmdb: Whether to use LMDB (recommended True for parallel training)
            curriculum_stage: Curriculum stage for reward computation
            use_legacy_actions: Whether to use legacy action space (16 discrete actions)
            action_space_mode: Action space mode ("legacy", "multi_head", "single_head_large")
        """
        if not scene_numbers:
            raise ValueError("scene_numbers must be a non-empty list")

        self.scene_numbers = scene_numbers

        # Initialize with first scene (will be changed on first reset)
        initial_scene = scene_numbers[0]

        super().__init__(
            rho=rho,
            scene_number=initial_scene,
            render=render,
            include_event_in_info=include_event_in_info,
            grid_size=grid_size,
            transition_tables_path=transition_tables_path,
            max_actions=max_actions,
            use_lmdb=use_lmdb,
            curriculum_stage=curriculum_stage,
            use_legacy_actions=use_legacy_actions,
            action_space_mode=action_space_mode,
            stop_stagnation_steps=stop_stagnation_steps,
            stop_stagnation_bonus=stop_stagnation_bonus,
        )

    def reset(self, random_start=True, start_position=None, start_rotation=None):
        """
        Reset environment with a randomly selected scene.

        Args:
            random_start: Whether to use random starting position (default True)
            start_position: Optional fixed starting position (if random_start=False)
            start_rotation: Optional fixed starting rotation (if random_start=False)

        Returns:
            Observation from the new episode
        """
        # Randomly select a scene from the pool
        new_scene = random.choice(self.scene_numbers)

        # Load transition table for the new scene (uses cache)
        # LMDB makes this efficient: memory-mapped, shared between processes
        if new_scene != self.scene_number:
            self.scene_number = new_scene
            self.transition_table = self._load_transition_table(
                self.scene_number,
                transition_tables_path=self.transition_tables_path,
                use_lmdb=self.use_lmdb,
                grid_size=self.grid_size,
                rot_step=15,
            )
            # Also load the ground truth graph for the new scene
            self.gt_graph = self.get_ground_truth_graph(f"FloorPlan{self.scene_number}")

        # Call parent's reset with the new scene
        return super().reset(
            scene_number=None,  # Already set above
            random_start=random_start,
            start_position=start_position,
            start_rotation=start_rotation
        )
