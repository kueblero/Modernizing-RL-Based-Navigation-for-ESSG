"""
Training Manager for unified RL training.

Handles environment and agent setup for REINFORCE, A2C, and PPO agents.
"""

import os

import torch

from components.environments.random_scene_env import RandomSceneEnv
from components.environments.vec_env import VecEnv


class TrainingManager:
    """
    Training manager that handles environment and agent setup.
    Similar to VecTrialRunner but for regular training.
    """

    def __init__(self, config, n_envs, device):
        self.config = config
        self.agent_config = config["agent"]
        self.navigation_config = config["navigation"]
        self.env_config = config["env"]
        self.training_config = config.get("training", {})
        self.n_envs = n_envs
        self.device = device

        # Seed for reproducibility
        self.seed = config.get("seed", 42)

        # Environment parameters
        self.scene_numbers = list(range(1, 28))  # FloorPlan 1-27
        self.rho = self.env_config["rho"]
        self.max_actions = self.env_config.get("max_actions", self.agent_config["num_steps"])
        self.use_legacy_actions = self.env_config.get("use_legacy_actions", False)
        self.action_space_mode = self.env_config.get("action_space_mode", "multi_head")
        self.curriculum_stage = self.env_config.get("curriculum_stage", 1)
        self.stop_stagnation_steps = self.env_config.get("stop_stagnation_steps", 5)
        self.stop_stagnation_bonus = self.env_config.get("stop_stagnation_bonus", 0.02)
        # torch.compile can be toggled via config (agent.use_torch_compile) or env (DISABLE_COMPILE=1)
        self.use_torch_compile = self.agent_config.get("use_torch_compile", False)
        self.disable_compile = os.environ.get("DISABLE_COMPILE", "0") == "1"
        # Allow override via env (e.g., when copying transition tables to node-local scratch)
        self.transition_tables_path = os.environ.get(
            "TRANSITION_TABLES_PATH", self.env_config.get("transition_tables_path", "components/data/transition_tables")
        )

    def _create_vec_env(self):
        """
        Create vectorized environment with random scene sampling.

        Returns:
            VecEnv instance with n_envs parallel environments
        """

        def make_env():
            return RandomSceneEnv(
                scene_numbers=self.scene_numbers,
                rho=self.rho,
                max_actions=self.max_actions,
                use_lmdb=True,
                render=False,
                include_event_in_info=False,
                grid_size=0.1,
                transition_tables_path=self.transition_tables_path,
                curriculum_stage=self.curriculum_stage,
                use_legacy_actions=self.use_legacy_actions,
                action_space_mode=self.action_space_mode,
                stop_stagnation_steps=self.stop_stagnation_steps,
                stop_stagnation_bonus=self.stop_stagnation_bonus,
            )

        env_fns = [make_env for _ in range(self.n_envs)]
        return VecEnv(env_fns, seed=self.seed)

    def _create_agent(self):
        """
        Create agent for training.

        Returns:
            Agent instance (REINFORCE, A2C, or PPO depending on config)
        """
        # Use a dummy env for agent initialization
        dummy_env = RandomSceneEnv(
            scene_numbers=[self.scene_numbers[0]],
            rho=self.rho,
            max_actions=self.max_actions,
            use_lmdb=True,
            render=False,
            include_event_in_info=False,
            transition_tables_path=self.transition_tables_path,
            use_legacy_actions=self.use_legacy_actions,
            action_space_mode=self.action_space_mode,
            curriculum_stage=self.curriculum_stage,
            stop_stagnation_steps=self.stop_stagnation_steps,
            stop_stagnation_bonus=self.stop_stagnation_bonus,
        )

        agent_name = self.agent_config["name"].lower()
        print(f"[INFO] Initializing {agent_name.upper()} agent")

        if agent_name == "reinforce":
            from components.agents.reinforce_agent import ReinforceAgent

            agent = ReinforceAgent(
                env=dummy_env, navigation_config=self.navigation_config, agent_config=self.agent_config, device=self.device
            )

        elif agent_name == "ppo":
            from components.agents.ppo_agent import PPOAgent

            agent = PPOAgent(
                env=dummy_env,
                navigation_config=self.navigation_config,
                agent_config=self.agent_config,
                device=self.device,
                use_mixed_precision=True,
                use_torch_compile=self.use_torch_compile and not self.disable_compile,  # Selective compilation of hot paths only
            )
        else:
            raise ValueError(f"Unknown agent: {agent_name}")

        # Load IL pre-trained weights if specified
        if self.training_config.get("il_pretrain", False):
            il_checkpoint = self.training_config.get("il_checkpoint")
            if il_checkpoint and os.path.exists(il_checkpoint):
                print(f"[INFO] Loading IL pre-trained weights from: {il_checkpoint}")

                # Load checkpoint
                checkpoint = torch.load(il_checkpoint, map_location=self.device)

                # Check if policy should be reinitialized
                reinit_policy = self.training_config.get("reinit_policy_head", False)

                if reinit_policy:
                    # Only load encoder weights, keep policy randomly initialized
                    print(f"[INFO] Reinitializing policy - only loading encoder weights")
                    encoder_state = {k: v for k, v in checkpoint.items() if k.startswith("encoder.")}
                    agent.load_state_dict(encoder_state, strict=False)
                    print(f"[INFO] Components loaded from IL:")
                    print(f"  - FeatureEncoder (RGB, Depth, Action Embedding, Scene Graphs)")
                    print(f"[INFO] Components randomly initialized:")
                    print(f"  - NavigationPolicy (Core, Shared Layers, Action Heads, Value Head)")
                else:
                    # Load complete model (current behavior)
                    agent.load_state_dict(checkpoint)
                    print(f"[INFO] Loaded complete model (encoder + policy)")
            else:
                print(f"[WARNING] IL checkpoint not found: {il_checkpoint}. Starting from scratch.")

        dummy_env.close()
        return agent
