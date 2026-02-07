import json
from argparse import ArgumentParser
from pathlib import Path

import torch

from ImitationLearning.dataset.il_dataset import ImitationLearningDataset
from ImitationLearning.runner.il_train_runner import ILTrainRunner
from components.agents.reinforce_agent import ReinforceAgent
class _ILDummyLegacyEnv:
    """
    Minimal dummy environment to provide legacy action space dimensions for IL training.
    No stepping or observations are required; only the action space shape matters.
    """

    def __init__(self):
        self.use_legacy_actions = True
        self.action_angles = [0, 45, 90, 135, 180, 225, 270, 315]
        self.action_lengths = [0.0, 0.3]
        self.legacy_actions = [(a, l) for a in self.action_angles for l in self.action_lengths]

    def get_action_space_dims(self):
        return {"num_actions": len(self.legacy_actions)}


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--weights_save_folder", type=str, default=None, help="Directory for saving model weights. If None, a default folder is used."
    )
    parser.add_argument(
        "--checkpoint_path_load", type=str, default=None, help="Path to load model weights from. If None, training starts from scratch."
    )
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training.")
    parser.add_argument(
        "--scenario_config",
        type=str,
        default=None,
        help="Path to ablation scenario config JSON (default: configs/scenario0_reinforce_il.json).",
    )
    args = parser.parse_args()

    # --- Define all relevant paths ---
    project_root = Path(__file__).parent.parent.resolve()
    default_scenario_config_path = project_root / "experiments" / "ablation_study" / "configs" / "scenario0_reinforce_il.json"
    scenario_config_path = Path(args.scenario_config) if args.scenario_config else default_scenario_config_path
    data_dir = project_root / "components" / "data" / "il_dataset"
    default_weights_folder = project_root / "components" / "data" / "model_weights" / "il"
    weights_save_folder = Path(args.weights_save_folder) if args.weights_save_folder else default_weights_folder
    weights_save_folder.mkdir(parents=True, exist_ok=True)

    # --- Load configs ---
    if not scenario_config_path.exists():
        raise FileNotFoundError(f"Scenario config not found: {scenario_config_path}")
    with open(scenario_config_path, "r") as f:
        scenario_config = json.load(f)

    navigation_config = scenario_config["navigation"]
    agent_config = scenario_config["agent"]

    # Override agent name to "reinforce" for IL training (no value head needed)
    agent_config["name"] = "reinforce"

    print("\nLoaded navigation config:")
    for k, v in navigation_config.items():
        print(f"  {k}: {v}")

    # --- Prepare dataset ---
    dataset = ImitationLearningDataset(data_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Create dummy environment for agent initialization ---
    # The ReinforceAgent needs an env, but we only use it for action-space initialization
    dummy_env = _ILDummyLegacyEnv()

    # --- Create agent (use ReinforceAgent for IL training) ---
    # This ensures 100% compatibility with RL training later
    agent = ReinforceAgent(env=dummy_env, navigation_config=navigation_config, agent_config=agent_config, device=device)

    print(f"\n[INFO] Created ReinforceAgent for IL training")
    print(f"[INFO] Action space: Legacy (16 discrete actions)")
    print(f"[INFO] This ensures full compatibility with RL training")

    # --- Optionally load weights from checkpoint ---
    if args.checkpoint_path_load:
        checkpoint_path = Path(args.checkpoint_path_load)
        if checkpoint_path.exists():
            agent.load_state_dict(torch.load(str(checkpoint_path), map_location=device))
            print(f"Weights loaded from: {checkpoint_path}")
        else:
            print(f"WARNING: Checkpoint not found at {checkpoint_path}. Training will start from scratch.")

    # --- Start training ---
    print(f"\n[INFO] Starting IL training for {args.epochs} epochs")
    print(f"[INFO] Weights will be saved in: {weights_save_folder}\n")

    runner = ILTrainRunner(agent, dataset, device=device, lr=0.0001, batch_size=args.batch_size)
    runner.run(num_epochs=args.epochs, save_folder=str(weights_save_folder))


if __name__ == "__main__":
    main()
