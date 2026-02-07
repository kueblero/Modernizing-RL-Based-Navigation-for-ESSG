"""
Analyze parameter spaces in existing Optuna studies
"""
import optuna
from optuna.storages import RDBStorage
import sys
from pathlib import Path
import json

def analyze_sqlite_study(db_path: str):
    """Analyze parameter distributions in a SQLite study."""
    sqlite_url = f"sqlite:///{db_path}"

    try:
        storage = RDBStorage(url=sqlite_url)

        # Get all study names
        all_studies = optuna.get_all_study_names(storage=storage)

        if not all_studies:
            print(f"No studies found in {db_path}")
            return

        for study_name in all_studies:
            study = optuna.load_study(study_name=study_name, storage=storage)

            print(f"\n{'='*80}")
            print(f"Study: {study_name}")
            print(f"Database: {db_path}")
            print(f"{'='*80}")
            print(f"Total trials: {len(study.trials)}")

            if len(study.trials) == 0:
                print("No trials found.")
                continue

            # Get best trial
            try:
                best = study.best_trial
                print(f"Best value: {best.value:.4f}")
                print(f"\nBest params:")
                for param, value in best.params.items():
                    print(f"  {param}: {value}")
            except:
                print("No completed trials.")

            # Analyze parameter distributions from first complete trial
            complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            if complete_trials:
                first_trial = complete_trials[0]
                print(f"\n[PARAMETER DISTRIBUTIONS] (from Trial {first_trial.number}):")

                for param_name, param_value in first_trial.params.items():
                    # Get distribution from trial
                    dist = first_trial.distributions.get(param_name)
                    if dist:
                        if isinstance(dist, optuna.distributions.CategoricalDistribution):
                            print(f"  {param_name}: CategoricalDistribution")
                            print(f"    Choices: {dist.choices}")
                        elif isinstance(dist, optuna.distributions.FloatDistribution):
                            print(f"  {param_name}: FloatDistribution({dist.low}, {dist.high})")
                        elif isinstance(dist, optuna.distributions.IntDistribution):
                            print(f"  {param_name}: IntDistribution({dist.low}, {dist.high})")
                        else:
                            print(f"  {param_name}: {type(dist).__name__}")

            # Show all unique parameter sets
            param_sets = {}
            for trial in study.trials:
                if trial.state == optuna.trial.TrialState.COMPLETE:
                    for param_name in trial.params.keys():
                        if param_name not in param_sets:
                            dist = trial.distributions.get(param_name)
                            if isinstance(dist, optuna.distributions.CategoricalDistribution):
                                param_sets[param_name] = tuple(dist.choices)

            if param_sets:
                print(f"\n[ALL CATEGORICAL PARAMETERS]:")
                for param_name, choices in param_sets.items():
                    print(f"  {param_name}: {choices}")

    except Exception as e:
        print(f"Error analyzing {db_path}: {e}")


def analyze_current_config(scenario: str):
    """Show what parameter space the current code would use."""
    print(f"\n{'='*80}")
    print(f"CURRENT CONFIG FOR: {scenario}")
    print(f"{'='*80}")

    # Import to get search space
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from optim.param_optimizer import get_param_search_space

    # Determine settings
    use_legacy_actions = scenario in ["ppo_legacy", "reinforce_il"]
    curriculum_enabled = "curriculum" in scenario
    agent_name = "reinforce" if "reinforce" in scenario else "ppo"

    # Get search space
    search_space = get_param_search_space(agent_name, use_legacy_actions, curriculum_enabled)

    print(f"Agent: {agent_name}")
    print(f"Legacy actions: {use_legacy_actions}")
    print(f"Curriculum: {curriculum_enabled}")
    print(f"\n[PARAMETER SEARCH SPACE]:")
    for param, choices in search_space.items():
        print(f"  {param}: {choices}")


if __name__ == "__main__":
    print("="*80)
    print("OPTUNA STUDY PARAMETER ANALYZER")
    print("="*80)

    # Analyze local SQLite databases
    optim_dir = Path(__file__).parent
    sqlite_dbs = list(optim_dir.glob("optuna_params_*.db"))

    if sqlite_dbs:
        print(f"\nFound {len(sqlite_dbs)} SQLite databases:")
        for db in sqlite_dbs:
            analyze_sqlite_study(str(db))
    else:
        print("\nNo SQLite databases found in optim/")

    # If scenario provided, show current config
    if len(sys.argv) > 1:
        scenario = sys.argv[1]
        analyze_current_config(scenario)

        print(f"\n{'='*80}")
        print("COMPARISON")
        print(f"{'='*80}")
        print("Check if the parameter choices above match!")
        print("If they differ, you need to either:")
        print("  1. Delete the study and start fresh")
        print("  2. Use a different study_name")
        print("  3. Revert your code to match the old parameters")