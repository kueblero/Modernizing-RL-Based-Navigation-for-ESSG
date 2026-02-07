#!/usr/bin/env python3
"""
Analyze training runs to check for missing seeds and config consistency.

This script:
1. Groups runs by scenario
2. Checks if all runs from the same scenario have identical configs
3. Identifies which seeds are missing to have 5 complete runs per scenario
4. Shows which seed indices are missing based on the seed generation pattern
"""

import json
import os
import sys
from collections import defaultdict
from pathlib import Path


def get_seed_from_index(index, master_seed=42):
    """
    Generate seed from index using the same pattern as multi_seed_train_eval.py.

    This matches the generate_seeds() function:
        np.random.seed(master_seed)
        all_seeds = np.random.randint(1, 10000, size=total_seeds + start_seed).tolist()
        return all_seeds[start_seed:start_seed + total_seeds]

    For master_seed=42, the sequence is:
        Index 0: 7271
        Index 1: 861
        Index 2: 5391
        Index 3: 5192
        Index 4: 5735
        ...
    """
    import numpy as np

    np.random.seed(master_seed)
    all_seeds = np.random.randint(1, 10000, size=index + 1).tolist()
    return all_seeds[index]


def extract_scenario_and_seed(run_dir_name):
    """
    Extract scenario name and seed from run directory name.
    Expected format: {scenario}_seed_{seed}_{timestamp}
    """
    parts = run_dir_name.split("_seed_")
    if len(parts) != 2:
        return None, None

    scenario_name = parts[0]
    seed_and_timestamp = parts[1].split("_")
    if len(seed_and_timestamp) < 1:
        return scenario_name, None

    try:
        seed = int(seed_and_timestamp[0])
        return scenario_name, seed
    except ValueError:
        return scenario_name, None


def get_training_event_file(run_dir):
    """
    Find the TensorBoard event file in the run directory.
    Returns the path to the event file, or None if not found.
    """
    event_files = list(run_dir.glob("events.out.tfevents.*"))
    if not event_files:
        return None
    # Return the most recent event file
    return max(event_files, key=lambda p: p.stat().st_mtime)


def load_config_from_run(run_dir):
    """
    Load config from a run directory's TensorBoard event file.
    """
    from tensorboard.backend.event_processing import event_accumulator

    event_path = get_training_event_file(run_dir)
    if event_path is None:
        return None

    try:
        ea = event_accumulator.EventAccumulator(str(event_path))
        ea.Reload()
        tags = ea.Tags()

        config_tag = "full_config/text_summary"
        if config_tag not in tags.get("tensors", []):
            return None

        tensor_events = ea.Tensors(config_tag)
        if not tensor_events:
            return None

        raw = tensor_events[0].tensor_proto.string_val[0]
        cfg_text = raw.decode("utf-8")
        config = json.loads(cfg_text)
        return config
    except Exception as e:
        return None


def compare_configs(config1, config2):
    """
    Compare two configs, ignoring expected differences.
    Returns (is_equal, differences) tuple.
    """
    # Keys to ignore (normally differ between runs)
    ignore_keys = {
        "seed",
        "run_name",
        "timestamp",
        "device",
        "num_workers",
        "run_id",
        "output_dir",
        "wandb_run_id",
        "wandb_run_name",
        # Parameters that can differ due to config changes
        "agent.adaptive_entropy",
        "agent.entropy_coef_max",
        "agent.target_entropy",
        "training.save_interval",
    }

    # Recursive function to compare nested dictionaries
    def compare_dicts(d1, d2, path=""):
        diffs = []
        all_keys = set(d1.keys()) | set(d2.keys())

        for key in all_keys:
            full_path = f"{path}.{key}" if path else key

            # Check both the key and the full_path against ignore_keys
            if key in ignore_keys or full_path in ignore_keys:
                continue

            if key not in d1:
                diffs.append(f"  - {full_path}: missing in first config")
            elif key not in d2:
                diffs.append(f"  - {full_path}: missing in second config")
            elif isinstance(d1[key], dict) and isinstance(d2[key], dict):
                diffs.extend(compare_dicts(d1[key], d2[key], full_path))
            elif d1[key] != d2[key]:
                diffs.append(f"  - {full_path}: {d1[key]} != {d2[key]}")

        return diffs

    diffs = compare_dicts(config1, config2)
    return len(diffs) == 0, diffs


def find_config_for_scenario(scenario_name):
    """
    Find the config file for a scenario in configs/ directory.
    """
    configs_dir = Path("configs")
    if not configs_dir.exists():
        return None

    # Try exact match
    config_path = configs_dir / f"{scenario_name}.json"
    if config_path.exists():
        with open(config_path, "r") as f:
            return json.load(f)

    # Try fuzzy match
    for config_file in configs_dir.glob("*.json"):
        if scenario_name in config_file.stem:
            with open(config_file, "r") as f:
                return json.load(f)

    return None


def analyze_runs(runs_dir="RL_training/runs"):
    """
    Main analysis function.
    """
    runs_path = Path(runs_dir)
    if not runs_path.exists():
        print(f"[ERROR] Runs directory not found: {runs_dir}")
        return

    # Group runs by scenario
    scenarios = defaultdict(list)

    for run_dir in sorted(runs_path.iterdir()):
        if not run_dir.is_dir():
            continue

        scenario_name, seed = extract_scenario_and_seed(run_dir.name)
        if scenario_name is None:
            print(f"[WARN] Could not parse run directory: {run_dir.name}")
            continue

        config = load_config_from_run(run_dir)

        scenarios[scenario_name].append({"dir": run_dir, "seed": seed, "config": config})

    if not scenarios:
        print("[ERROR] No runs found")
        return

    print("=" * 80)
    print("TRAINING RUNS ANALYSIS")
    print("=" * 80)
    print()

    # Analyze each scenario
    for scenario_name in sorted(scenarios.keys()):
        runs = scenarios[scenario_name]
        seeds = [r["seed"] for r in runs if r["seed"] is not None]

        print(f"📊 Scenario: {scenario_name}")
        print(f"   Runs found: {len(runs)}")
        print(f"   Seeds: {sorted(seeds)}")

        # Check config consistency
        configs = [r["config"] for r in runs if r["config"] is not None]
        if len(configs) > 1:
            reference_config = configs[0]
            is_consistent = True
            for i, run in enumerate(runs):
                if run["config"]:
                    is_equal, diffs = compare_configs(reference_config, run["config"])
                    if not is_equal:
                        is_consistent = False
                        print(f"   ❌ WARNING: {run['dir'].name} has different config:")
                        for diff in diffs:
                            print(f"      {diff}")

            if is_consistent:
                print("   ✅ All configs are consistent")
        elif len(configs) == 1:
            print("   ℹ️  Only one config found (cannot compare)")
        else:
            print("   ⚠️  No configs found in run directories")

        # Check for missing seeds
        target_count = 5
        if len(seeds) < target_count:
            print(f"   ⚠️  Missing {target_count - len(seeds)} seed(s) to reach {target_count} runs")

            # Find which seed indices are missing
            # Assume seeds follow a pattern or are from a known set
            expected_seeds = [get_seed_from_index(i) for i in range(target_count)]
            missing_seeds = [s for s in expected_seeds if s not in seeds]

            if missing_seeds:
                print("   🔍 Missing seeds (standard pattern):")
                for missing_seed in missing_seeds:
                    seed_index = expected_seeds.index(missing_seed)
                    print(f"      - Seed {missing_seed} (index {seed_index})")
            else:
                # Seeds don't follow standard pattern, list what we have and suggest random seeds
                print("   🔍 Seeds don't follow standard pattern. Current seeds:", sorted(seeds))
                print(f"      Suggestion: Generate {target_count - len(seeds)} more random seed(s)")
        elif len(seeds) == target_count:
            print(f"   ✅ Complete: {target_count}/{target_count} seeds")
        else:
            print(f"   ⚠️  Extra runs: {len(seeds)}/{target_count} seeds")

            # Find which seeds are expected and which are extra
            expected_seeds = [get_seed_from_index(i) for i in range(target_count)]

            # Check for duplicate seeds
            seed_counts = {}
            for seed in seeds:
                seed_counts[seed] = seed_counts.get(seed, 0) + 1

            duplicates = [seed for seed, count in seed_counts.items() if count > 1]
            if duplicates:
                print(f"   🔄 Duplicate seeds found:")
                for dup_seed in sorted(duplicates):
                    dup_runs = [r["dir"].name for r in runs if r["seed"] == dup_seed]
                    print(f"      - Seed {dup_seed} appears {seed_counts[dup_seed]} times:")
                    for run_name in dup_runs[1:]:  # Keep first, mark rest as duplicates
                        print(f"        ❌ DELETE: {run_name}")

            # Find extra seeds (not in expected pattern)
            extra_seeds = [s for s in seeds if s not in expected_seeds]
            if extra_seeds:
                print(f"   ➕ Extra seeds (not in standard pattern):")
                for extra_seed in sorted(set(extra_seeds)):
                    extra_runs = [r["dir"].name for r in runs if r["seed"] == extra_seed]
                    for run_name in extra_runs:
                        print(f"      ❌ DELETE: {run_name} (seed {extra_seed})")

            # Show which expected seeds we have
            present_expected_seeds = [s for s in expected_seeds if s in seeds]
            if present_expected_seeds:
                print(f"   ✅ Present expected seeds: {sorted(present_expected_seeds)}")

        # Check if checkpoints exist
        checkpoints_info = []
        for run in runs:
            checkpoint_dir = run["dir"] / "checkpoints"
            if checkpoint_dir.exists():
                best_model = checkpoint_dir / "best_model.pth"
                final_model = checkpoint_dir / "final_model.pth"

                if best_model.exists():
                    checkpoints_info.append(f"best")
                elif final_model.exists():
                    checkpoints_info.append(f"final")
                else:
                    checkpoints_info.append(f"none")
            else:
                checkpoints_info.append(f"missing")

        checkpoint_summary = ", ".join(checkpoints_info) if checkpoints_info else "none"
        print(f"   📦 Checkpoints: {checkpoint_summary}")

        print()

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    total_scenarios = len(scenarios)
    complete_scenarios = sum(1 for runs in scenarios.values() if len([r for r in runs if r["seed"]]) >= 5)
    incomplete_scenarios = total_scenarios - complete_scenarios

    print(f"Total scenarios: {total_scenarios}")
    print(f"Complete (≥5 seeds): {complete_scenarios}")
    print(f"Incomplete (<5 seeds): {incomplete_scenarios}")
    print()

    if incomplete_scenarios > 0:
        print("🚀 TO DO: Run missing seeds for these scenarios:")
        for scenario_name, runs in sorted(scenarios.items()):
            seeds = [r["seed"] for r in runs if r["seed"] is not None]
            if len(seeds) < 5:
                expected_seeds = [get_seed_from_index(i) for i in range(5)]
                missing_seeds = [s for s in expected_seeds if s not in seeds]
                print(f"   - {scenario_name}: seeds {missing_seeds}")
    else:
        print("✅ All scenarios have at least 5 seeds!")


if __name__ == "__main__":
    # Set working directory to project root
    desired_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
    current_directory = os.getcwd()

    if current_directory != desired_directory:
        os.chdir(desired_directory)
        sys.path.append(desired_directory)

    analyze_runs()
