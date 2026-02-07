#!/usr/bin/env bash
# Direct execution script for multi-seed training on local/remote machines
# Use this for machines without SLURM or for local development
#
# Usage:
#   ./run_multi_seed_local.sh SCENARIO NUM_SEEDS [START_SEED] [N_ENVS]
#
# Examples:
#   ./run_multi_seed_local.sh 2 5              # Train 5 seeds (0-4) for scenario 2 (with depth)
#   ./run_multi_seed_local.sh 2_1 5            # Train 5 seeds for scenario 2.1 (no depth)
#   ./run_multi_seed_local.sh 2 3 0 8          # Train seeds 0-2 with 8 parallel envs
#   ./run_multi_seed_local.sh 2 2 3 8          # Train seeds 3-4 with 8 parallel envs
#   ./run_multi_seed_local.sh baseline 5       # Train baseline (no depth) scenario
#   ./run_multi_seed_local.sh 5 5              # Train scenario 5 (multi-head, no depth)
#
# Available scenarios:
#   0       - REINFORCE with IL pretraining (main baseline)
#   1       - REINFORCE without IL pretraining
#   2       - PPO legacy actions (with depth)
#   2_1     - PPO legacy actions (no depth)
#   3       - PPO multi-head (no curriculum, with depth)
#   4       - PPO multi-head + curriculum (with depth)
#   5       - PPO multi-head (no depth, no curriculum)
#   6       - PPO single-head (no curriculum, with depth)
#   7       - PPO single-head + curriculum (with depth)
#   8       - PPO multi-head + curriculum (no depth)
#   baseline - Li et al. baseline (no depth, with IL)

set -euo pipefail

# Parse arguments
SCENARIO="${1:-1}"
NUM_SEEDS="${2:-5}"
START_SEED="${3:-0}"
N_ENVS="${4:-}"

# Find config file based on scenario
if [ "$SCENARIO" = "baseline" ]; then
  CONFIG_PATH="configs/baseline_scenario_reinforce_nodepth_il.json"
else
  CONFIG_GLOB="configs/scenario${SCENARIO}_*.json"
  CONFIG_PATH="$(ls $CONFIG_GLOB 2>/dev/null | head -1 || true)"
fi

if [ -z "${CONFIG_PATH}" ] || [ ! -f "${CONFIG_PATH}" ]; then
  echo "ERROR: No config file found for scenario '${SCENARIO}'"
  echo ""
  echo "Available scenarios:"
  echo "  0        - REINFORCE with IL (scenario0_reinforce_il.json)"
  echo "  1        - REINFORCE without IL (scenario1_reinforce_no_il.json)"
  echo "  2        - PPO legacy actions (with depth)"
  echo "  2_1      - PPO legacy actions (no depth)"
  echo "  3        - PPO multi-head (no curriculum, with depth)"
  echo "  4        - PPO multi-head + curriculum (with depth)"
  echo "  5        - PPO multi-head (no depth, no curriculum)"
  echo "  6        - PPO single-head (no curriculum, with depth)"
  echo "  7        - PPO single-head + curriculum (with depth)"
  echo "  8        - PPO multi-head + curriculum (no depth)"
  echo "  baseline - Li et al. baseline, no depth (baseline_scenario_reinforce_nodepth_il.json)"
  exit 1
fi

echo "=========================================="
echo "Multi-Seed RL Training (Local Execution)"
echo "=========================================="
echo "Scenario:     $SCENARIO"
echo "Config:       $CONFIG_PATH"
echo "Num seeds:    $NUM_SEEDS"
echo "Start seed:   $START_SEED"
echo "Seed range:   $START_SEED to $((START_SEED + NUM_SEEDS - 1))"
if [ -n "$N_ENVS" ]; then
  echo "Parallel envs: $N_ENVS"
else
  echo "Parallel envs: auto (from config or CPU count)"
fi
echo "Started at:   $(date)"
echo "=========================================="

# Build Python command
PYTHON_CMD="PYTHONPATH=. python -u RL_training/multi_seed_train_eval.py \
  --config $CONFIG_PATH \
  --num_seeds $NUM_SEEDS \
  --start_seed $START_SEED \
  --save_model"

if [ -n "$N_ENVS" ]; then
  PYTHON_CMD="$PYTHON_CMD --n_envs $N_ENVS"
fi

# Set environment variables for optimal performance
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Run training
echo ""
echo "Starting training..."
echo ""

eval $PYTHON_CMD

echo ""
echo "=========================================="
echo "Training completed at: $(date)"
echo "Results saved to RL_training/runs/"
echo "=========================================="
