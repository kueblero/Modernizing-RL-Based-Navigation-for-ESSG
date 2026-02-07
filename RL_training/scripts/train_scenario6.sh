#!/bin/bash
set -euo pipefail

# Scenario 6: PPO + Single-Head (504) + Depth - No Curriculum
# Tests single-head architecture with large action space (504 actions)
# No curriculum learning - full action space from start
# Comparable to Scenario 3 (Multi-Head No CL)

echo "========================================="
echo "Scenario 6: PPO + Single-Head 504 - No Curriculum"
echo "========================================="

PYTHONPATH=. python RL_training/train.py \
    --config configs/scenario6_ppo_singlehead_no_curriculum.json \
    --save_model

echo ""
echo "========================================="
echo "Scenario 6 Training Complete"
echo "TensorBoard logs: RL_training/runs/Scenario_6_*"
echo "========================================="
