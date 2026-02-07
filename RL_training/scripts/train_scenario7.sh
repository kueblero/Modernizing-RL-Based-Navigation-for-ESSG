#!/bin/bash
set -euo pipefail

# Scenario 7: PPO + Single-Head (504) + Depth + Curriculum
# Tests single-head architecture with 4-stage adaptive curriculum
# Curriculum: 16 → 48 → 108 → 504 actions
# Identical curriculum strategy to Scenario 4 but with single action head

echo "========================================="
echo "Scenario 7: PPO + Single-Head 504 + Curriculum"
echo "========================================="

PYTHONPATH=. python RL_training/train.py \
    --config configs/scenario7_ppo_singlehead_curriculum.json \
    --save_model

echo ""
echo "========================================="
echo "Scenario 7 Training Complete"
echo "TensorBoard logs: RL_training/runs/Scenario_7_*"
echo "========================================="
