#!/bin/bash
set -euo pipefail

# Scenario 4: PPO + Multi-head Actions (24×21=504) + Curriculum
# Tests if curriculum learning helps with large action space
# Uses unified RL_training/train.py with UnifiedTrainRunner

echo "========================================="
echo "Scenario 4: PPO + Multi-head + Curriculum"
echo "========================================="

PYTHONPATH=. python RL_training/train.py \
    --config configs/scenario4_ppo_multihead_curriculum.json \
    --save_model

echo ""
echo "========================================="
echo "Scenario 4 Training Complete"
echo "TensorBoard logs: RL_training/runs/Scenario_4_*"
echo "========================================="
