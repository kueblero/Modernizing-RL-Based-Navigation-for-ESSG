#!/bin/bash
set -euo pipefail

# Scenario 2: PPO + Legacy Actions (16 discrete) - No IL
# Tests if PPO alone can match REINFORCE+IL performance
# Uses unified RL_training/train.py with UnifiedTrainRunner

echo "========================================="
echo "Scenario 2: PPO + Legacy - No IL"
echo "========================================="

PYTHONPATH=. python RL_training/train.py \
    --config configs/scenario2_ppo_legacy.json \
    --save_model

echo ""
echo "========================================="
echo "Scenario 2 Training Complete"
echo "TensorBoard logs: RL_training/runs/Scenario_2_*"
echo "========================================="