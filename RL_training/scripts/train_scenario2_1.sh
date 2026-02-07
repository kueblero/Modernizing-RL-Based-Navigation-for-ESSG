#!/bin/bash
set -euo pipefail

# Scenario 2.1: PPO + Legacy Actions (16 discrete) - No Depth
# Ablation of scenario2 without depth input
# Uses unified RL_training/train.py with UnifiedTrainRunner

echo "========================================="
echo "Scenario 2.1: PPO + Legacy - No Depth"
echo "========================================="

PYTHONPATH=. python RL_training/train.py \
    --config configs/scenario2_1_ppo_legacy_no_depth.json \
    --save_model

echo ""
echo "========================================="
echo "Scenario 2.1 Training Complete"
echo "TensorBoard logs: RL_training/runs/Scenario_2_1_*"
echo "========================================="
