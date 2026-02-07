#!/bin/bash
set -euo pipefail

# Scenario 0: REINFORCE + Legacy Actions (16 discrete) + IL Pre-training
# This is the main baseline approach with IL pre-training
# Uses unified RL_training/train.py with UnifiedTrainRunner

echo "========================================="
echo "Scenario 0: REINFORCE + Legacy + IL"
echo "========================================="

PYTHONPATH=. python RL_training/train.py \
    --config configs/scenario0_reinforce_il.json \
    --save_model

echo ""
echo "========================================="
echo "Scenario 0 Training Complete"
echo "TensorBoard logs: RL_training/runs/Scenario_0_*"
echo "========================================="