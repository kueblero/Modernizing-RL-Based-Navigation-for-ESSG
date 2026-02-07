#!/bin/bash
set -euo pipefail

# Scenario 1: REINFORCE + Legacy Actions (16 discrete) - No IL Pre-training
# REINFORCE baseline without imitation learning pretraining

echo "========================================="
echo "Scenario 1: REINFORCE + Legacy (No IL)"
echo "========================================="

PYTHONPATH=. python RL_training/train.py \
    --config configs/scenario1_reinforce_no_il.json \
    --save_model

echo ""
echo "========================================="
echo "Scenario 1 Training Complete"
echo "TensorBoard logs: RL_training/runs/Scenario_1_*"
echo "========================================="