#!/bin/bash
set -euo pipefail

# Scenario 3: PPO + Multi-head Actions (24×21=504) - No Curriculum
# Tests large action space without curriculum learning
# Uses unified RL_training/train.py with UnifiedTrainRunner

echo "========================================="
echo "Scenario 3: PPO + Multi-head - No Curriculum"
echo "========================================="

PYTHONPATH=. python RL_training/train.py \
    --config configs/scenario3_ppo_multihead_no_curriculum.json \
    --save_model

echo ""
echo "========================================="
echo "Scenario 3 Training Complete"
echo "TensorBoard logs: RL_training/runs/Scenario_3_*"
echo "========================================="