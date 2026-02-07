#!/bin/bash
set -euo pipefail

# Scenario 8: PPO + Multi-head Actions (24×21=504) + Curriculum - No Depth
# Tests curriculum learning with large action space WITHOUT depth input
# Uses unified RL_training/train.py with UnifiedTrainRunner

echo "========================================="
echo "Scenario 8: PPO + Multi-head + Curriculum - No Depth"
echo "========================================="

PYTHONPATH=. python RL_training/train.py \
    --config configs/scenario8_ppo_multihead_curriculum_no_depth.json \
    --save_model

echo ""
echo "========================================="
echo "Scenario 8 Training Complete"
echo "TensorBoard logs: RL_training/runs/Scenario_8_*"
echo "========================================="
