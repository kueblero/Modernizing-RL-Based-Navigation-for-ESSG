#!/bin/bash
set -euo pipefail
# Run hyperparameter optimization for PPO + Multi-head Actions (Scenario 3)

echo "========================================================================"
echo "Hyperparameter Optimization: PPO + Multi-head - No Curriculum (504 actions)"
echo "========================================================================"

PYTHONPATH=. python optim/param_optimizer.py \
    --scenario ppo_multihead \
    --n_trials 50 \
    --n_jobs 1

echo ""
echo "Optimization completed!"
echo "Results saved to:"
echo "  - optim/best_params_ppo_multihead.json"
echo "  - optim/trials_ppo_multihead.csv"
echo "  - optim/optuna_params_ppo_multihead.db"
