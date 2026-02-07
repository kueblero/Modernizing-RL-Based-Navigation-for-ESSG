#!/bin/bash
set -euo pipefail
# Run hyperparameter optimization for PPO + Legacy Actions (Scenario 2)

echo "========================================================================"
echo "Hyperparameter Optimization: PPO + Legacy Actions (16 discrete)"
echo "========================================================================"

PYTHONPATH=. python optim/param_optimizer.py \
    --scenario ppo_legacy \
    --n_trials 50 \
    --n_jobs 1

echo ""
echo "Optimization completed!"
echo "Results saved to:"
echo "  - optim/best_params_ppo_legacy.json"
echo "  - optim/trials_ppo_legacy.csv"
echo "  - optim/optuna_params_ppo_legacy.db"
