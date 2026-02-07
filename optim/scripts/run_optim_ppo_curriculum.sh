#!/bin/bash
set -euo pipefail
# Run hyperparameter optimization for PPO + Curriculum Learning (Scenario 4)

echo "========================================================================"
echo "Hyperparameter Optimization: PPO + Multi-head + Curriculum (504 actions)"
echo "========================================================================"

PYTHONPATH=. python optim/param_optimizer.py \
    --scenario ppo_curriculum \
    --n_trials 50 \
    --n_jobs 1

echo ""
echo "Optimization completed!"
echo "Results saved to:"
echo "  - optim/best_params_ppo_curriculum.json"
echo "  - optim/trials_ppo_curriculum.csv"
echo "  - optim/optuna_params_ppo_curriculum.db"
