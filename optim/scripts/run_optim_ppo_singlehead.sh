#!/bin/bash
set -euo pipefail
# Run hyperparameter optimization for PPO + Single-head Actions (Scenario 6)

echo "========================================================================"
echo "Hyperparameter Optimization: PPO + Single-head - No Curriculum (504 actions)"
echo "Focused search around Scenario 3 optima (reduced trials)"
echo "========================================================================"

PYTHONPATH=. python optim/param_optimizer.py \
    --scenario ppo_singlehead_large \
    --n_trials 50 \
    --n_jobs 1

echo ""
echo "Optimization completed!"
echo "Results saved to:"
echo "  - optim/best_params_ppo_singlehead_large.json"
echo "  - optim/trials_ppo_singlehead_large.csv"
echo "  - optim/optuna_params_ppo_singlehead_large.db"
