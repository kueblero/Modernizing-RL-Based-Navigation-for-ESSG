#!/bin/bash
set -euo pipefail
# Run hyperparameter optimization for REINFORCE + IL pre-training (Scenario 1)

echo "========================================================================"
echo "Hyperparameter Optimization: REINFORCE + IL Pre-training"
echo "========================================================================"

PYTHONPATH=. python optim/param_optimizer.py \
    --scenario reinforce_il \
    --n_trials 50 \
    --n_jobs 1

echo ""
echo "Optimization completed!"
echo "Results saved to:"
echo "  - optim/best_params_reinforce_il.json"
echo "  - optim/trials_reinforce_il.csv"
echo "  - optim/optuna_params_reinforce_il.db"
