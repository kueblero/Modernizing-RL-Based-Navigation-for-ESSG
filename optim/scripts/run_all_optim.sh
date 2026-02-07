#!/bin/bash
# Run all hyperparameter optimizations sequentially

echo "========================================================================"
echo "Running ALL Hyperparameter Optimizations"
echo "========================================================================"
echo ""
echo "This will run optimizations for:"
echo "  1. REINFORCE + IL Pre-training (200 blocks)"
echo "  2. PPO + Legacy Actions (200 blocks)"
echo "  3. PPO + Curriculum Learning (500 blocks)"
echo ""
echo "Estimated time: ~24-48 hours (depends on hardware)"
echo ""

read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "Aborted."
    exit 1
fi

# Run REINFORCE + IL
echo ""
echo "========================================================================"
echo "1/3: REINFORCE + IL Pre-training"
echo "========================================================================"
bash optim/scripts/run_optim_reinforce_il.sh

# Run PPO Legacy
echo ""
echo "========================================================================"
echo "2/3: PPO + Legacy Actions"
echo "========================================================================"
bash optim/scripts/run_optim_ppo_legacy.sh

# Run PPO Curriculum
echo ""
echo "========================================================================"
echo "3/3: PPO + Curriculum Learning"
echo "========================================================================"
bash optim/scripts/run_optim_ppo_curriculum.sh

echo ""
echo "========================================================================"
echo "All optimizations completed!"
echo "========================================================================"
echo ""
echo "Results:"
echo "  - optim/best_params_reinforce_il.json"
echo "  - optim/best_params_ppo_legacy.json"
echo "  - optim/best_params_ppo_curriculum.json"
