# Hyperparameter Optimization for RL Agents

Tools for optimizing hyperparameters for REINFORCE and PPO agents using Optuna.

## Overview

Five optimization scenarios are supported:

1. **REINFORCE + IL Pre-training** (`reinforce_il`)
   - Config: `configs/scenario0_reinforce_il.json`
   - Optimizes: `alpha`, `gamma`, `collision_loss_coef`
   - Training: 400 blocks, pruning after 200 blocks

2. **PPO + Legacy Actions** (`ppo_legacy`)
   - Config: `configs/scenario2_ppo_legacy.json`
   - Optimizes: `alpha`, `clip_epsilon`, `entropy_coef`, `gae_lambda`, `collision_loss_coef`
   - Training: 400 blocks, pruning after 200 blocks

3. **PPO + Multi-head - No Curriculum** (`ppo_multihead`)
   - Config: `configs/scenario3_ppo_multihead_no_curriculum.json`
   - Optimizes: `alpha`, `clip_epsilon`, `entropy_coef`, `gae_lambda`, `collision_loss_coef`
   - Training: 400 blocks, pruning after 200 blocks

4. **PPO + Multi-head + Curriculum** (`ppo_multihead_curriculum`)
   - Config: `configs/scenario4_ppo_multihead_curriculum.json`
   - Optimizes baseline parameters + curriculum parameters (`min_stage_blocks`, `plateau_threshold`, `entropy_boost_factor`, `entropy_decay_blocks`, `force_promotion_blocks`)
   - Training: 500 blocks, pruning after 350 blocks

5. **PPO + Single-head - No Curriculum** (`ppo_singlehead`)
   - Config: `configs/scenario6_ppo_singlehead_no_curriculum.json`
   - Optimizes: `alpha`, `clip_epsilon`, `entropy_coef`, `gae_lambda`, `collision_loss_coef`
   - Action space: 504 flat actions (no multi-head decomposition)
   - Training: 400 blocks, pruning after 200 blocks

## Objective Metric

The optimization maximizes a penalized score:

- **Mean score for terminated episodes** (successful STOP)
- **Penalized by:**
  - collision rate
  - truncation rate

If no terminated episodes are available, it falls back to the mean score of all episodes.

## Hyperparameter Search Spaces

Search spaces use categorical values for stability and interpretability.

### REINFORCE
Search space: **6 x 4 x 3 = 72 combinations**

| Parameter | Values |
|-----------|--------|
| `alpha` | [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3] |
| `gamma` | [0.90, 0.95, 0.97, 0.99] |
| `collision_loss_coef` | [0.1, 0.2, 0.3] |

### PPO
Search space (legacy): **4 x 5 x 5 x 3 x 4 = 1,200**
Search space (multi-head): **4 x 5 x 5 x 3 x 4 = 1,200**

| Parameter | Values | Notes |
|-----------|--------|-------|
| `alpha` | [1e-5, 3e-5, 5e-5, 1e-4] | Learning rate |
| `clip_epsilon` | [0.1, 0.15, 0.2, 0.25, 0.3] | PPO clipping |
| `gae_lambda` | [0.90, 0.93, 0.95, 0.97, 0.98] | GAE lambda |
| `collision_loss_coef` | [0.1, 0.2, 0.3] | Collision head weight |
| `entropy_coef` (legacy) | [0.01, 0.05, 0.10, 0.15] | 16 actions |
| `entropy_coef` (multi-head) | [0.05, 0.10, 0.20, 0.30] | 504 actions |

## Usage

### Local Execution

```bash
# REINFORCE + IL
bash optim/scripts/run_optim_reinforce_il.sh

# PPO + Legacy Actions
bash optim/scripts/run_optim_ppo_legacy.sh

# PPO + Multi-head (No Curriculum)
bash optim/scripts/run_optim_ppo_multihead.sh

# PPO + Multi-head + Curriculum Learning
bash optim/scripts/run_optim_ppo_multihead_curriculum.sh

# PPO + Single-head (No Curriculum)
bash optim/scripts/run_optim_ppo_singlehead.sh
```

Or run directly:

```bash
# Scenario 3 (recommended to run first)
python optim/param_optimizer.py \
  --scenario ppo_multihead \
  --n_trials 50 \
  --n_jobs 1

# Scenario 4 (run after optimizing Scenario 3)
python optim/param_optimizer.py \
  --scenario ppo_multihead_curriculum \
  --n_trials 50 \
  --n_jobs 1
```

### SLURM Cluster Execution

```bash
# Submit all optimizations
bash optim/scripts/submit_all_optim.sh

# Submit a single optimization
bash optim/scripts/submit_optim.sh ppo_legacy 100
```

See `optim/SLURM_USAGE.md` for cluster details.

## Output Files

After optimization, the following files are generated in `optim/`:

- `best_params_{scenario}.json` - Best parameters (updated after each trial)
- `trials_{scenario}.csv` - Full trial history
- `optuna_params_{scenario}.db` - Optuna SQLite database
- `best_params_{scenario}_final.json` - Final results

### Example Output

```json
{
  "params": {
    "alpha": 0.000123,
    "clip_epsilon": 0.21,
    "entropy_coef": 0.10,
    "gae_lambda": 0.95,
    "collision_loss_coef": 0.2
  },
  "value": 0.734,
  "mean_score": 0.68,
  "mean_steps": 24.5,
  "mean_collision_rate": 0.08,
  "truncation_rate": 0.12,
  "success_rate": 0.42,
  "total_episodes": 3200,
  "trial_number": 23
}
```

## Pruning Strategy

MedianPruner is used for early stopping:

- REINFORCE / PPO Legacy / PPO Multihead: warmup 200 blocks
- PPO Multihead + Curriculum: warmup 350 blocks

## Monitoring

```bash
# Best params so far (example for Scenario 3)
cat optim/best_params_ppo_multihead.json

# Trial history
less optim/trials_ppo_multihead.csv

# For curriculum optimization
cat optim/best_params_ppo_multihead_curriculum.json
```

## Recommendations

- **REINFORCE**: 20-30 trials
- **PPO Legacy/Multihead/Singlehead**: 30-50 trials
- **PPO Multihead + Curriculum**: 40-60 trials (more parameters to optimize)
- Use `n_jobs=1` unless you have enough RAM for multiple VecEnvs

### Optimization Strategy

For best results with curriculum learning:

1. **First**: Optimize Scenario 3 (`ppo_multihead`) to find optimal baseline parameters
2. **Then**: Copy baseline parameters to Scenario 4 config
3. **Finally**: Optimize Scenario 4 (`ppo_multihead_curriculum`) to tune curriculum-specific parameters

This two-phase approach reduces the search space and improves convergence.
