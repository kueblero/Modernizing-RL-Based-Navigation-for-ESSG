# Ablation Study: Navigation Module

## Overview

Comprehensive ablation study comparing IL pretraining, action spaces, feature modalities, and curriculum learning across 10 scenarios:

**Baseline Scenarios:**
- **Scenario 0:** REINFORCE baseline (legacy actions, no IL)
- **Scenario 1:** REINFORCE + IL pretraining
- **Scenario 2:** PPO + legacy actions

**Multi-head Action Space:**
- **Scenario 3:** PPO + multi-head (no curriculum)
- **Scenario 4:** PPO + multi-head + curriculum (adaptive)
- **Scenario 5:** PPO + multi-head (no depth, no curriculum)
- **Scenario 8:** PPO + multi-head + curriculum (no depth)

**Single-head Action Space:**
- **Scenario 6:** PPO + single-head (no curriculum)
- **Scenario 7:** PPO + single-head + curriculum (adaptive)

**External Baseline:**
- **Baseline:** Li et al. baseline (REINFORCE + IL, no depth)

---

## Experimental Settings

| Scenario | Agent | Action Space | Features | Curriculum | Training Blocks |
|----------|-------|--------------|----------|------------|-----------------|
| 0 | REINFORCE | Legacy (16) | RGB+Depth | No | 750 |
| 1 | REINFORCE + IL | Legacy (16) | RGB+Depth | No | 750 |
| 2 | PPO | Legacy (16) | RGB+Depth | No | 750 |
| 3 | PPO | Multi-head (24×21) + stop | RGB+Depth | No (Stage 6 from start) | 1000 |
| 4 | PPO | Multi-head (24×21) + stop | RGB+Depth | Yes (adaptive Stages 1–6) | 1000 |
| 5 | PPO | Multi-head (24×21) + stop | RGB only | No (Stage 6 from start) | 1000 |
| 6 | PPO | Single-head (504) | RGB+Depth | No (Stage 6 from start) | 1000 |
| 7 | PPO | Single-head (504) | RGB+Depth | Yes (adaptive Stages 1–6) | 1000 |
| 8 | PPO | Multi-head (24×21) + stop | RGB only | Yes (adaptive Stages 1–6) | 1000 |
| Baseline | REINFORCE + IL | Legacy (16) | RGB only | No | 750 |

**Curriculum stages (masking) for Scenario 4:**
1) 8×2 = 16 actions  
2) 8×6 = 48 actions  
3) 12×9 = 108 actions  
4) 16×13 = 208 actions  
5) 20×17 = 340 actions  
6) 24×21 = 504 actions  
Stage transitions follow adaptive plateau detection from the config.

---

## How to Run

### On the cluster (SLURM)

```bash
# Single scenario
sbatch --export=SCENARIO=1 RL_training/scripts/train_scenario.sbatch

# All scenarios
for i in 0 1 2 3 4; do
  sbatch --export=SCENARIO=$i RL_training/scripts/train_scenario.sbatch
done
```

### Local

```bash
# Single scenario
RL_training/scripts/train_scenario1.sh

# All scenarios
RL_training/scripts/train_all_scenarios.sh
```

---

## Monitoring

### TensorBoard

```bash
tensorboard --logdir=RL_training/runs
```

See `docs/TENSORBOARD.md` for details.

### Job status (cluster)

```bash
squeue -u $USER
tail -f logs/train_<jobid>.out
```

---

## Config Files

```
configs/
├── baseline_scenario_reinforce_nodepth_il.json
├── scenario0_reinforce_il.json
├── scenario1_reinforce_no_il.json
├── scenario2_ppo_legacy.json
├── scenario3_ppo_multihead_no_curriculum.json
├── scenario4_ppo_multihead_curriculum.json
├── scenario5_ppo_multihead_no_depth.json
├── scenario6_ppo_singlehead_no_curriculum.json
├── scenario7_ppo_singlehead_curriculum.json
└── scenario8_ppo_multihead_curriculum_no_depth.json
```

---

## Expected Trends (qualitative)

| Scenario | Sample Efficiency | Final Performance | Action Precision | Stability |
|----------|------------------|-------------------|------------------|-----------|
| 0 | Low | Low | Low | Medium |
| 1 | Medium | Medium | Low | High |
| 2 | High | Medium | Low | High |
| 3 | Low | Low–Medium | High | Low |
| 4 | Medium–High | High | High | High |
| 5 | Low | Low–Medium | High | Medium |
| 6 | Low | Medium | High | Low |
| 7 | Medium | High | High | Medium |
| 8 | Medium | Medium–High | High | Medium |
| Baseline | Low–Medium | Low–Medium | Low | High |

---

## Outputs

```
RL_training/runs/
├── scenario0_reinforce_il_seed_<N>_<timestamp>/
├── scenario1_reinforce_no_il_seed_<N>_<timestamp>/
├── scenario2_ppo_legacy_seed_<N>_<timestamp>/
├── scenario3_ppo_multihead_no_curriculum_seed_<N>_<timestamp>/
├── scenario4_ppo_multihead_curriculum_seed_<N>_<timestamp>/
├── scenario5_ppo_multihead_no_depth_seed_<N>_<timestamp>/
├── scenario6_ppo_singlehead_no_curriculum_seed_<N>_<timestamp>/
├── scenario7_ppo_singlehead_curriculum_seed_<N>_<timestamp>/
├── scenario8_ppo_multihead_curriculum_no_depth_seed_<N>_<timestamp>/
└── baseline_scenario_reinforce_nodepth_il_seed_<N>_<timestamp>/
```

---

## Further Docs

- [Metrics](metrics.md)
- [Action Spaces](ACTION_SPACES.md)
- [TensorBoard](../TENSORBOARD.md)
- [LMDB Tables](../../scripts/README_LMDB.md)
