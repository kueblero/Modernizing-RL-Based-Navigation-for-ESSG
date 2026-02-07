# TensorBoard on the Cluster

How to view training runs on the Uni-Augsburg cluster.

## Overview

Training logs are written to `RL_training/runs/`. You can view them via:
1) **SSH tunnel** (live monitoring), or
2) **Local copy** (offline analysis).

## Option 1: SSH Tunnel (Live)

### Step 1: SSH with port forwarding

```bash
ssh -L 6006:localhost:6006 kueblero@oc-appsrv01.informatik.uni-augsburg.de
```

### Step 2: Start TensorBoard on the cluster

```bash
cd /data/oc-compute03/kueblero/projects/NavigationModule

srun --partition=cpu --time=04:00:00 --mem=4G \
  /run/current-system/sw/bin/nix develop .#default --command \
  tensorboard --logdir=RL_training/runs --bind_all --port=6006
```

### Step 3: Open in browser

```
http://localhost:6006
```

Notes:
- Use `tmux`/`screen` if you don’t want the session to block your shell.
- `--bind_all` is required for port forwarding.

---

## Option 2: Local Copy (Offline)

### Step 1: Download runs

```bash
rsync -avzP --progress \
  kueblero@oc-appsrv01.informatik.uni-augsburg.de:/data/oc-compute03/kueblero/projects/NavigationModule/RL_training/runs/ \
  ./RL_training/runs/
```

### Step 2: Start TensorBoard locally

```bash
tensorboard --logdir=RL_training/runs
```

### Step 3: Open in browser

```
http://localhost:6006
```

---

## Run Directory Layout

```
RL_training/runs/
├── Scenario_0_REINFORCE_Legacy_.../
├── Scenario_1_REINFORCE_Legacy_IL_.../
├── Scenario_2_PPO_Legacy_.../
├── Scenario_3_PPO_Multihead_NoCurr_.../
└── Scenario_4_PPO_Multihead_Curriculum_.../
```

---

## Metrics in TensorBoard

### Block / Rollout
- `Block/Mean_Reward`, `Block/Mean_Steps`, `Block/Mean_Score`
- `Block/Mean_Path_Length`, `Block/Mean_Exploration_Coverage`
- `Block/Num_Terminated`, `Block/Num_Truncated`, `Block/Terminated_Truncated_Ratio`
- `Rollout/Mean_*` (moving window)

### Loss / Policy
- `Loss/policy_loss`, `Loss/value_loss` (PPO)
- `Loss/collision_loss` (multi-head)
- `policy/entropy`, `policy/approx_kl`, `policy/clip_fraction`
- `policy/ret_std` (REINFORCE), `policy/collision_acc`

### Actions (per block)
- `actions/total`, `actions/movements`, `actions/pure_rotations`, `actions/stops`, `actions/idle`
- `actions/pure_rotation_rate`, `actions/stop_rate`
- `actions/unique_actions_used`, `actions/unique_directions_used`, `actions/unique_lengths_used`
- `actions/top_direction_*`, `actions/top_length_*` (multi-head)

### Reward components
- `Reward/recall_node`, `Reward/recall_edge`
- `Reward/collision_rate`, `Reward/time_penalty`, `Reward/total_reward`

### Curriculum
- `curriculum/stage`, `curriculum/entropy_coef`
- `curriculum/num_valid_actions`
- `curriculum/new_action_coverage_pct`, `curriculum/new_actions_tried`

---

## Troubleshooting

**Port in use**
```bash
tensorboard --logdir=RL_training/runs --port=6007
ssh -L 6007:localhost:6007 kueblero@oc-appsrv01.informatik.uni-augsburg.de
```

**No data visible**
1) `ls RL_training/runs/`  
2) `find RL_training/runs -name "events*"`  
3) Wait 1–2 minutes; TensorBoard may take time to load.

**Connection drops**
- Use `tmux`/`screen`
- Check VPN
- SSH keep-alive: `ssh -o ServerAliveInterval=60 ...`

---

## Links
- [TensorBoard docs](https://www.tensorflow.org/tensorboard)
- [Ablation Study](ablation_study/)
