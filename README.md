# Modernising Reinforcement Learning–Based Navigation for Embodied Semantic Scene Graph Generation

This repository contains the code accompanying the paper *Modernising Reinforcement Learning–Based Navigation for Embodied Semantic Scene Graph Generation*.

It implements an AI2-THOR navigation module trained with reinforcement learning (REINFORCE/PPO) and optional imitation learning pretraining to let embodied agents build semantic scene graphs efficiently under a limited action budget.
We compare compact vs. high-resolution discrete action spaces, single-head vs. multi-head policies, and optional curriculum learning and depth-based collision supervision.

## Quick Start

⚠️ **Before training**: Large data files (~25GB) must be generated first. See [Data Generation](#data-generation) section below.

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Generate Required Data (~8-16 hours total)

```bash
# 1. Transition tables (~6-12 hours, ~18GB)
python components/scripts/generate_transition_tables.py

# 2. Ground truth graphs (~30-60 minutes, ~500MB)
python components/scripts/generate_gt_graphs.py

# 3. IL dataset (optional, ~4-8 hours, ~1GB)
python ImitationLearning/scripts/generate_il_dataset.py
```

### Multi-Seed RL Training (Recommended)

```bash
python RL_training/multi_seed_train_eval.py \
  --config configs/scenario4_ppo_multihead_curriculum.json \
  --num_seeds 5 \
  --start_seed 0 \
  --n_envs 8 \
  --save_model
```

### Single-Run RL Training

```bash
python RL_training/train.py --config configs/scenario2_ppo_legacy.json --save_model
```

### Imitation Learning (IL)

```bash
# Requires IL dataset (see data generation above)
python ImitationLearning/train_il.py \
  --scenario_config configs/scenario0_reinforce_il.json \
  --epochs 50 \
  --batch_size 32
```

### Hyperparameter Optimization

```bash
python optim/param_optimizer.py --scenario ppo_legacy --n_trials 50 --n_jobs 1
```

### View Pre-computed Results

```bash
# No data generation needed - TensorBoard event files are included
tensorboard --logdir RL_training/runs
```

## Action Spaces

- Legacy: 16 discrete actions (8 directions x 2 lengths).
- Multi-head: direction (24) x length (21) plus a binary stop head.

See `experiments/ablation_study/ACTION_SPACES.md` for details.

## TensorBoard

TensorBoard logs are stored under `RL_training/runs/`.

```bash
tensorboard --logdir=RL_training/runs
```

More details: `docs/TENSORBOARD.md`.

## Cluster Scripts

- RL training scripts: `RL_training/scripts/`
  - `train_scenario*.sh`, `train_all_scenarios.sh`, `train_scenario.sbatch`
- IL training scripts: `ImitationLearning/scripts/`
  - `train_il.sbatch`, `submit_train_il.sh`
- Optimization scripts: `optim/scripts/`
  - `run_optim_*.sh`, `submit_optim.sh`

## Data Generation

⚠️ **Large data files (~25GB) are not included in this repository** and must be generated locally. The repository includes only:
- Source code and configurations
- TensorBoard event files for experiment visualization
- Data generation scripts

### System Requirements

- **Disk Space**: ~30GB free space
- **RAM**: 16GB recommended (32GB for parallel training)
- **GPU**: NVIDIA GPU with CUDA support (for training)
- **Python**: 3.8+
- **AI2-THOR**: Compatible with Python 3.8-3.10

### Setup Instructions

#### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

#### 2. Generate Transition Tables (~18GB, ~6-12 hours)

Pre-computed state transitions for all 30 AI2-THOR kitchen floor plans. These tables cache RGB frames, depth images, object metadata, and segmentation masks for every reachable position and rotation.

```bash
python components/scripts/generate_transition_tables.py
```

**Output**: `components/data/transition_tables/FloorPlan{1-30}.lmdb/`
- ~18GB total (FloorPlan7 is largest at ~1.2GB)
- LMDB format for fast random access
- Time: ~6-12 hours depending on hardware

**Parameters** (edit in script):
- `grid_size=0.1` - Agent grid resolution (meters)
- `render_width=300, render_height=300` - Image resolution
- `jpg_quality=85` - JPEG compression quality

#### 3. Generate Ground Truth Graphs (~500MB, ~30-60 minutes)

Ground truth navigation graphs for shortest path calculations and evaluation metrics.

```bash
python components/scripts/generate_gt_graphs.py
```

**Output**: `components/data/gt_graphs/FloorPlan{1-30}.json`
- ~500MB total (large JSON files, 10-21MB each)
- Contains full navigable graph with edges and distances

#### 4. Generate IL Dataset (~1GB, ~4-8 hours)

Imitation learning dataset with expert demonstrations using shortest paths.

```bash
python ImitationLearning/scripts/generate_il_dataset.py
```

**Output**: `components/data/il_dataset/FloorPlan{1-30}/*.pkl`
- ~1GB total (pickle files, 10-24MB each)
- Expert trajectories for behavior cloning

#### 5. (Optional) Train IL Pre-trained Model (~170MB, ~2-4 hours)

Pre-train a model using imitation learning for warm-start in RL scenarios:

```bash
python ImitationLearning/train_il.py \
  --scenario_config configs/scenario0_reinforce_il.json \
  --epochs 50 \
  --batch_size 32
```

**Output**: `components/data/model_weights/il/il_pretrained_model.pth` (~170MB)

### Using Pre-trained RL Models

Final model weights from the ablation study experiments (~7GB total) are **not included** in the repository. You can:

**Option A: View Training Results** - TensorBoard event files are included in `RL_training/runs/*/`:

```bash
tensorboard --logdir RL_training/runs
```

**Option B: Retrain Models** - Use the provided configs and training scripts:

```bash
# Example: Scenario 7 (PPO Singlehead with Curriculum)
python RL_training/multi_seed_train_eval.py \
  --config configs/scenario7_ppo_singlehead_curriculum.json \
  --num_seeds 5 \
  --n_envs 8 \
  --save_model
```

### Data Structure Overview

```
components/data/
├── transition_tables/           # ~18GB - Pre-computed state transitions
│   ├── FloorPlan1.lmdb/        # ~695MB
│   ├── FloorPlan7.lmdb/        # ~1.2GB (largest)
│   └── ...
├── gt_graphs/                   # ~500MB - Navigation graphs
│   ├── FloorPlan1.json         # ~10MB
│   ├── FloorPlan7.json         # ~21MB (largest)
│   └── ...
├── il_dataset/                  # ~1GB - IL demonstrations
│   ├── FloorPlan1/*.pkl
│   ├── FloorPlan7/*.pkl        # ~18-24MB files
│   └── ...
└── model_weights/
    └── il/
        └── il_pretrained_model.pth  # ~170MB (optional)

RL_training/runs/                # ~7GB model weights (not included)
├── scenario0_reinforce_il_seed_*/
│   ├── events.out.tfevents.*   # ✅ Included (TensorBoard logs)
│   └── checkpoints/
│       └── final_model.pth     # ❌ Not included (~166MB each)
└── ...
```

## Project Structure

- `RL_training/` - RL training entry points and runners
- `ImitationLearning/` - IL dataset + training
- `components/` - environments, models, agents, utilities
- `optim/` - Optuna optimization
- `configs/` - Scenario configs used in ablation study
- `experiments/ablation_study/` - Study docs
