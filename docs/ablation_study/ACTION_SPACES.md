# Action Space Definitions

## Overview

This document describes the two action spaces used in the navigation system:

1. **Legacy Actions** (16 discrete) - Used for IL and baseline scenarios
2. **Multi-head Actions** (24 x 21 + stop head) - Used for advanced scenarios with curriculum

---

## Legacy Action Space (Baseline)

**Used in:**
- Imitation Learning (IL) dataset generation
- Scenario 0/1: REINFORCE + Legacy
- Scenario 2: PPO + Legacy

**Configuration:**
```python
use_legacy_actions=True
```

### Action Structure

Actions are discrete indices representing `(relative_angle, length)` pairs.

- **Directions (8):** 0, 45, 90, 135, 180, 225, 270, 315 degrees
- **Lengths (2):** 0.0m (stop/rotate) and 0.3m (move)
- **Total actions:** 8 x 2 = 16

**STOP:** `(angle=0, length=0.0)` which is index 0 in `legacy_actions`.

### Execution Order

1. Move in the **current** direction by `length` (if > 0)
2. Rotate by `relative_angle` (preparation for the next step)

**Pure rotation:** `length=0.0` and `angle != 0`

---

## Multi-head Action Space

**Used in:**
- Scenario 3: PPO + Multi-head (no curriculum)
- Scenario 4: PPO + Multi-head + Curriculum

**Configuration:**
```python
use_legacy_actions=False
```

### Action Structure

Actions are tuples `(dir_idx, len_idx, stop_flag)`.

- **Direction (24):** 0, 15, 30, ..., 345 degrees (relative rotation)
- **Length (21):** 0.0m, 0.1m, ..., 2.0m
- **Stop head (2):** `stop_flag` in {0, 1}

**STOP:** `stop_flag=1` (dir/len ignored). If `stop_flag` is missing, `(dir=0, len=0)` is treated as STOP.

**Pure rotation:** `len_idx=0`, `dir_idx != 0`, `stop_flag=0`

### Execution Order

1. Move in the **current** direction by `length`
2. Rotate by `relative_angle`

---

## Curriculum Learning (Multi-head Only)

The policy always uses full heads (24 x 21), but sampling is masked by stage.

### Stage 1
- Directions: 8 (0, 45, 90, 135, 180, 225, 270, 315)
- Lengths: 2 (0.0, 0.3)
- Total: 16

### Stage 2
- Directions: 8 (same as Stage 1)
- Lengths: 6 (0.0, 0.3, 0.5, 1.0, 1.5, 2.0)
- Total: 48

### Stage 3
- Directions: 12 (0, 30, 60, ..., 330)
- Lengths: 9 (0.0, 0.3, 0.5, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0)
- Total: 108

### Stage 4
- Directions: 16 (0..225 with 15 degree offsets)
- Lengths: 13 (0.0, 0.2, 0.3, 0.5, 0.7, 0.8, 1.0, 1.2, 1.4, 1.5, 1.7, 1.8, 2.0)
- Total: 208

### Stage 5
- Directions: 20 (most angles)
- Lengths: 17 (0.0..2.0 with finer steps)
- Total: 340

### Stage 6
- Directions: 24 (full 15-degree grid)
- Lengths: 21 (0.0..2.0 in 0.1m steps)
- Total: 504

---

## Comparison

| Aspect | Legacy (16) | Multi-head Stage 1 (16) | Multi-head Full (504) |
|--------|-------------|-------------------------|-----------------------|
| Directions | 8 (relative) | 8 (relative) | 24 (relative) |
| Direction Precision | 45 deg | 45 deg | 15 deg |
| Distance Options | 2 (0.0, 0.3) | 2 | 21 (0.0-2.0) |
| Stop | Action index 0 | Stop head + (0,0) fallback | Stop head + (0,0) fallback |
| Action Space Size | 16 | 16 | 504 (+ stop head) |

---

## Implementation Notes

### Legacy Actions (code)

```python
self.action_angles = [0, 45, 90, 135, 180, 225, 270, 315]
self.action_lengths = [0.0, 0.3]
self.legacy_actions = [(angle, length) for angle in self.action_angles for length in self.action_lengths]
self.num_actions = len(self.legacy_actions)  # 16
self.stop_index = 0
```

### Multi-head Actions (code)

```python
self.action_angles = list(range(0, 360, 15))  # 24 directions
self.action_lengths = [round(x * 0.1, 1) for x in range(0, 21)]  # 21 lengths
self.num_directions = len(self.action_angles)
self.num_lengths = len(self.action_lengths)
```

---

## Usage in Ablation Study

| Scenario | Action Space | Curriculum | IL Pre-training |
|----------|--------------|------------|-----------------|
| 0: REINFORCE | Legacy (16) | No | No |
| 1: REINFORCE + IL | Legacy (16) | No | Yes |
| 2: PPO Legacy | Legacy (16) | No | No |
| 3: PPO Multi-head | Multi-head (504 + stop) | No (Stage 6) | No |
| 4: PPO Multi-head + Curriculum | Multi-head (504 + stop) | Yes (Stages 1-6) | No |
