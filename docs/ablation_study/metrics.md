# Evaluation Metrics

This document defines all metrics used in the ablation study, their computation methods, and implementation status.

---

## 1. Navigation Metrics

### 1.1 Average Episode Length (Steps)

**Definition:** Mean number of actions taken per episode until reaching the goal or exceeding `max_actions`.

**Computation:**
```python
episode_length = number_of_steps_taken
avg_episode_length = mean(episode_lengths)
```

**Status:** ✅ **Implemented**
- Tracked in: `RL_training/runner/unified_train_runner.py`
- Logged in: `Block/Mean_Steps`, `Rollout/Mean_Steps`

---

### 1.2 Total Path Length (Distance)

**Definition:** Sum of Euclidean distances traveled during an episode.

**Computation:**
```python
path_length = sum(euclidean_distance(pos[i], pos[i+1]) for i in range(len(positions)-1))
```

**Status:** ✅ **Implemented**
- Tracked in: `components/environments/thor_env.py`, `components/environments/precomputed_thor_env.py`
- Logged in: `Block/Mean_Path_Length`, `Rollout/Mean_Path_Length`

**Interpretation:**
- Measures how much the agent actually moves (vs. just taking steps)
- Lower is better for efficient navigation

---

### 1.3 Exploration Coverage

**Definition:** Percentage of reachable positions visited during an episode.

**Computation:**
```python
reachable_positions = set(reachable_positions)
visited_positions = set(positions_visited_during_episode)
coverage = len(visited_positions) / len(reachable_positions)
```

**Status:** ✅ **Implemented**
- Tracked in: `components/environments/abstract_thor_env.py`
- Logged in: `Block/Mean_Exploration_Coverage`, `Rollout/Mean_Exploration_Coverage`

**Notes:**
- ThorEnv loads reachable positions via `GetReachablePositions` and caches them per scene.
- PrecomputedThorEnv uses transition table keys to derive reachable positions.

---

## 2. Scene Graph Metrics

### 2.1 Object Recall (Nodes) and Relation Recall (Edges)

**Definition:** Proportion of ground truth nodes/edges discovered by the agent.

**Status:** ✅ **Implemented**
- Computed in: `components/environments/abstract_thor_env.py`
- Logged in: `Reward/recall_node`, `Reward/recall_edge`

---

## 3. Learning Metrics

### 3.1 Learning Curves (Return vs. Training Steps)

**Definition:** Plot of cumulative reward (return) over training episodes/steps.

**Status:** ✅ **Implemented**
- Logged via `Block/Mean_Reward` and `Rollout/Mean_Reward`

**Suggested Post-Processing:**
- Area Under Curve (AUC)
- Convergence Episode (first time reaching X% of max performance)
- Final Performance (mean over last N episodes)

---

### 3.2 Stability (Variance Across Seeds)

**Definition:** Measure of training stability by running multiple seeds and computing variance.

**Status:** 🚧 **Partially Implemented**
- Seed setting exists in configs
- Multi-seed runner + aggregation still needed

---

### 3.3 Sample Efficiency (Time to Performance Level)

**Definition:** Number of episodes/steps required to reach best performance threshold.

**Status:** 🚧 **Partially Implemented**
- Returns tracked during training
- Threshold detection still needed

---
