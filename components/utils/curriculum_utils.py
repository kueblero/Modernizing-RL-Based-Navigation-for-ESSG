"""
Curriculum Learning Utilities

Provides flexible curriculum stage generation with geometric progression.
Used for multi-head action space curriculum (dirs × lens).

Features:
- Automatic stage calculation based on n_stages
- Geometric progression (exponential growth)
- Mapping to valid (dirs, lens) combinations
- Support for "last"/"max"/"full" syntax
"""

import math
from typing import List, Tuple, Union


def compute_geometric_stages(
    n_stages: int,
    min_actions: int = 16,
    max_actions: int = 504,
    max_dirs: int = 24,
    max_lens: int = 21,
    start_dirs: int = 8,
    start_lens: int = 2,
) -> List[Tuple[int, int, int]]:
    """
    Compute curriculum stages using geometric progression with navigation-aware constraints.

    Stage 1 always starts with legacy actions (8 dirs × 2 lens = 16 actions).
    Early stages prioritize adding more lengths (more important for movement).
    Later stages add more directions (fine-tuning orientation).

    Args:
        n_stages: Number of curriculum stages
        min_actions: Minimum actions in first stage (default: 16)
        max_actions: Maximum actions in last stage (default: 504 = 24×21)
        max_dirs: Maximum number of directions (default: 24)
        max_lens: Maximum number of lengths (default: 21)
        start_dirs: Initial number of directions (default: 8, legacy actions)
        start_lens: Initial number of lengths (default: 2, legacy actions)

    Returns:
        List of (stage_num, n_dirs, n_lens) tuples

    Example:
        >>> compute_geometric_stages(n_stages=6)
        [(1, 8, 2), (2, 8, 6), (3, 12, 9), (4, 16, 13), (5, 20, 17), (6, 24, 21)]
        # Stage 1: Legacy actions, then gradually expand
    """
    if n_stages < 1:
        raise ValueError(f"n_stages must be >= 1, got {n_stages}")

    if n_stages == 1:
        # Single stage: use max actions
        return [(1, max_dirs, max_lens)]

    stages = []

    # Stage 1: Always use legacy actions (8 dirs × 2 lens)
    stages.append((1, start_dirs, start_lens))

    if n_stages == 2:
        # Only 2 stages: legacy -> full
        stages.append((2, max_dirs, max_lens))
        return stages

    # Compute geometric progression of target actions for stages 2 to n-1
    # Stage n always uses max_actions
    # We already have stage 1 (16 actions), need to get to max_actions (504)
    ratio = (max_actions / min_actions) ** (1.0 / (n_stages - 1))

    for stage_idx in range(1, n_stages):
        # Last stage: always use full action space
        if stage_idx == n_stages - 1:
            n_dirs, n_lens = max_dirs, max_lens
        else:
            # Intermediate stages: use geometric progression
            target_actions = min_actions * (ratio ** stage_idx)

            # Find best (dirs, lens) combination with navigation-aware heuristic
            n_dirs, n_lens = _find_navigation_aware_combination(
                target_actions, max_dirs, max_lens, start_dirs, start_lens, stage_idx, n_stages
            )

        stages.append((stage_idx + 1, n_dirs, n_lens))

    return stages


def _find_navigation_aware_combination(
    target_actions: float,
    max_dirs: int,
    max_lens: int,
    start_dirs: int,
    start_lens: int,
    stage_idx: int,
    total_stages: int,
) -> Tuple[int, int]:
    """
    Find (dirs, lens) combination that prioritizes lengths in early stages.

    Strategy:
    - Early stages (< 50%): Prefer increasing lengths over directions
    - Later stages (>= 50%): Balance both dimensions
    - Always ensure monotonic growth in both dims

    Args:
        target_actions: Target number of actions
        max_dirs: Maximum directions
        max_lens: Maximum lengths
        start_dirs: Starting directions (8)
        start_lens: Starting lengths (2)
        stage_idx: Current stage index (0-indexed)
        total_stages: Total number of stages

    Returns:
        (n_dirs, n_lens) tuple
    """
    # Valid direction counts (must divide 360 evenly)
    valid_dirs = []
    for n in range(start_dirs, max_dirs + 1):
        if 360 % n == 0:
            valid_dirs.append(n)

    # Compute stage progression (0 to 1)
    progress = stage_idx / (total_stages - 1)

    best_combo = (start_dirs, start_lens)
    best_error = float("inf")

    for n_dirs in valid_dirs:
        if n_dirs > max_dirs:
            continue

        # Compute ideal n_lens for this n_dirs
        ideal_lens = target_actions / n_dirs
        ideal_lens = max(start_lens, min(ideal_lens, max_lens))

        # Try floor and ceil
        for n_lens in [math.floor(ideal_lens), math.ceil(ideal_lens)]:
            n_lens = max(start_lens, min(n_lens, max_lens))

            # Compute error with navigation-aware weighting
            actual_actions = n_dirs * n_lens
            action_error = abs(actual_actions - target_actions)

            # In early stages (progress < 0.5), penalize increasing directions
            # Prefer to increase lengths first
            if progress < 0.5:
                # Penalize adding many directions early
                dir_penalty = (n_dirs - start_dirs) * 5  # Higher penalty early
                weighted_error = action_error + dir_penalty
            else:
                # Later stages: just minimize action error
                weighted_error = action_error

            if weighted_error < best_error:
                best_error = weighted_error
                best_combo = (n_dirs, n_lens)

    return best_combo


def _find_closest_valid_combination(
    target_actions: float, max_dirs: int, max_lens: int
) -> Tuple[int, int]:
    """
    Find the closest valid (dirs, lens) combination to target_actions.

    Valid directions: divisors of 360 (e.g., 8, 12, 16, 20, 24)
    Valid lengths: any integer from 2 to max_lens

    Args:
        target_actions: Target number of actions (dirs × lens)
        max_dirs: Maximum directions
        max_lens: Maximum lengths

    Returns:
        (n_dirs, n_lens) tuple that minimizes |dirs × lens - target_actions|
    """
    # Valid direction counts (must divide 360 evenly for relative rotations)
    # 360° / n_dirs = step size
    valid_dirs = []
    for n in range(1, max_dirs + 1):
        if 360 % n == 0:  # Must divide 360
            valid_dirs.append(n)

    # Find best combination
    best_combo = (8, 2)  # Default: minimum
    best_error = float("inf")

    for n_dirs in valid_dirs:
        if n_dirs > max_dirs:
            continue

        # Compute ideal n_lens for this n_dirs
        ideal_lens = target_actions / n_dirs
        ideal_lens = max(2, min(ideal_lens, max_lens))  # Clamp to valid range

        # Try floor and ceil
        for n_lens in [math.floor(ideal_lens), math.ceil(ideal_lens)]:
            n_lens = max(2, min(n_lens, max_lens))  # Clamp

            # Compute error
            actual_actions = n_dirs * n_lens
            error = abs(actual_actions - target_actions)

            if error < best_error:
                best_error = error
                best_combo = (n_dirs, n_lens)

    return best_combo


def get_valid_direction_indices(n_dirs: int, max_dirs: int = 24) -> List[int]:
    """
    Get valid direction indices for a given stage.

    Directions are uniformly spaced angles from 0 to 360 degrees.
    For n_dirs < max_dirs, we take a uniform subset.

    Args:
        n_dirs: Number of directions for this stage
        max_dirs: Total number of directions in full action space (default: 24)

    Returns:
        List of direction indices (0 to max_dirs-1)

    Example:
        >>> get_valid_direction_indices(8, 24)
        [0, 3, 6, 9, 12, 15, 18, 21]  # Every 3rd direction (45° spacing)
    """
    if n_dirs > max_dirs:
        raise ValueError(f"n_dirs ({n_dirs}) cannot exceed max_dirs ({max_dirs})")

    if n_dirs == max_dirs:
        return list(range(max_dirs))

    # Compute step size to get uniform spacing
    step = max_dirs / n_dirs

    # Generate indices
    indices = [int(round(i * step)) % max_dirs for i in range(n_dirs)]

    return indices


def get_valid_length_indices(n_lens: int, max_lens: int = 21) -> List[int]:
    """
    Get valid length indices for a given stage.

    Lengths range from 0.0m to 2.0m in 0.1m increments (21 total).
    For n_lens < max_lens, we prioritize:
    1. Always include 0.0m (index 0) for rotation
    2. Always include 0.3m (index 3) as minimum movement
    3. Always include 2.0m (index 20) as maximum movement
    4. Fill in intermediate lengths uniformly between 0.3m and 2.0m

    Args:
        n_lens: Number of lengths for this stage
        max_lens: Total number of lengths in full action space (default: 21)

    Returns:
        List of length indices (0 to max_lens-1)

    Example:
        >>> get_valid_length_indices(6, 21)
        [0, 3, 7, 11, 15, 20]  # 0.0, 0.3, 0.7, 1.1, 1.5, 2.0
    """
    if n_lens > max_lens:
        raise ValueError(f"n_lens ({n_lens}) cannot exceed max_lens ({max_lens})")

    if n_lens == max_lens:
        return list(range(max_lens))

    if n_lens == 1:
        return [0]  # Only rotation (0.0m)

    if n_lens == 2:
        return [0, 3]  # 0.0m and 0.3m

    # For many lengths (>= 15), use sequential filling to ensure monotonic growth
    if n_lens >= 15:
        # Take first (n_lens-1) indices plus max_lens-1 (to always include 2.0m)
        indices = list(range(n_lens - 1)) + [max_lens - 1]
        return sorted(set(indices))[:n_lens]  # Remove duplicates and limit to n_lens

    # For fewer lengths, always include 0 (rotation), 3 (0.3m), and max_lens-1 (2.0m)
    indices = [0, 3, max_lens - 1]

    # Fill in intermediate indices uniformly between 3 and max_lens-1
    remaining = n_lens - 3
    if remaining > 0:
        step = (max_lens - 1 - 3) / (remaining + 1)
        for i in range(1, remaining + 1):
            idx = int(round(3 + i * step))
            idx = max(4, min(idx, max_lens - 2))
            if idx not in indices:
                indices.append(idx)

    # Sort and ensure uniqueness
    indices = sorted(set(indices))

    # If we have too many (due to rounding), trim
    if len(indices) > n_lens:
        # Keep first (0) and last (max), sample middle
        middle = indices[1:-1]
        step = len(middle) / (n_lens - 2)
        new_middle = [middle[int(round(i * step))] for i in range(n_lens - 2)]
        indices = [indices[0]] + new_middle + [indices[-1]]

    return indices[:n_lens]


def parse_curriculum_stage(
    stage_value: Union[int, str], n_stages: int
) -> int:
    """
    Parse curriculum stage value, supporting both int and string syntax.

    Args:
        stage_value: Stage value from config (int or string)
        n_stages: Total number of stages

    Returns:
        Stage number (1-indexed)

    Supported values:
        - int (1 to n_stages): Direct stage number
        - "last", "max", "full": Final stage (n_stages)
        - "first", "min": First stage (1)

    Example:
        >>> parse_curriculum_stage("last", n_stages=6)
        6
        >>> parse_curriculum_stage(3, n_stages=6)
        3
    """
    if isinstance(stage_value, int):
        if stage_value < 1 or stage_value > n_stages:
            raise ValueError(
                f"Stage {stage_value} out of range [1, {n_stages}]"
            )
        return stage_value

    if isinstance(stage_value, str):
        stage_str = stage_value.lower().strip()

        if stage_str in ["last", "max", "full"]:
            return n_stages
        elif stage_str in ["first", "min"]:
            return 1
        else:
            # Try to parse as int
            try:
                stage_int = int(stage_str)
                return parse_curriculum_stage(stage_int, n_stages)
            except ValueError:
                raise ValueError(
                    f"Invalid curriculum stage: '{stage_value}'. "
                    f"Use int (1-{n_stages}), 'last', 'max', 'full', 'first', or 'min'."
                )

    raise ValueError(
        f"curriculum_stage must be int or str, got {type(stage_value)}: {stage_value}"
    )


def print_curriculum_stages(stages: List[Tuple[int, int, int]]) -> None:
    """
    Print curriculum stages in a readable format.

    Args:
        stages: List of (stage_num, n_dirs, n_lens) tuples
    """
    print("\n[INFO] Curriculum Stages:")
    for stage_num, n_dirs, n_lens in stages:
        n_actions = n_dirs * n_lens
        angle_step = 360 / n_dirs
        print(
            f"  Stage {stage_num}: {n_actions:3d} actions "
            f"({n_dirs:2d} dirs × {n_lens:2d} lens) - "
            f"{angle_step:.1f}° steps, 0.0-2.0m"
        )
    print()
