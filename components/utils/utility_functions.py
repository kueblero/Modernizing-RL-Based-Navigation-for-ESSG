import json
import torch
import random
import numpy as np


def print_dict(d, indent=0):
    """
    Recursively prints a nested dictionary with indentation.

    Args:
        d: Dictionary to print
        indent: Starting indentation level

    Returns:
        None
    """
    for key, value in d.items():
        if isinstance(value, dict):
            print(" " * indent + f"{key}:")
            print_dict(value, indent + 2)
        else:
            print(" " * indent + f"{key}: {value}")


def read_config(file_name, use_print=True) -> dict:
    """
    Load and optionally print JSON configuration file.

    Args:
        file_name: Path to JSON config file
        use_print: Whether to print config to console

    Returns:
        dict: Parsed configuration
    """
    with open(file_name, "r", encoding="utf-8") as file:
        data = json.load(file)
    if use_print:
        print(f"\nLoaded {file_name}:")
        print_dict(data, indent=2)
    return data


def set_seeds(seed, strict_determinism=False):
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed value
        strict_determinism: If True, enables full deterministic mode for complete reproducibility.
                          WARNING: This significantly slows down training (~20-30% slower) but
                          guarantees bitwise identical results across runs.

    Determinism levels:
        strict_determinism=False (default):
            - Sets basic seeds for Python, NumPy, PyTorch
            - Training is mostly reproducible but may have minor variations due to:
              * CuDNN non-deterministic algorithms (faster but not guaranteed identical)
              * TensorFloat32 precision reduction on Ampere+ GPUs
              * Parallel worker race conditions
            - Recommended for normal training (faster, "good enough" reproducibility)

        strict_determinism=True:
            - Enables all deterministic modes
            - Disables CuDNN benchmark and non-deterministic algorithms
            - Configures CUDA for deterministic operations
            - Sets environment variables for hash and CUBLAS determinism
            - Guarantees bitwise identical results across runs with same seed
            - Recommended only for final publication runs or debugging
    """
    import os

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if strict_determinism:
        print("[INFO] Strict determinism enabled - training will be slower but fully reproducible")

        # Disable CuDNN benchmark (finds fastest algorithms but non-deterministic)
        torch.backends.cudnn.benchmark = False

        # Enable CuDNN deterministic mode (uses slower but deterministic algorithms)
        torch.backends.cudnn.deterministic = True

        # Force PyTorch to use deterministic algorithms where available
        # Some operations don't have deterministic implementations and will error
        # We use warn_only=True to warn instead of crashing
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except AttributeError:
            # PyTorch < 1.8 compatibility
            torch._set_deterministic(True)

        # Set environment variables for full determinism
        # PYTHONHASHSEED: Makes Python's hash() function deterministic
        os.environ['PYTHONHASHSEED'] = str(seed)

        # CUBLAS_WORKSPACE_CONFIG: Required for deterministic CUDA operations
        # :4096:8 means 4096MB workspace with 8 streams (good balance)
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

        print("[INFO] Deterministic settings:")
        print(f"  - cudnn.benchmark = {torch.backends.cudnn.benchmark}")
        print(f"  - cudnn.deterministic = {torch.backends.cudnn.deterministic}")
        print(f"  - PYTHONHASHSEED = {os.environ.get('PYTHONHASHSEED')}")
        print(f"  - CUBLAS_WORKSPACE_CONFIG = {os.environ.get('CUBLAS_WORKSPACE_CONFIG')}")
    else:
        # Default mode: Enable benchmark for speed, but not fully deterministic
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False


def generate_seeds(num_seeds=30):
    random.seed(42)
    return [random.randint(0, 1000000) for _ in range(num_seeds)]


def log_mem(tag):
    import torch, psutil, os

    torch.cuda.synchronize()
    print(f"\n=== {tag} ===")
    print(f"CUDA allocated: {torch.cuda.memory_allocated() / 1024 ** 2:.1f} MB")
    print(f"CUDA reserved : {torch.cuda.memory_reserved() / 1024 ** 2:.1f} MB")
    print(f"CPU RSS       : {psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2:.1f} MB")


def log_summary(tag):
    import torch

    torch.cuda.synchronize()
    print(f"\n==== MEMORY SUMMARY: {tag} ====")
    print(torch.cuda.memory_summary())
