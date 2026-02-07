"""
Simple Vectorized Environment Wrapper for parallel scene execution.

Instead of running scenes sequentially, we run N environments in parallel using multiprocessing.
This amortizes the cost of PPO's multiple epochs by collecting more data per update.
"""

import multiprocessing as mp
from typing import List, Tuple, Any
import numpy as np


class CloudpickleWrapper:
    """
    Wrapper to make functions picklable for multiprocessing.
    """
    def __init__(self, fn):
        self.fn = fn

    def __call__(self):
        return self.fn()


def worker(remote, parent_remote, env_fn_wrapper, worker_id, base_seed):
    """
    Worker process for vectorized environment.
    Each worker runs one environment instance.

    Args:
        remote: Pipe for communication with main process
        parent_remote: Parent's end of pipe (to be closed)
        env_fn_wrapper: Wrapped function that creates the environment
        worker_id: Unique ID for this worker (0, 1, 2, ...)
        base_seed: Base seed from config, worker uses base_seed + worker_id
    """
    import random
    import numpy as np
    import torch

    parent_remote.close()

    # Set unique seed for this worker for reproducibility
    worker_seed = base_seed + worker_id
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(worker_seed)

    try:
        env = env_fn_wrapper()
        # Store seed in environment for reset reproducibility
        env._worker_seed = worker_seed
        env._reset_count = 0
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        remote.send(('error', f"Failed to create environment:\n{tb}"))
        return

    while True:
        try:
            cmd, data = remote.recv()

            if cmd == 'step':
                try:
                    obs = env.step(data)
                    # Extract components from Observation object
                    reward = obs.reward if hasattr(obs, 'reward') else 0.0
                    terminated = obs.terminated if hasattr(obs, 'terminated') else False
                    truncated = obs.truncated if hasattr(obs, 'truncated') else False
                    info = obs.info if hasattr(obs, 'info') else {}
                    remote.send((obs, reward, terminated, truncated, info))
                except Exception as e:
                    import traceback
                    tb = traceback.format_exc()
                    remote.send(('error', f"Error in step():\n{tb}"))

            elif cmd == 'reset':
                try:
                    obs = env.reset()
                    # Extract info from Observation object
                    info = obs.info if hasattr(obs, 'info') else {}
                    remote.send((obs, info))
                except Exception as e:
                    import traceback
                    tb = traceback.format_exc()
                    remote.send(('error', f"Error in reset():\n{tb}"))

            elif cmd == 'close':
                remote.close()
                break

            elif cmd == 'get_action_space_dims':
                remote.send(env.get_action_space_dims())

            elif cmd == 'get_attr':
                remote.send(getattr(env, data))

            elif cmd == 'set_attr':
                attr_name, value = data
                setattr(env, attr_name, value)
                remote.send(None)

            elif cmd == 'call_method':
                method_name, args, kwargs = data
                method = getattr(env, method_name)
                result = method(*args, **kwargs)
                remote.send(result)

            else:
                raise NotImplementedError(f"Unknown command: {cmd}")

        except EOFError:
            break
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            try:
                remote.send(('error', f"Unexpected error in worker:\n{tb}"))
            except:
                pass  # Pipe might be broken
            break


class VecEnv:
    """
    Vectorized environment that runs multiple environments in parallel using multiprocessing.

    Usage:
        def make_env(scene_name):
            return PrecomputedThorEnv(scene_name=scene_name, ...)

        env_fns = [lambda: make_env(f"FloorPlan{i}") for i in range(1, 5)]
        vec_env = VecEnv(env_fns, seed=42)

        obs, info = vec_env.reset()  # Reset all envs
        actions = [(0, 1), (1, 2), (0, 0), (2, 3)]  # One action per env
        obs, rewards, terminated, truncated, infos = vec_env.step(actions)
    """

    def __init__(self, env_fns: List[callable], seed: int = 42):
        """
        Args:
            env_fns: List of functions that create environment instances
            seed: Base seed for reproducibility. Each worker gets seed + worker_id.
        """
        self.num_envs = len(env_fns)
        self.closed = False
        self.base_seed = seed

        # Create pipes for communication
        self.remotes, self.work_remotes = zip(*[mp.Pipe() for _ in range(self.num_envs)])

        # Start worker processes with unique seeds
        self.processes = []
        for worker_id, (work_remote, remote, env_fn) in enumerate(zip(self.work_remotes, self.remotes, env_fns)):
            args = (work_remote, remote, CloudpickleWrapper(env_fn), worker_id, seed)
            process = mp.Process(target=worker, args=args, daemon=True)
            process.start()
            self.processes.append(process)
            work_remote.close()

        # Get action space dims from first env
        self.remotes[0].send(('get_action_space_dims', None))
        result = self.remotes[0].recv()
        if isinstance(result, tuple) and len(result) == 2 and result[0] == 'error':
            raise RuntimeError(f"Failed to initialize VecEnv:\n{result[1]}")
        self._action_space_dims = result

    def step(self, actions: List[Tuple[int, int]]):
        """
        Step all environments with given actions.

        Args:
            actions: List of (dir_idx, len_idx) tuples, one per environment

        Returns:
            observations: List of observations
            rewards: List of rewards
            terminated: List of terminated flags
            truncated: List of truncated flags
            infos: List of info dicts
        """
        assert len(actions) == self.num_envs, f"Expected {self.num_envs} actions, got {len(actions)}"

        # Send actions to all workers
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))

        # Collect results
        results = []
        for i, remote in enumerate(self.remotes):
            result = remote.recv()
            if isinstance(result, tuple) and len(result) == 2 and result[0] == 'error':
                raise RuntimeError(f"Worker {i} error:\n{result[1]}")
            results.append(result)
        obs, rewards, terminated, truncated, infos = zip(*results)

        return list(obs), list(rewards), list(terminated), list(truncated), list(infos)

    def reset(self, scene_indices: List[int] = None):
        """
        Reset environments.

        Args:
            scene_indices: Optional list of environment indices to reset.
                          If None, resets all environments.

        Returns:
            observations: List of observations
            infos: List of info dicts
        """
        if scene_indices is None:
            scene_indices = list(range(self.num_envs))

        # Send reset command
        for idx in scene_indices:
            self.remotes[idx].send(('reset', None))

        # Collect results
        results = []
        for idx in scene_indices:
            result = self.remotes[idx].recv()
            if isinstance(result, tuple) and len(result) == 2 and result[0] == 'error':
                raise RuntimeError(f"Worker {idx} error:\n{result[1]}")
            results.append(result)
        obs, infos = zip(*results)

        # For non-reset envs, return None
        full_obs = [None] * self.num_envs
        full_infos = [None] * self.num_envs
        for i, idx in enumerate(scene_indices):
            full_obs[idx] = obs[i]
            full_infos[idx] = infos[i]

        if len(scene_indices) == self.num_envs:
            return list(obs), list(infos)
        else:
            return full_obs, full_infos

    def close(self):
        """Close all worker processes."""
        if self.closed:
            return

        # Tell workers to exit
        for remote in self.remotes:
            remote.send(('close', None))

        # Give workers a moment to exit cleanly, then force-terminate stragglers
        for process in self.processes:
            process.join(timeout=5)
        for process in self.processes:
            if process.is_alive():
                process.terminate()

        # Close pipes to free FDs
        for remote in self.remotes:
            remote.close()

        self.closed = True

    def get_action_space_dims(self):
        """Get action space dimensions."""
        return self._action_space_dims

    @property
    def valid_direction_indices(self):
        """Get valid direction indices from first environment (for curriculum masking)."""
        return self.get_attr("valid_direction_indices", indices=[0])[0]

    @property
    def valid_length_indices(self):
        """Get valid length indices from first environment (for curriculum masking)."""
        return self.get_attr("valid_length_indices", indices=[0])[0]

    @property
    def curriculum_stage(self):
        """Get current curriculum stage from first environment."""
        return self.get_attr("curriculum_stage", indices=[0])[0]

    def get_attr(self, attr_name: str, indices: List[int] = None):
        """
        Get attribute from environments.

        Args:
            attr_name: Name of attribute to get
            indices: Optional list of environment indices. If None, gets from all.

        Returns:
            List of attribute values
        """
        if indices is None:
            indices = list(range(self.num_envs))

        for idx in indices:
            self.remotes[idx].send(('get_attr', attr_name))

        return [self.remotes[idx].recv() for idx in indices]

    def set_attr(self, attr_name: str, values: List[Any], indices: List[int] = None):
        """
        Set attribute in environments.

        Args:
            attr_name: Name of attribute to set
            values: List of values to set (one per environment)
            indices: Optional list of environment indices. If None, sets in all.
        """
        if indices is None:
            indices = list(range(self.num_envs))

        assert len(values) == len(indices), "Number of values must match number of indices"

        for idx, value in zip(indices, values):
            self.remotes[idx].send(('set_attr', (attr_name, value)))

        # Wait for confirmation
        for idx in indices:
            self.remotes[idx].recv()

    def call_method(self, method_name: str, args: List[Any] = None, kwargs: dict = None, indices: List[int] = None):
        """
        Call a method on environments.

        Args:
            method_name: Name of method to call
            args: List of positional arguments for the method (default: [])
            kwargs: Dictionary of keyword arguments for the method (default: {})
            indices: Optional list of environment indices. If None, calls on all.

        Returns:
            List of return values from the method calls
        """
        if indices is None:
            indices = list(range(self.num_envs))
        if args is None:
            args = []
        if kwargs is None:
            kwargs = {}

        for idx in indices:
            self.remotes[idx].send(('call_method', (method_name, args, kwargs)))

        return [self.remotes[idx].recv() for idx in indices]

    def __len__(self):
        return self.num_envs

    def __del__(self):
        if not self.closed:
            self.close()


def make_vec_env(env_fn: callable, n_envs: int = 4, seed: int = 42, **env_kwargs) -> VecEnv:
    """
    Convenience function to create a vectorized environment.

    Args:
        env_fn: Function that creates a single environment (e.g., PrecomputedThorEnv)
        n_envs: Number of parallel environments
        seed: Base seed for reproducibility. Each worker gets seed + worker_id.
        **env_kwargs: Keyword arguments passed to env_fn

    Returns:
        VecEnv instance

    Example:
        from components.environments.precomputed_thor_env import PrecomputedThorEnv

        vec_env = make_vec_env(
            PrecomputedThorEnv,
            n_envs=4,
            seed=42,
            render=False,
            rho=0.001,
            max_actions=40
        )
    """
    env_fns = [lambda: env_fn(**env_kwargs) for _ in range(n_envs)]
    return VecEnv(env_fns, seed=seed)
