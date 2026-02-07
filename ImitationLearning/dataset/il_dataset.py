import os
import pickle
from glob import glob

import torch
from torch.utils.data import Dataset

from components.graph.global_graph import GlobalSceneGraph
from components.graph.scene_graph import SceneGraph


def list_all_pkl_files(data_dir):
    """
    List all pickle files in directory recursively.

    Args:
        data_dir: Root directory to search

    Returns:
        Sorted list of pickle file paths
    """
    pattern = os.path.join(data_dir, "**", "*.pkl")
    return sorted(glob(pattern, recursive=True))


class ImitationLearningDataset(Dataset):
    """PyTorch Dataset for imitation learning episodes."""
    def __init__(self, data_dir, seq_len=8):
        """
        Initialize IL dataset from pickle files.

        Args:
            data_dir: Directory containing episode pickle files
            seq_len: Sequence length for sliding window

        Attributes:
            windows: List of dicts with file paths and indices
            num_actions: Number of actions in the environment
        """
        super().__init__()
        self.seq_len = seq_len
        self.windows = []  # list of {"path": ..., "start": ...}
        self.num_actions = None

        for file_path in list_all_pkl_files(data_dir):
            with open(file_path, "rb") as f:
                ep = pickle.load(f)
            num_steps = len(ep)
            if self.num_actions is None:
                self.num_actions = ep[0].get("num_actions", None)
            for t in range(0, num_steps, seq_len):
                self.windows.append({"path": file_path, "start_idx": t, "length": min(seq_len, num_steps - t)})

    def __len__(self):
        """Return number of sequences in dataset."""
        return len(self.windows)

    def __getitem__(self, idx):
        """
        Get a sequence of observations and actions.

        Args:
            idx: Sequence index

        Returns:
            Tuple of (observations, last_actions, target_actions, sequence_length)
        """
        entry = self.windows[idx]
        with open(entry["path"], "rb") as f:
            ep = pickle.load(f)
        chunk = ep[entry["start_idx"] : entry["start_idx"] + entry["length"]]

        obs, last_act, tgt_act = [], [], []
        for s in chunk:
            obs_copy = s["obs"]

            # Convert dicts back to SceneGraph objects.
            # New layout: [rgb, depth, local_sg, global_sg]
            if len(obs_copy.state) > 2 and isinstance(obs_copy.state[2], dict):
                obs_copy.state[2] = SceneGraph.from_dict(obs_copy.state[2])
            if len(obs_copy.state) > 3 and isinstance(obs_copy.state[3], dict):
                obs_copy.state[3] = GlobalSceneGraph.from_dict(obs_copy.state[3])

            # Backward compatibility: some older data saved SGs at indices 1/2
            if isinstance(obs_copy.state[1], dict):
                obs_copy.state[1] = SceneGraph.from_dict(obs_copy.state[1])
            if len(obs_copy.state) > 2 and isinstance(obs_copy.state[2], dict):
                # If still dict after previous conversions, treat as global SG
                obs_copy.state[2] = GlobalSceneGraph.from_dict(obs_copy.state[2])

            obs.append(obs_copy)
            last_act.append(s["last_action"])
            tgt_act.append(torch.tensor(s["obs"].info["action"], dtype=torch.long))

        return obs, last_act, tgt_act, len(chunk)
