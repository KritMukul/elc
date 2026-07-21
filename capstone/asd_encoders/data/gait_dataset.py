import numpy as np
import torch
from torch.utils.data import Dataset

from preprocessing.gait_preprocess import build_gait_index
from data.augmentations import GaitAugment


class GaitDataset(Dataset):
    def __init__(self, dataset_root, subjects, tasks=("walk", "dance"),
                 win_frames=150, overlap=0.5, cache_dir="cache/gait",
                 train=True, augment=True):
        full_index = build_gait_index(dataset_root, tasks=tasks,
                                       win_frames=win_frames, overlap=overlap,
                                       cache_dir=cache_dir)
        self.index = [r for r in full_index if r["subject"] in subjects]
        self.aug = GaitAugment(p=0.5) if (train and augment) else None
        self._cache = {}

    def __len__(self):
        return len(self.index)

    def _get_windows(self, cache_path):
        if cache_path not in self._cache:
            self._cache[cache_path] = np.load(cache_path)
        return self._cache[cache_path]

    def __getitem__(self, idx):
        rec = self.index[idx]
        windows = self._get_windows(rec["cache_path"])
        x = windows[rec["window_idx"]].copy()  # [T, V, C]

        if self.aug is not None:
            x = self.aug(x)

        return {
            "x": torch.from_numpy(x).float(),
            "label": torch.tensor(rec["label"], dtype=torch.long),
            "subject": rec["subject"],
        }
