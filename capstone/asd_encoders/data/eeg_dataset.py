import numpy as np
import torch
from torch.utils.data import Dataset

from preprocessing.eeg_preprocess import build_eeg_index
from data.augmentations import EEGAugment


def kwargs_from_cfg(cfg):
    """Every EEGDataset knob that comes from the config, in one place.

    Kept here so train_eeg.py and extract_embeddings.py cannot drift apart and
    build indices with different preprocessing (which would silently produce
    two different caches).
    """
    return dict(
        tasks=cfg["tasks"],
        sfreq=cfg["sfreq"],
        win_sec=cfg["win_sec"],
        overlap=cfg["overlap"],
        cache_dir=cfg["cache_dir"],
        condition=cfg.get("condition", "all"),
        force_filter=cfg.get("force_filter", False),
        clip_sd=cfg.get("clip_sd"),
    )


class EEGDataset(Dataset):


    def __init__(self, dataset_root, subjects, tasks=("walk", "dance"),
                 sfreq=250, win_sec=4.0, overlap=0.5, cache_dir="cache/eeg",
                 train=True, augment=True, condition="all",
                 force_filter=False, clip_sd=None):
        full_index = build_eeg_index(dataset_root, tasks=tasks, sfreq=sfreq,
                                      win_sec=win_sec, overlap=overlap,
                                      cache_dir=cache_dir, condition=condition,
                                      force_filter=force_filter, clip_sd=clip_sd)
        self.index = [r for r in full_index if r["subject"] in subjects]
        self.train = train
        self.augment = EEGAugment(sfreq=sfreq, p=0.5) if (train and augment) else None
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
        x = windows[rec["window_idx"]].copy()  # [channels, time]

        if self.augment is not None:
            x = self.augment(x)

        return {
            "x": torch.from_numpy(x).float(),
            "label": torch.tensor(rec["label"], dtype=torch.long),
            "subject": rec["subject"],
        }
