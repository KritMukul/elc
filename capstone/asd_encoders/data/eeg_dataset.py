import numpy as np
import torch
from torch.utils.data import Dataset

from preprocessing.eeg_preprocess import build_eeg_index
from data.augmentations import EEGAugment


class EEGDataset(Dataset):


    def __init__(self, dataset_root, subjects, tasks=("walk", "dance"),
                 sfreq=250, win_sec=4.0, overlap=0.5, cache_dir="cache/eeg",
                 train=True, augment=True):
        full_index = build_eeg_index(dataset_root, tasks=tasks, sfreq=sfreq,
                                      win_sec=win_sec, overlap=overlap,
                                      cache_dir=cache_dir)
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
