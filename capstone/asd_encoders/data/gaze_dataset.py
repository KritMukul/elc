import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from preprocessing.gaze_preprocess import build_gaze_index
from data.augmentations import GazeAugment


class GazeDataset(Dataset):
    def __init__(self, raw_root, metadata_csv, participants, img_size=224,
                 cache_dir="cache/gaze", train=True, augment=True):
        full_index = build_gaze_index(raw_root, metadata_csv, img_size=img_size,
                                       cache_dir=cache_dir)
        self.index = [r for r in full_index if r["participant"] in participants]
        self.img_size = img_size
        self.aug = GazeAugment(img_size=img_size, p=0.5, train=(train and augment))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        rec = self.index[idx]
        if rec.get("image_path") and rec["image_path"] is not None:
            img = cv2.imread(rec["image_path"])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = rec["_array"]

        img = self.aug(img)                       # HWC float32, normalized [-1,1]
        img = np.transpose(img, (2, 0, 1))         # CHW

        return {
            "x": torch.from_numpy(img).float(),
            "label": torch.tensor(rec["label"], dtype=torch.long),
            "subject": rec["participant"],
        }
