import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from preprocessing.gaze_preprocess import build_gaze_index
from data.augmentations import GazeAugment


class GazeDataset(Dataset):
    """Scanpath images, filtered to a set of stimulus ids.

    The stimulus is the split group: Saliency4ASD does not identify viewers
    across stimuli, so it is the only group the data supports. See
    preprocessing/gaze_preprocess.py for what that costs.
    """

    def __init__(self, raw_root, stimuli, img_size=224,
                 cache_dir="cache/gaze", train=True, augment=True):
        full_index = build_gaze_index(raw_root, img_size=img_size, cache_dir=cache_dir)
        stimuli = {int(s) for s in stimuli}
        self.index = [r for r in full_index if r["stimulus"] in stimuli]
        self.img_size = img_size
        self.aug = GazeAugment(img_size=img_size, p=0.5, train=(train and augment))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        rec = self.index[idx]
        if rec.get("image_path"):
            img = cv2.imread(rec["image_path"], cv2.IMREAD_COLOR)
            if img is None:
                raise RuntimeError(f"Could not read cached gaze image {rec['image_path']}")
        else:
            img = rec["_array"]

        img = self.aug(img)                       # HWC float32, normalized [-1,1]
        img = np.transpose(img, (2, 0, 1))         # CHW

        return {
            "x": torch.from_numpy(img).float(),
            "label": torch.tensor(rec["label"], dtype=torch.long),
            # the fusion stage groups by this key - for gaze that is the stimulus
            "subject": str(rec["stimulus"]),
        }
