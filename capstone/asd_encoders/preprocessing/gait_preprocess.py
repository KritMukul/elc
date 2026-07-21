

import re
import os
import glob
import numpy as np
import scipy.io as sio


def discover_blocks(subject_dir, task, prefix):
    pattern = os.path.join(subject_dir, task, f"{prefix}_*.mat")
    files = glob.glob(pattern)

    def block_num(f):
        m = re.search(r"(\d+)\.mat$", os.path.basename(f))
        return int(m.group(1)) if m else 0

    return sorted(files, key=block_num)


def _load_mocap_mat(path):
  
    mat = sio.loadmat(path, simplify_cells=True)
    best = None
    for k, v in mat.items():
        if k.startswith("__"):
            continue
        arr = np.asarray(v)
        if arr.dtype.kind not in "fc":
            continue
        if best is None or arr.size > best.size:
            best = arr

    if best is None:
        raise ValueError(f"No numeric marker array found in {path}")

    if best.ndim == 3:

        coord_axis = next((i for i, s in enumerate(best.shape) if s == 3), 2)
        best = np.moveaxis(best, coord_axis, -1)
        if best.shape[0] < best.shape[1]:

            pass
        return best.astype(np.float32)

    if best.ndim == 2:

        if best.shape[1] % 3 == 0:
            v = best.shape[1] // 3
            return best.reshape(best.shape[0], v, 3).astype(np.float32)

        if best.shape[0] % 3 == 0:
            v = best.shape[0] // 3
            return best.T.reshape(-1, v, 3).astype(np.float32)

    raise ValueError(f"Could not interpret marker array shape {best.shape} in {path}")


def load_concatenated_gait(subject_dir, task):
    files = discover_blocks(subject_dir, task, "mdata")
    if not files:
        return None
    blocks = [_load_mocap_mat(f) for f in files]
    min_v = min(b.shape[1] for b in blocks)
    blocks = [b[:, :min_v, :] for b in blocks]
    return np.concatenate(blocks, axis=0)  


def fill_missing_markers(x):

    x = x.copy()
    t = x.shape[0]
    for v in range(x.shape[1]):
        for c in range(x.shape[2]):
            series = x[:, v, c]
            nan_mask = np.isnan(series)
            if nan_mask.all():
                continue
            if nan_mask.any():
                idx = np.arange(t)
                series[nan_mask] = np.interp(idx[nan_mask], idx[~nan_mask], series[~nan_mask])
                x[:, v, c] = series
    return x


def normalize_skeleton(x):

    center = np.nanmean(x, axis=1, keepdims=True)
    x = x - center
    scale = np.nanstd(x) + 1e-8
    return (x / scale).astype(np.float32)


def resample_sequence(x, target_len):

    t, v, c = x.shape
    if t == target_len:
        return x
    orig_idx = np.linspace(0, t - 1, t)
    new_idx = np.linspace(0, t - 1, target_len)
    out = np.empty((target_len, v, c), dtype=np.float32)
    for vi in range(v):
        for ci in range(c):
            out[:, vi, ci] = np.interp(new_idx, orig_idx, x[:, vi, ci])
    return out


def window_sequence(x, win_frames=150, overlap=0.5):

    step = int(win_frames * (1 - overlap))
    windows = []
    for start in range(0, max(1, x.shape[0] - win_frames + 1), step):
        windows.append(x[start:start + win_frames])
    if not windows:
        windows = [resample_sequence(x, win_frames)]
    return np.stack(windows, axis=0)


def build_gait_index(dataset_root, tasks=("walk", "dance"), win_frames=150,
                      overlap=0.5, cache_dir=None):
    index = []
    subject_dirs = sorted(glob.glob(os.path.join(dataset_root, "[PS]*")))
    for sdir in subject_dirs:
        subj = os.path.basename(sdir)
        label = 1 if subj.startswith("P") else 0
        for task in tasks:
            cache_path = None
            if cache_dir:
                os.makedirs(cache_dir, exist_ok=True)
                cache_path = os.path.join(cache_dir, f"{subj}_{task}_gait.npy")

            if cache_path and os.path.exists(cache_path):
                windows = np.load(cache_path)
            else:
                raw = load_concatenated_gait(sdir, task)
                if raw is None:
                    continue
                raw = fill_missing_markers(raw)
                raw = normalize_skeleton(raw)
                windows = window_sequence(raw, win_frames=win_frames, overlap=overlap)
                if cache_path is not None:
                    np.save(cache_path, windows)

            for i in range(windows.shape[0]):
                index.append({
                    "subject": subj, "task": task, "label": label,
                    "cache_path": cache_path, "window_idx": i,
                })
    return index
