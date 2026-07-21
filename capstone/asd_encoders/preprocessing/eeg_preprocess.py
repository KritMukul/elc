

import re
import glob
import os
import numpy as np
import scipy.io as sio
import mne


def _load_mat_eeg(path):

    mat = sio.loadmat(path, simplify_cells=True)

    candidate = None
    for k, v in mat.items():
        if k.startswith("__"):
            continue
        arr = np.asarray(v)
        if arr.ndim == 2 and arr.size > (candidate.size if candidate is not None else 0):
            candidate = arr
    if candidate is None:
        raise ValueError(f"No usable EEG array found in {path}")

    if candidate.shape[0] > candidate.shape[1]:
        candidate = candidate.T
    return candidate.astype(np.float64)


def discover_eeg_blocks(subject_dir, task):

    pattern = os.path.join(subject_dir, task, "eeg_*.mat")
    files = glob.glob(pattern)

    def block_num(f):
        m = re.search(r"(\d+)\.mat$", os.path.basename(f))
        return int(m.group(1)) if m else 0

    return sorted(files, key=block_num)


def load_concatenated_eeg(subject_dir, task):

    files = discover_eeg_blocks(subject_dir, task)
    if not files:
        return None
    blocks = [_load_mat_eeg(f) for f in files]
    min_ch = min(b.shape[0] for b in blocks)
    blocks = [b[:min_ch] for b in blocks] 
    return np.concatenate(blocks, axis=1)


def preprocess_raw_eeg(data, sfreq=250, l_freq=1.0, h_freq=45.0, notch=50.0,
                        ch_names=None):

    n_ch = data.shape[0]
    if ch_names is None:
        ch_names = [f"EEG{i}" for i in range(n_ch)]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data, info, verbose=False)

    raw.filter(l_freq=l_freq, h_freq=h_freq, fir_design="firwin", verbose=False)
    raw.notch_filter(freqs=notch, verbose=False)
    raw.set_eeg_reference("average", verbose=False)

    clean = raw.get_data()

    
    thresh = 5 * np.std(clean)
    clean = np.clip(clean, -thresh, thresh)
    return clean.astype(np.float32)


def window_and_normalize(data, sfreq=250, win_sec=4.0, overlap=0.5):

    win_len = int(win_sec * sfreq)
    step = int(win_len * (1 - overlap))
    n_ch, n_t = data.shape
    windows = []
    for start in range(0, n_t - win_len + 1, step):
        w = data[:, start:start + win_len]
        mu = w.mean(axis=1, keepdims=True)
        sd = w.std(axis=1, keepdims=True) + 1e-8
        windows.append(((w - mu) / sd).astype(np.float32))
    if not windows:
        return np.empty((0, n_ch, win_len), dtype=np.float32)
    return np.stack(windows, axis=0)


def build_eeg_index(dataset_root, tasks=("walk", "dance"), sfreq=250,
                     win_sec=4.0, overlap=0.5, cache_dir=None):
  
    index = []
    subject_dirs = sorted(glob.glob(os.path.join(dataset_root, "[PS]*")))
    for sdir in subject_dirs:
        subj = os.path.basename(sdir)
        label = 1 if subj.startswith("P") else 0
        for task in tasks:
            cache_path = None
            if cache_dir:
                os.makedirs(cache_dir, exist_ok=True)
                cache_path = os.path.join(cache_dir, f"{subj}_{task}_eeg.npy")

            if cache_path and os.path.exists(cache_path):
                windows = np.load(cache_path)
            else:
                raw = load_concatenated_eeg(sdir, task)
                if raw is None:
                    continue
                clean = preprocess_raw_eeg(raw, sfreq=sfreq)
                windows = window_and_normalize(clean, sfreq=sfreq,
                                                win_sec=win_sec, overlap=overlap)
                if cache_path is not None:
                    np.save(cache_path, windows)

            for i in range(windows.shape[0]):
                index.append({
                    "subject": subj, "task": task, "label": label,
                    "cache_path": cache_path, "window_idx": i,
                })
    return index
