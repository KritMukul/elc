"""
EEG loading for EEGLAB-preprocessed recordings (.set + .fdt).

Layout expected under dataset_root:

    <dataset_root>/
        P1/dance/<...>/eegData_all.set   (+ .fdt)
           walk/ <...>/eegData_all.set
        S1/...

The nesting between the task folder and the .set file is discovered rather
than hardcoded, since it varies by dataset export.

These files are already preprocessed, so this module does NOT blindly re-run a
filter chain over them - re-filtering already-filtered data distorts it. What
EEGLAB recorded in the file header (highpass/lowpass/sfreq) is read back and
only the missing steps are applied. Set `force_filter: true` in the config to
override that and filter regardless.
"""

import glob
import os

import numpy as np
import mne

mne.set_log_level("ERROR")


def discover_eeg_file(subject_dir, task, condition="all"):
    """The single eegData_<condition>.set under this subject/task, at any depth.

    Returns None when the task is absent. Raises when more than one matches -
    a dyadic recording can put a partner's data in a sibling folder, and
    silently concatenating two people's EEG would be worse than stopping.
    """
    pattern = os.path.join(subject_dir, task, "**", f"eegData_{condition}.set")
    matches = sorted(glob.glob(pattern, recursive=True))

    # also allow the file to sit directly in the task folder
    direct = os.path.join(subject_dir, task, f"eegData_{condition}.set")
    if os.path.exists(direct) and direct not in matches:
        matches.append(direct)

    if not matches:
        return None
    if len(matches) > 1:
        raise SystemExit(
            f"Found {len(matches)} '{condition}' recordings under "
            f"{os.path.join(subject_dir, task)}:\n  "
            + "\n  ".join(matches)
            + "\nOne subject/task must map to one recording. If these are different "
              "people (e.g. a dyadic session), narrow the path so only the participant's "
              "own folder matches."
        )
    return matches[0]


def load_eeglab_raw(path, target_sfreq=None):
    """Read a .set/.fdt pair, resampled to target_sfreq when it differs."""
    raw = mne.io.read_raw_eeglab(path, preload=True, verbose=False)
    native = float(raw.info["sfreq"])
    if target_sfreq and abs(native - target_sfreq) > 1e-6:
        raw.resample(target_sfreq, verbose=False)
    return raw, native


def preprocess_raw_eeg(raw, l_freq=1.0, h_freq=45.0, notch=50.0,
                       force_filter=False, clip_sd=None):
    """Apply only the steps the file has not already had.

    EEGLAB writes the achieved passband into the file header, so a recording
    that is already 1-45 Hz is left alone instead of being filtered twice.
    """
    applied = []
    have_hp = raw.info.get("highpass") or 0.0
    have_lp = raw.info.get("lowpass") or raw.info["sfreq"] / 2.0

    need_hp = force_filter or (l_freq is not None and have_hp < l_freq - 1e-6)
    need_lp = force_filter or (h_freq is not None and have_lp > h_freq + 1e-6)

    if need_hp or need_lp:
        raw.filter(l_freq=l_freq if need_hp else None,
                   h_freq=h_freq if need_lp else None,
                   fir_design="firwin", verbose=False)
        applied.append(f"bandpass({l_freq if need_hp else '-'}, {h_freq if need_lp else '-'})")

    # a lowpass at or below the line frequency already removes it
    if notch and (force_filter or (h_freq is None or h_freq > notch)):
        raw.notch_filter(freqs=notch, verbose=False)
        applied.append(f"notch({notch})")

    data = raw.get_data().astype(np.float32)

    if clip_sd:
        thresh = clip_sd * np.std(data)
        data = np.clip(data, -thresh, thresh)
        applied.append(f"clip({clip_sd}sd)")

    return data, applied


def window_and_normalize(data, sfreq=250, win_sec=4.0, overlap=0.5):
    """Per-window, per-channel z-scored windows -> [n_windows, channels, samples]."""
    win_len = int(win_sec * sfreq)
    step = max(1, int(win_len * (1 - overlap)))
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
                    win_sec=4.0, overlap=0.5, cache_dir=None, condition="all",
                    force_filter=False, clip_sd=None):
    """One record per window: {subject, task, label, cache_path, window_idx}."""
    index = []
    subject_dirs = sorted(glob.glob(os.path.join(dataset_root, "[PS]*")))
    described = False

    for sdir in subject_dirs:
        if not os.path.isdir(sdir):
            continue
        subj = os.path.basename(sdir)
        label = 1 if subj.startswith("P") else 0

        for task in tasks:
            cache_path = None
            if cache_dir:
                os.makedirs(cache_dir, exist_ok=True)
                # every parameter that changes the array is part of the key, so a
                # config change cannot silently reuse a stale cache
                cache_path = os.path.join(
                    cache_dir,
                    f"{subj}_{task}_{condition}_w{win_sec}_o{overlap}_fs{sfreq}.npy")

            if cache_path and os.path.exists(cache_path):
                windows = np.load(cache_path)
            else:
                path = discover_eeg_file(sdir, task, condition=condition)
                if path is None:
                    continue

                raw, native_sfreq = load_eeglab_raw(path, target_sfreq=sfreq)
                data, applied = preprocess_raw_eeg(
                    raw, force_filter=force_filter, clip_sd=clip_sd)

                if not described:
                    print(f"EEG: {len(raw.ch_names)} channels, native {native_sfreq:g} Hz "
                          f"-> {sfreq} Hz, header passband "
                          f"{raw.info.get('highpass')}-{raw.info.get('lowpass')} Hz")
                    print(f"     channels: {raw.ch_names}")
                    print(f"     steps applied here: {applied if applied else 'none (already preprocessed)'}")
                    described = True

                windows = window_and_normalize(data, sfreq=sfreq, win_sec=win_sec,
                                               overlap=overlap)
                if cache_path is not None:
                    np.save(cache_path, windows)

            for i in range(windows.shape[0]):
                index.append({
                    "subject": subj, "task": task, "label": label,
                    "cache_path": cache_path, "window_idx": i,
                })

    return index
