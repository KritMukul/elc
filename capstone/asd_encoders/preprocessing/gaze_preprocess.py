"""
Saliency4ASD (ICME 2019) scanpath -> heatmap/scanpath image preprocessing.

Layout this expects under raw_root:

    TrainingData/
        ASD/ASD_scanpath_<stimulus>.txt
        TD/ TD_scanpath_<stimulus>.txt
        Images/<stimulus>.png

Each scanpath file holds *every* viewer of one stimulus, concatenated, with
columns `Idx, x, y, duration`; a new viewer begins where Idx returns to 0. The
diagnosis is the folder name - there is no participant metadata file, and
individual viewers are not identifiable across stimuli.

Consequence for cross-validation: the only group the released data actually
supports is the stimulus image, so that is what build_gaze_index exposes as the
split key. A given viewer contributes scanpaths to many stimuli and therefore
appears on both sides of any split - state that limitation when reporting gaze
numbers; it cannot be engineered away from this data.
"""

import os
import glob
import re

import numpy as np
import pandas as pd
import cv2


SCANPATH_RE = re.compile(r"scanpath_(\d+)\.txt$", re.IGNORECASE)
GROUP_LABELS = {"ASD": 1, "TD": 0}

_SIZE_CACHE = {}


def stimulus_size(images_dir, stimulus_id):
    """(width, height) of a stimulus image.

    Fixation coordinates live in the stimulus' own pixel space and every
    stimulus has its own dimensions, so normalising by a fixed screen size
    squeezes every scanpath into a corner of the canvas.
    """
    if stimulus_id in _SIZE_CACHE:
        return _SIZE_CACHE[stimulus_id]

    matches = glob.glob(os.path.join(images_dir, f"{stimulus_id}.*"))
    size = None
    for path in sorted(matches):
        img = cv2.imread(path)
        if img is not None:
            size = (img.shape[1], img.shape[0])
            break

    _SIZE_CACHE[stimulus_id] = size
    return size


def load_scanpath_blocks(path):
    """Split one scanpath file into per-viewer DataFrames.

    Returns a list of frames with lowercase columns x, y and (when present)
    duration - one per viewer of this stimulus.
    """
    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower() for c in df.columns]

    missing = {"idx", "x", "y"} - set(df.columns)
    if missing:
        raise ValueError(
            f"{path}: missing column(s) {sorted(missing)}; found {list(df.columns)}"
        )

    df = df.dropna(subset=["idx", "x", "y"])
    if df.empty:
        return []

    idx = df["idx"].to_numpy()
    starts = np.flatnonzero(idx == 0)
    if starts.size == 0:
        return []

    rows = np.arange(starts[0], len(df))
    blocks = np.split(rows, starts[1:] - starts[0])
    return [df.iloc[b] for b in blocks if len(b) > 0]


def scanpath_to_images(block, width, height, img_size=224, sigma=25):
    """Build (heatmap, scanpath) uint8 images of shape [img_size, img_size]."""
    x = block["x"].to_numpy(dtype=np.float32)
    y = block["y"].to_numpy(dtype=np.float32)

    # coordinates are pixels in the stimulus' own frame
    x_norm = np.clip(x / max(width, 1e-6), 0, 1) * (img_size - 1)
    y_norm = np.clip(y / max(height, 1e-6), 0, 1) * (img_size - 1)
    xi_all = x_norm.astype(int)
    yi_all = y_norm.astype(int)

    # --- heatmap: accumulate fixations then Gaussian blur ---
    heat = np.zeros((img_size, img_size), dtype=np.float32)
    for xi, yi in zip(xi_all, yi_all):
        heat[yi, xi] += 1.0
    # zero-pad rather than reflect: fixation density outside the stimulus is
    # genuinely zero, and at this sigma reflection drags peaks toward the edges
    heat = cv2.GaussianBlur(heat, (0, 0), sigmaX=sigma, borderType=cv2.BORDER_CONSTANT)
    if heat.max() > 0:
        heat = heat / heat.max()
    heat_img = (heat * 255).astype(np.uint8)

    # --- scanpath: fixation order as connected, dwell-scaled circles ---
    scan_img = np.zeros((img_size, img_size), dtype=np.uint8)
    if "duration" in block.columns:
        dur = block["duration"].to_numpy(dtype=np.float32)
    else:
        dur = np.ones_like(x_norm)
    dur_norm = np.clip(dur / (dur.max() + 1e-8), 0.2, 1.0)

    pts = list(zip(xi_all, yi_all))
    for i, (xi, yi) in enumerate(pts):
        radius = int(3 + 12 * dur_norm[i])
        cv2.circle(scan_img, (int(xi), int(yi)), radius, color=180, thickness=-1)
        if i > 0:
            cv2.line(scan_img, (int(pts[i - 1][0]), int(pts[i - 1][1])),
                     (int(xi), int(yi)), color=90, thickness=1)

    return heat_img, scan_img


def _stack_rgb(heat, scan):
    """Channel order is fixed here and written straight through cv2.imwrite.

    cv2 treats the array as BGR on write and gaze_dataset reads it back with
    IMREAD_COLOR without a colour conversion, so what the model sees is
    identical whether the sample came from cache or was built in memory.
    """
    return np.stack([heat, scan, np.zeros_like(heat)], axis=-1)


def build_gaze_index(raw_root, img_size=224, cache_dir=None):
    """One record per (group, stimulus, viewer) scanpath.

    `stimulus` is the cross-validation group - see the module docstring.
    """
    images_dir = os.path.join(raw_root, "Images")
    if not os.path.isdir(images_dir):
        raise SystemExit(
            f"No Images/ directory under {raw_root}. Stimulus dimensions are needed "
            "to place fixations; point raw_root at the TrainingData folder."
        )

    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)

    index = []
    missing_sizes = set()

    for group, label in sorted(GROUP_LABELS.items()):
        group_dir = os.path.join(raw_root, group)
        if not os.path.isdir(group_dir):
            raise SystemExit(f"Expected a {group}/ directory under {raw_root}.")

        for path in sorted(glob.glob(os.path.join(group_dir, "*scanpath_*.txt"))):
            m = SCANPATH_RE.search(os.path.basename(path))
            if not m:
                continue
            stimulus = int(m.group(1))

            size = stimulus_size(images_dir, stimulus)
            blocks = load_scanpath_blocks(path)
            if size is None:
                # fall back to this stimulus' own fixation extent rather than
                # inventing a screen size, and say so loudly
                missing_sizes.add(stimulus)
                if not blocks:
                    continue
                width = max(float(b["x"].max()) for b in blocks) + 1
                height = max(float(b["y"].max()) for b in blocks) + 1
            else:
                width, height = size

            for block_i, block in enumerate(blocks):
                # img_size is part of the key so a changed config cannot silently
                # reuse images rendered at the old resolution
                img_path = (os.path.join(cache_dir, f"{group}_{stimulus}_{block_i}_{img_size}.png")
                            if cache_dir else None)

                if img_path and os.path.exists(img_path):
                    index.append({"stimulus": stimulus, "group": group, "block": block_i,
                                  "label": label, "image_path": img_path, "_array": None})
                    continue

                heat, scan = scanpath_to_images(block, width, height, img_size=img_size)
                rgb = _stack_rgb(heat, scan)
                if img_path:
                    cv2.imwrite(img_path, rgb)
                index.append({"stimulus": stimulus, "group": group, "block": block_i,
                              "label": label, "image_path": img_path,
                              "_array": None if img_path else rgb})

    if missing_sizes:
        print(f"[warn] No image found in {images_dir} for {len(missing_sizes)} stimulus id(s); "
              f"used the fixation bounding box instead: {sorted(missing_sizes)[:10]}"
              f"{' ...' if len(missing_sizes) > 10 else ''}")

    if not index:
        raise SystemExit(f"No scanpaths found under {raw_root}/{{ASD,TD}}.")

    return index
