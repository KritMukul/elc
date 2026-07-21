

import os
import glob
import numpy as np
import pandas as pd
import cv2


def _resolve_columns(df):
    cols = {c.lower(): c for c in df.columns}

    def find(*keys):
        for k in keys:
            for lc, orig in cols.items():
                if k in lc:
                    return orig
        return None

    x_col = find("gazex", "fpogx", "x_pos", "x")
    y_col = find("gazey", "fpogy", "y_pos", "y")
    t_col = find("time", "timestamp")
    dur_col = find("duration", "fpogd")
    return x_col, y_col, t_col, dur_col


def load_gaze_trace(csv_path):
    df = pd.read_csv(csv_path)
    x_col, y_col, t_col, dur_col = _resolve_columns(df)
    if x_col is None or y_col is None:
        raise ValueError(f"Could not resolve gaze x/y columns in {csv_path}: {df.columns.tolist()}")

    df = df[[c for c in [x_col, y_col, t_col, dur_col] if c is not None]].dropna()
    df = df.rename(columns={x_col: "x", y_col: "y"})
    if t_col:
        df = df.rename(columns={t_col: "t"})
    if dur_col:
        df = df.rename(columns={dur_col: "dur"})
    return df


def gaze_to_images(df, img_size=224, screen_w=1920, screen_h=1080, sigma=25):
    """
    Build (heatmap, scanpath) uint8 images of shape [img_size, img_size].
    """
    x = df["x"].to_numpy(dtype=np.float32)
    y = df["y"].to_numpy(dtype=np.float32)

    # normalize gaze coords to [0, img_size)
    x_norm = np.clip(x / max(screen_w, 1e-6), 0, 1) * (img_size - 1)
    y_norm = np.clip(y / max(screen_h, 1e-6), 0, 1) * (img_size - 1)

    # --- heatmap: accumulate points then Gaussian blur ---
    heat = np.zeros((img_size, img_size), dtype=np.float32)
    for xi, yi in zip(x_norm.astype(int), y_norm.astype(int)):
        heat[yi, xi] += 1.0
    heat = cv2.GaussianBlur(heat, (0, 0), sigmaX=sigma)
    if heat.max() > 0:
        heat = heat / heat.max()
    heat_img = (heat * 255).astype(np.uint8)

    # --- scanpath: draw fixation order as connected, size-scaled circles ---
    scan_img = np.zeros((img_size, img_size), dtype=np.uint8)
    dur = df["dur"].to_numpy(dtype=np.float32) if "dur" in df.columns else np.ones_like(x_norm)
    dur_norm = np.clip(dur / (dur.max() + 1e-8), 0.2, 1.0)
    pts = list(zip(x_norm.astype(int), y_norm.astype(int)))
    for i, (xi, yi) in enumerate(pts):
        radius = int(3 + 12 * dur_norm[i])
        cv2.circle(scan_img, (xi, yi), radius, color=180, thickness=-1)
        if i > 0:
            cv2.line(scan_img, pts[i - 1], pts[i], color=90, thickness=1)

    return heat_img, scan_img


def build_gaze_index(raw_root, metadata_csv, img_size=224, cache_dir=None):

    meta = pd.read_csv(metadata_csv)
    id_col = next((c for c in meta.columns if "id" in c.lower()), meta.columns[0])
    label_col = next((c for c in meta.columns
                       if any(k in c.lower() for k in ["diagnos", "group", "label", "asd"])),
                      None)
    if label_col is None:
        raise ValueError("Could not find a diagnosis/label column in metadata CSV")

    def to_label(v):
        s = str(v).strip().lower()
        return 1 if s in ("asd", "1", "autism", "yes", "true") else 0

    csv_dir = os.path.join(raw_root, "Eye-tracking Output")
    index = []
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)

    for _, row in meta.iterrows():
        pid = str(row[id_col]).strip()
        candidates = glob.glob(os.path.join(csv_dir, f"{pid}.csv"))
        if not candidates:
            continue
        label = to_label(row[label_col])

        img_path = os.path.join(cache_dir, f"{pid}.png") if cache_dir else None
        if img_path and os.path.exists(img_path):
            index.append({"participant": pid, "label": label, "image_path": img_path})
            continue

        df = load_gaze_trace(candidates[0])
        heat, scan = gaze_to_images(df, img_size=img_size)
        blank = np.zeros_like(heat)
        rgb = np.stack([heat, scan, blank], axis=-1)

        if img_path:
            cv2.imwrite(img_path, rgb)
        index.append({"participant": pid, "label": label,
                       "image_path": img_path, "_array": None if img_path else rgb})
    return index
