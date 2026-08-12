"""Regression tests for the Saliency4ASD gaze preprocessing.

Self-contained: builds a synthetic TrainingData tree in a temp directory, so it
needs neither the real dataset nor a GPU nor torch - only numpy, pandas and cv2.

Run from capstone/asd_encoders/ :   python tests/test_gaze_preprocess.py
"""
import os
import shutil
import sys
import tempfile

import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from preprocessing.gaze_preprocess import (      # noqa: E402
    build_gaze_index, load_scanpath_blocks, scanpath_to_images, stimulus_size,
)

FAILURES = []


def check(name, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not cond:
        FAILURES.append(name)


# stimulus id -> (width, height); deliberately different, since coordinates are
# in each stimulus' own pixel space
SIZES = {1: (800, 600), 2: (640, 480)}

# stimulus -> group -> list of viewer blocks, each a list of (idx, x, y, duration)
SCANPATHS = {
    1: {
        "ASD": [[(0, 700, 100, 250)],
                [(0, 10, 10, 30), (1, 20, 20, 40), (2, 30, 30, 50)]],
        "TD":  [[(0, 400, 300, 120), (1, 410, 310, 90)]],
    },
    2: {
        "ASD": [[(0, 320, 240, 100), (1, 330, 250, 100)]],
        "TD":  [[(0, 5, 5, 10)], [(0, 600, 400, 60)], [(0, 100, 100, 20), (1, 110, 110, 20)]],
    },
}


def build_tree(root):
    os.makedirs(os.path.join(root, "Images"))
    for sid, (w, h) in SIZES.items():
        cv2.imwrite(os.path.join(root, "Images", f"{sid}.png"),
                    np.full((h, w, 3), 128, dtype=np.uint8))

    for group in ("ASD", "TD"):
        os.makedirs(os.path.join(root, group))
        for sid, per_group in SCANPATHS.items():
            lines = ["Idx, x, y, duration"]
            for block in per_group[group]:
                lines.extend(",".join(str(v) for v in row) for row in block)
            with open(os.path.join(root, group, f"{group}_scanpath_{sid}.txt"), "w") as f:
                f.write("\n".join(lines) + "\n")


def main():
    root = tempfile.mkdtemp(prefix="s4asd_")
    cache = tempfile.mkdtemp(prefix="s4asd_cache_")
    try:
        build_tree(root)
        images_dir = os.path.join(root, "Images")

        # stimulus dimensions are read, not assumed
        check("stimulus_size reads real dims", stimulus_size(images_dir, 1) == (800, 600),
              f"got {stimulus_size(images_dir, 1)}")

        # a viewer's record starts where Idx returns to 0
        blocks = load_scanpath_blocks(os.path.join(root, "TD", "TD_scanpath_2.txt"))
        check("block count from Idx resets", len(blocks) == 3, f"got {len(blocks)}")
        check("block sizes", [len(b) for b in blocks] == [1, 1, 2],
              f"got {[len(b) for b in blocks]}")

        # 'Idx' also contains the letter x - the old substring column matcher
        # returned it as the x column, feeding fixation order in as a coordinate
        b = load_scanpath_blocks(os.path.join(root, "ASD", "ASD_scanpath_1.txt"))[0]
        check("x column is x, not Idx", int(b["x"].iloc[0]) == 700, f"got {int(b['x'].iloc[0])}")
        check("y column is y", int(b["y"].iloc[0]) == 100, f"got {int(b['y'].iloc[0])}")

        # the fixation lands where its coordinates say, scaled by this stimulus
        img_size = 224
        heat, scan = scanpath_to_images(b, 800, 600, img_size=img_size)
        row, col = np.unravel_index(int(np.argmax(heat)), heat.shape)
        exp_col = int(round(700 / 800 * (img_size - 1)))
        exp_row = int(round(100 / 600 * (img_size - 1)))
        check("heatmap peak at the right pixel",
              abs(col - exp_col) <= 1 and abs(row - exp_row) <= 1,
              f"peak=({row},{col}) expected~({exp_row},{exp_col})")
        check("scanpath image non-empty", scan.max() > 0, f"max={scan.max()}")
        check("peak is NOT at column 0 (the old Idx bug)", col > 5, f"col={col}")

        # one record per (group, stimulus, viewer)
        index = build_gaze_index(root, img_size=img_size, cache_dir=cache)
        expected = sum(len(v) for per in SCANPATHS.values() for v in per.values())
        check("record count", len(index) == expected, f"got {len(index)} expected {expected}")
        check("stimuli discovered", sorted({r['stimulus'] for r in index}) == [1, 2],
              f"got {sorted({r['stimulus'] for r in index})}")

        asd = [r for r in index if r["group"] == "ASD"]
        td = [r for r in index if r["group"] == "TD"]
        check("ASD labelled 1", all(r["label"] == 1 for r in asd), f"n={len(asd)}")
        check("TD labelled 0", all(r["label"] == 0 for r in td), f"n={len(td)}")
        # both classes on every stimulus is what makes a plain KFold over
        # stimuli safe in train_gaze.py - no stratification needed
        check("both classes on every stimulus",
              all({r["label"] for r in index if r["stimulus"] == s} == {0, 1} for s in (1, 2)))

        # img_size is part of the cache key, so changing it cannot silently
        # reuse images rendered at the old resolution
        names = os.listdir(cache)
        check("cache files written", len(names) == expected, f"got {len(names)}")
        check("img_size in cache filename", all(f"_{img_size}.png" in n for n in names),
              f"e.g. {names[0]}")

        index2 = build_gaze_index(root, img_size=160, cache_dir=cache)
        check("different img_size writes new files",
              len(os.listdir(cache)) == 2 * expected, f"got {len(os.listdir(cache))}")
        check("re-index still consistent", len(index2) == expected, f"got {len(index2)}")

        index3 = build_gaze_index(root, img_size=img_size, cache_dir=cache)
        check("cached run matches fresh run", len(index3) == len(index),
              f"{len(index3)} vs {len(index)}")
        check("cached records carry image_path", all(r["image_path"] for r in index3))

    finally:
        shutil.rmtree(root, ignore_errors=True)
        shutil.rmtree(cache, ignore_errors=True)

    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {FAILURES}")
        sys.exit(1)
    print(f"all {19} gaze preprocessing checks passed")


if __name__ == "__main__":
    main()
