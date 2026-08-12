"""
Dump frozen 512-D encoder embeddings for the fusion stage.

Pass every fold's checkpoint together with --out-of-fold and each checkpoint
contributes only the subjects it never trained on, concatenated into one .npz.
Extracting every subject from a single fold's checkpoint (the old behaviour)
hands the fusion stage embeddings that are ~80% in-sample, which inflates
fusion metrics no matter how cleanly the fusion split itself is done.
"""

import argparse
import os

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm


def extract(model, loader, device):
    model.eval()
    embs, labels, subjects = [], [], []
    with torch.no_grad():
        for batch in tqdm(loader):
            x = batch["x"].to(device)
            emb = model.forward_features(x)
            embs.append(emb.cpu().numpy())
            labels.extend(batch["label"].numpy().tolist())
            subjects.extend(batch["subject"])
    return np.concatenate(embs, axis=0), np.array(labels), np.array(subjects)


def all_subject_ids(modality, cfg):
    """Every subject/participant the modality knows about (the unfiltered default)."""
    import glob

    if modality == "gaze":
        from preprocessing.gaze_preprocess import build_gaze_index
        index = build_gaze_index(cfg["raw_root"], img_size=cfg["img_size"],
                                 cache_dir=cfg["cache_dir"])
        # gaze groups by stimulus image, not by viewer - see gaze_preprocess.py
        return {str(r["stimulus"]) for r in index}

    subject_dirs = sorted(glob.glob(os.path.join(cfg["dataset_root"], "[PS]*")))
    return {os.path.basename(d) for d in subject_dirs}


def build_model_and_dataset(modality, cfg, ckpt, device, subjects):
    """Rebuild the trained encoder and a dataset restricted to `subjects`."""
    if modality == "eeg":
        from data.eeg_dataset import EEGDataset
        from models.eeg_conformer import EEGConformer
        # channel count comes from the checkpoint (train_eeg.py infers it from
        # the recordings); the YAML value is only a fallback for old checkpoints
        n_channels = ckpt.get("n_channels", cfg["model"]["n_channels"])
        model = EEGConformer(n_channels=n_channels,
                             conv_emb_dim=cfg["model"]["conv_emb_dim"],
                             n_layers=cfg["model"]["n_layers"],
                             n_heads=cfg["model"]["n_heads"],
                             out_dim=cfg["model"]["out_dim"],
                             dropout=cfg["model"]["dropout"]).to(device)
        model.load_state_dict(ckpt["model_state"])
        ds = EEGDataset(cfg["dataset_root"], subjects, tasks=cfg["tasks"],
                        sfreq=cfg["sfreq"], win_sec=cfg["win_sec"], overlap=cfg["overlap"],
                        cache_dir=cfg["cache_dir"], train=False, augment=False)

    elif modality == "gaze":
        from data.gaze_dataset import GazeDataset
        from models.gaze_vit import GazeViT
        model = GazeViT(backbone=cfg["model"]["backbone"], pretrained=False,
                        out_dim=cfg["model"]["out_dim"], dropout=cfg["model"]["dropout"],
                        freeze_blocks=cfg["model"]["freeze_blocks"]).to(device)
        model.load_state_dict(ckpt["model_state"])
        ds = GazeDataset(cfg["raw_root"], subjects,
                         img_size=cfg["img_size"], cache_dir=cfg["cache_dir"],
                         train=False, augment=False)

    else:  # gait
        from data.gait_dataset import GaitDataset
        from models.gait_stgcn import STGCN
        model = STGCN(n_joints=ckpt["n_joints"], in_channels=cfg["model"]["in_channels"],
                      hidden=cfg["model"]["hidden"], out_dim=cfg["model"]["out_dim"],
                      dropout=cfg["model"]["dropout"]).to(device)
        model.load_state_dict(ckpt["model_state"])
        ds = GaitDataset(cfg["dataset_root"], subjects, tasks=cfg["tasks"],
                         win_frames=cfg["win_frames"], overlap=cfg["overlap"],
                         cache_dir=cfg["cache_dir"], train=False, augment=False)

    return model, ds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modality", choices=["eeg", "gaze", "gait"], required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True, nargs="+",
                        help="One checkpoint, or all fold checkpoints when using --out-of-fold.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--out-of-fold", action="store_true",
                        help="Each checkpoint emits only its own validation subjects, so no "
                             "embedding comes from an encoder that trained on that subject.")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not args.out_of_fold and len(args.checkpoint) > 1:
        raise SystemExit("Multiple checkpoints only make sense with --out-of-fold.")

    every_subject = all_subject_ids(args.modality, cfg)
    shards_e, shards_l, shards_s = [], [], []

    for ckpt_path in args.checkpoint:
        ckpt = torch.load(ckpt_path, map_location=device)

        if args.out_of_fold:
            val_subjects = ckpt.get("val_subjects")
            if val_subjects is None:
                raise SystemExit(
                    f"{ckpt_path} carries no 'val_subjects'. It predates the out-of-fold "
                    "change - retrain with the current train_*.py, or drop --out-of-fold "
                    "(and accept in-sample embeddings)."
                )
            subjects = set(val_subjects)
        else:
            subjects = every_subject

        model, ds = build_model_and_dataset(args.modality, cfg, ckpt, device, subjects)
        if len(ds) == 0:
            print(f"[warn] {ckpt_path}: no samples for {len(subjects)} subject(s), skipping.")
            continue

        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers)
        e, l, s = extract(model, loader, device)
        print(f"{os.path.basename(ckpt_path)}: {e.shape[0]} embeddings "
              f"from {len(set(s.tolist()))} subject(s)")
        shards_e.append(e)
        shards_l.append(l)
        shards_s.append(s)

    if not shards_e:
        raise SystemExit("No embeddings extracted - every checkpoint yielded an empty dataset.")

    embs = np.concatenate(shards_e, axis=0)
    labels = np.concatenate(shards_l, axis=0)
    subjects_out = np.concatenate(shards_s, axis=0)

    if args.out_of_fold:
        # folds partition the subjects, so no subject may appear in two shards
        seen = [set(s.tolist()) for s in shards_s]
        for i in range(len(seen)):
            for j in range(i + 1, len(seen)):
                overlap = seen[i] & seen[j]
                assert not overlap, (
                    f"Checkpoints {args.checkpoint[i]} and {args.checkpoint[j]} both emitted "
                    f"subjects {sorted(overlap)} - the fold splits are not disjoint."
                )
        missing = every_subject - set(subjects_out.tolist())
        if missing:
            print(f"[warn] {len(missing)} subject(s) never appeared in any fold's validation "
                  f"set: {sorted(missing)}")

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    np.savez(args.out, embeddings=embs, labels=labels, subjects=subjects_out)
    print(f"Saved {embs.shape[0]} embeddings of dim {embs.shape[1]} "
          f"({len(set(subjects_out.tolist()))} subjects) to {args.out}")


if __name__ == "__main__":
    main()
