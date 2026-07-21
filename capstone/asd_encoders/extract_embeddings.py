"""
NOT TESTED 
"""

import argparse
import glob
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modality", choices=["eeg", "gaze", "gait"], required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)

    if args.modality == "eeg":
        from data.eeg_dataset import EEGDataset
        from models.eeg_conformer import EEGConformer
        import glob as _glob
        subject_dirs = sorted(_glob.glob(os.path.join(cfg["dataset_root"], "[PS]*")))
        all_subjects = {os.path.basename(d) for d in subject_dirs}
        model = EEGConformer(n_channels=cfg["model"]["n_channels"],
                              conv_emb_dim=cfg["model"]["conv_emb_dim"],
                              n_layers=cfg["model"]["n_layers"],
                              n_heads=cfg["model"]["n_heads"],
                              out_dim=cfg["model"]["out_dim"],
                              dropout=cfg["model"]["dropout"]).to(device)
        model.load_state_dict(ckpt["model_state"])
        ds = EEGDataset(cfg["dataset_root"], all_subjects, tasks=cfg["tasks"],
                         sfreq=cfg["sfreq"], win_sec=cfg["win_sec"], overlap=cfg["overlap"],
                         cache_dir=cfg["cache_dir"], train=False, augment=False)

    elif args.modality == "gaze":
        from data.gaze_dataset import GazeDataset
        from models.gaze_vit import GazeViT
        from preprocessing.gaze_preprocess import build_gaze_index
        index = build_gaze_index(cfg["raw_root"], cfg["metadata_csv"],
                                  img_size=cfg["img_size"], cache_dir=cfg["cache_dir"])
        all_participants = {r["participant"] for r in index}
        model = GazeViT(backbone=cfg["model"]["backbone"], pretrained=False,
                         out_dim=cfg["model"]["out_dim"], dropout=cfg["model"]["dropout"],
                         freeze_blocks=cfg["model"]["freeze_blocks"]).to(device)
        model.load_state_dict(ckpt["model_state"])
        ds = GazeDataset(cfg["raw_root"], cfg["metadata_csv"], all_participants,
                          img_size=cfg["img_size"], cache_dir=cfg["cache_dir"],
                          train=False, augment=False)

    else:  # gait
        from data.gait_dataset import GaitDataset
        from models.gait_stgcn import STGCN
        import glob as _glob
        subject_dirs = sorted(_glob.glob(os.path.join(cfg["dataset_root"], "[PS]*")))
        all_subjects = {os.path.basename(d) for d in subject_dirs}
        n_joints = ckpt.get("n_joints")
        model = STGCN(n_joints=n_joints, in_channels=cfg["model"]["in_channels"],
                      hidden=cfg["model"]["hidden"], out_dim=cfg["model"]["out_dim"],
                      dropout=cfg["model"]["dropout"]).to(device)
        model.load_state_dict(ckpt["model_state"])
        ds = GaitDataset(cfg["dataset_root"], all_subjects, tasks=cfg["tasks"],
                          win_frames=cfg["win_frames"], overlap=cfg["overlap"],
                          cache_dir=cfg["cache_dir"], train=False, augment=False)

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    embs, labels, subjects = extract(model, loader, device)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(args.out, embeddings=embs, labels=labels, subjects=subjects)
    print(f"Saved {embs.shape[0]} embeddings of dim {embs.shape[1]} to {args.out}")


if __name__ == "__main__":
    main()
