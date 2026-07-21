

import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.gaze_dataset import GazeDataset
from models.gaze_vit import GazeViT
from preprocessing.gaze_preprocess import build_gaze_index
from utils.metrics import compute_all_metrics, print_metrics
from utils.seed import set_seed


def get_participant_labels(raw_root, metadata_csv, img_size, cache_dir):
    index = build_gaze_index(raw_root, metadata_csv, img_size=img_size, cache_dir=cache_dir)
    participants = [r["participant"] for r in index]
    labels = [r["label"] for r in index]
    return np.array(participants), np.array(labels)


def run_epoch(model, loader, optimizer, criterion, device, scaler, train=True):
    model.train() if train else model.eval()
    total_loss, all_true, all_pred, all_prob = 0.0, [], [], []

    with torch.set_grad_enabled(train):
        for batch in tqdm(loader, leave=False):
            x = batch["x"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)

            if train:
                optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type="cuda", enabled=scaler is not None):
                logits, _ = model(x)
                loss = criterion(logits, y)

            if train:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            total_loss += loss.item() * x.size(0)
            probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
            preds = logits.argmax(dim=1).detach().cpu().numpy()
            all_true.extend(y.cpu().numpy().tolist())
            all_pred.extend(preds.tolist())
            all_prob.extend(probs.tolist())

    metrics = compute_all_metrics(all_true, all_pred, all_prob)
    metrics["loss"] = total_loss / max(1, len(loader.dataset))
    return metrics


def main(cfg_path):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["train"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(cfg["train"]["checkpoint_dir"], exist_ok=True)

    participants, labels = get_participant_labels(cfg["raw_root"], cfg["metadata_csv"],
                                                    cfg["img_size"], cfg["cache_dir"])

    skf = StratifiedKFold(n_splits=cfg["train"]["n_folds"], shuffle=True,
                           random_state=cfg["train"]["seed"])

    fold_metrics = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(participants, labels)):
        print(f"\n===== Gaze Fold {fold + 1}/{cfg['train']['n_folds']} =====")
        train_p = set(participants[train_idx])
        val_p = set(participants[val_idx])

        train_ds = GazeDataset(cfg["raw_root"], cfg["metadata_csv"], train_p,
                                img_size=cfg["img_size"], cache_dir=cfg["cache_dir"],
                                train=True, augment=True)
        val_ds = GazeDataset(cfg["raw_root"], cfg["metadata_csv"], val_p,
                              img_size=cfg["img_size"], cache_dir=cfg["cache_dir"],
                              train=False, augment=False)

        train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"],
                                   shuffle=True, num_workers=cfg["train"]["num_workers"],
                                   pin_memory=True, drop_last=len(train_ds) > cfg["train"]["batch_size"])
        val_loader = DataLoader(val_ds, batch_size=cfg["train"]["batch_size"],
                                 shuffle=False, num_workers=cfg["train"]["num_workers"], pin_memory=True)

        model = GazeViT(backbone=cfg["model"]["backbone"], pretrained=cfg["model"]["pretrained"],
                         out_dim=cfg["model"]["out_dim"], dropout=cfg["model"]["dropout"],
                         freeze_blocks=cfg["model"]["freeze_blocks"]).to(device)

        criterion = nn.CrossEntropyLoss(label_smoothing=cfg["train"]["label_smoothing"])
        # separate (lower) LR for unfrozen backbone params vs. the new head
        backbone_params = [p for n, p in model.backbone.named_parameters() if p.requires_grad]
        head_params = list(model.proj.parameters()) + list(model.classifier.parameters())
        optimizer = torch.optim.AdamW([
            {"params": backbone_params, "lr": cfg["train"]["lr"] * cfg["train"]["backbone_lr_mult"]},
            {"params": head_params, "lr": cfg["train"]["lr"]},
        ], weight_decay=cfg["train"]["weight_decay"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["train"]["epochs"])
        scaler = torch.cuda.amp.GradScaler() if (cfg["train"]["amp"] and device.type == "cuda") else None

        best_f1, patience = -1, 0
        best_path = os.path.join(cfg["train"]["checkpoint_dir"], f"fold{fold}_best.pt")

        for epoch in range(cfg["train"]["epochs"]):
            train_metrics = run_epoch(model, train_loader, optimizer, criterion, device, scaler, train=True)
            val_metrics = run_epoch(model, val_loader, optimizer, criterion, device, scaler, train=False)
            scheduler.step()

            print(f"Epoch {epoch+1}: train_loss={train_metrics['loss']:.4f} "
                  f"val_loss={val_metrics['loss']:.4f} val_f1={val_metrics['f1']:.4f} "
                  f"val_acc={val_metrics['accuracy']:.4f}")

            if val_metrics["f1"] > best_f1:
                best_f1, patience = val_metrics["f1"], 0
                torch.save({"model_state": model.state_dict(), "cfg": cfg}, best_path)
            else:
                patience += 1
                if patience >= cfg["train"]["early_stop_patience"]:
                    print("Early stopping.")
                    break

        model.load_state_dict(torch.load(best_path)["model_state"])
        final_metrics = run_epoch(model, val_loader, optimizer, criterion, device, scaler, train=False)
        print_metrics(final_metrics, prefix=f"Gaze fold {fold}")
        fold_metrics.append(final_metrics)

    keys = ["accuracy", "f1", "roc_auc", "sensitivity", "specificity", "mcc"]
    print("\n===== Gaze Cross-Validation Summary =====")
    for k in keys:
        vals = [m[k] for m in fold_metrics]
        print(f"{k}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/gaze_config.yaml")
    args = parser.parse_args()
    main(args.config)
