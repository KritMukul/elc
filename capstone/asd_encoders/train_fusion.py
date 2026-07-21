
import argparse
import itertools
import os

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from models.fusion_transformer import CrossModalFusionTransformer
from utils.metrics import compute_all_metrics, print_metrics
from utils.seed import set_seed


def load_embeddings(path):
    data = np.load(path, allow_pickle=True)
    return data["embeddings"], data["labels"]


def make_synthetic_triplets(eeg_pool, gaze_pool, gait_pool, n_per_class, seed):

    rng = np.random.default_rng(seed)
    eeg_out, gaze_out, gait_out, y_out = [], [], [], []
    for label in (0, 1):
        for _ in range(n_per_class):
            eeg_out.append(eeg_pool[label][rng.integers(len(eeg_pool[label]))])
            gaze_out.append(gaze_pool[label][rng.integers(len(gaze_pool[label]))])
            gait_out.append(gait_pool[label][rng.integers(len(gait_pool[label]))])
            y_out.append(label)
    idx = rng.permutation(len(y_out))
    eeg_out, gaze_out, gait_out, y_out = (np.array(eeg_out)[idx], np.array(gaze_out)[idx],
                                           np.array(gait_out)[idx], np.array(y_out)[idx])
    return eeg_out, gaze_out, gait_out, y_out


class TripletDataset(Dataset):
    def __init__(self, eeg, gaze, gait, labels, active_modalities=("eeg", "gaze", "gait")):
        self.eeg, self.gaze, self.gait, self.labels = eeg, gaze, gait, labels
        self.active = set(active_modalities)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {
            "eeg": torch.from_numpy(self.eeg[idx]).float() if "eeg" in self.active else None,
            "gaze": torch.from_numpy(self.gaze[idx]).float() if "gaze" in self.active else None,
            "gait": torch.from_numpy(self.gait[idx]).float() if "gait" in self.active else None,
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }
        return item


def collate(batch, device):
    def stack_or_none(key):
        if batch[0][key] is None:
            return None
        return torch.stack([b[key] for b in batch]).to(device)

    eeg = stack_or_none("eeg")
    gaze = stack_or_none("gaze")
    gait = stack_or_none("gait")
    labels = torch.stack([b["label"] for b in batch]).to(device)
    return eeg, gaze, gait, labels


def run_epoch(model, loader, optimizer, criterion, device, train=True):
    model.train() if train else model.eval()
    total_loss, all_true, all_pred, all_prob = 0.0, [], [], []
    with torch.set_grad_enabled(train):
        for batch in loader:
            eeg, gaze, gait, y = batch
            if train:
                optimizer.zero_grad(set_to_none=True)
            logits, _ = model(eeg_emb=eeg, gaze_emb=gaze, gait_emb=gait)
            loss = criterion(logits, y)
            if train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * y.size(0)
            probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
            preds = logits.argmax(dim=1).detach().cpu().numpy()
            all_true.extend(y.cpu().numpy().tolist())
            all_pred.extend(preds.tolist())
            all_prob.extend(probs.tolist())
    metrics = compute_all_metrics(all_true, all_pred, all_prob)
    metrics["loss"] = total_loss / max(1, len(loader.dataset))
    return metrics


def build_pool(embeddings, labels):
    pool = {0: [], 1: []}
    for e, l in zip(embeddings, labels):
        pool[int(l)].append(e)
    return pool


def train_one_config(eeg, gaze, gait, y, active_modalities, cfg, device, tag):
    train_idx, val_idx = train_test_split(np.arange(len(y)), test_size=0.2,
                                           stratify=y, random_state=cfg["train"]["seed"])
    train_ds = TripletDataset(eeg[train_idx], gaze[train_idx], gait[train_idx], y[train_idx],
                               active_modalities=active_modalities)
    val_ds = TripletDataset(eeg[val_idx], gaze[val_idx], gait[val_idx], y[val_idx],
                             active_modalities=active_modalities)

    def make_loader(ds, shuffle):
        return DataLoader(ds, batch_size=cfg["train"]["batch_size"], shuffle=shuffle,
                           collate_fn=lambda b: collate(b, device))

    train_loader = make_loader(train_ds, True)
    val_loader = make_loader(val_ds, False)

    model = CrossModalFusionTransformer(emb_dim=cfg["model"]["emb_dim"],
                                         n_heads=cfg["model"]["n_heads"],
                                         n_layers=cfg["model"]["n_layers"],
                                         mlp_ratio=cfg["model"]["mlp_ratio"],
                                         dropout=cfg["model"]["dropout"]).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg["train"]["label_smoothing"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"],
                                   weight_decay=cfg["train"]["weight_decay"])

    best_f1, patience, best_state = -1, 0, None
    for epoch in range(cfg["train"]["epochs"]):
        train_metrics = run_epoch(model, train_loader, optimizer, criterion, device, train=True)
        val_metrics = run_epoch(model, val_loader, optimizer, criterion, device, train=False)
        if val_metrics["f1"] > best_f1:
            best_f1, patience, best_state = val_metrics["f1"], 0, {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= cfg["train"]["early_stop_patience"]:
                break

    model.load_state_dict(best_state)
    final_metrics = run_epoch(model, val_loader, optimizer, criterion, device, train=False)
    print_metrics(final_metrics, prefix=tag)

    os.makedirs(cfg["train"]["checkpoint_dir"], exist_ok=True)
    torch.save({"model_state": best_state, "cfg": cfg, "modalities": active_modalities},
               os.path.join(cfg["train"]["checkpoint_dir"], f"{tag}.pt"))
    return final_metrics


def main(cfg_path):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["train"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    eeg_emb, eeg_lab = load_embeddings(cfg["embeddings"]["eeg"])
    gaze_emb, gaze_lab = load_embeddings(cfg["embeddings"]["gaze"])
    gait_emb, gait_lab = load_embeddings(cfg["embeddings"]["gait"])

    eeg_pool = build_pool(eeg_emb, eeg_lab)
    gaze_pool = build_pool(gaze_emb, gaze_lab)
    gait_pool = build_pool(gait_emb, gait_lab)

    eeg, gaze, gait, y = make_synthetic_triplets(
        eeg_pool, gaze_pool, gait_pool,
        n_per_class=cfg["n_synthetic_samples_per_class"], seed=cfg["train"]["seed"])

    modality_names = {"eeg": "EEG", "gaze": "Gaze", "gait": "Gait"}
    combos = []
    for r in range(1, 4):
        combos.extend(itertools.combinations(["eeg", "gaze", "gait"], r))

    results = {}
    for combo in combos:
        tag = "+".join(modality_names[m] for m in combo)
        print(f"\n===== Fusion ablation: {tag} =====")
        metrics = train_one_config(eeg, gaze, gait, y, combo, cfg, device, tag)
        results[tag] = metrics

    print("\n===== Ablation Summary (EEG / Gaze / Gait contribution) =====")
    for tag, m in results.items():
        print(f"{tag:20s} acc={m['accuracy']:.4f} f1={m['f1']:.4f} "
              f"roc_auc={m['roc_auc']:.4f} mcc={m['mcc']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/fusion_config.yaml")
    args = parser.parse_args()
    main(args.config)
