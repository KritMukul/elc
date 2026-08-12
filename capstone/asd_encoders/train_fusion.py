"""
Cross-modal fusion training + modality ablations.

Real multimodal samples do not exist here: no subject was recorded on all three
modalities, so "triplets" are synthesised by pairing embeddings that merely
share a label. That is a stated limitation, not a bug - but it becomes a bug the
moment the synthesis happens before the train/val split, because the pools are
sampled *with replacement* and the same underlying embedding then lands on both
sides. This module therefore splits **subjects** first and synthesises triplets
independently within each side.

EEG and gait share a subject cohort (same dataset_root, same [PS]* directories,
eeg_*.mat vs mdata_*.mat), so their split is computed jointly - splitting them
separately would put subject P03 train-side in EEG and val-side in gait.
"""

import argparse
import itertools
import os

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset

from models.fusion_transformer import CrossModalFusionTransformer
from utils.metrics import compute_all_metrics, print_metrics
from utils.seed import set_seed


def load_embeddings(path):
    data = np.load(path, allow_pickle=True)
    if "subjects" not in data:
        raise SystemExit(
            f"{path} has no 'subjects' array. Re-run extract_embeddings.py - the fusion "
            "split needs subject identity to stay leak-free."
        )
    return data["embeddings"], data["labels"], data["subjects"]


def subject_folds(labels, subjects, n_folds, seed):
    """Stratified K-fold over *subjects*, returned as (train_set, val_set) pairs.

    Grouping by subject is what keeps one person's windows from straddling the
    fold boundary - the same guarantee train_eeg.py and train_gait.py get from
    splitting subject directories.
    """
    n_subjects = len(set(subjects.tolist()))
    if n_subjects < n_folds:
        raise SystemExit(
            f"Cannot build {n_folds} folds from {n_subjects} subject(s). "
            "Lower train.n_folds in the fusion config."
        )

    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = []
    for train_idx, val_idx in sgkf.split(np.zeros(len(labels)), labels, groups=subjects):
        train_s = set(subjects[train_idx].tolist())
        val_s = set(subjects[val_idx].tolist())
        assert train_s.isdisjoint(val_s), (
            f"StratifiedGroupKFold returned overlapping subjects: {sorted(train_s & val_s)}"
        )
        folds.append((train_s, val_s))
    return folds


def build_pool(embeddings, labels, subjects, allowed_subjects):
    """Class-keyed embedding pool restricted to `allowed_subjects`."""
    pool = {0: [], 1: []}
    for e, l, s in zip(embeddings, labels, subjects):
        if s in allowed_subjects:
            pool[int(l)].append(e)
    return pool


def make_synthetic_triplets(eeg_pool, gaze_pool, gait_pool, n_per_class, seed, side):
    """Label-matched random pairing, drawn only from the pools handed in.

    Sampling with replacement is fine *within* one side of the split; the defect
    this replaces was drawing both sides from one shared pool.
    """
    pools = {"eeg": eeg_pool, "gaze": gaze_pool, "gait": gait_pool}
    for name, pool in pools.items():
        for label in (0, 1):
            if not pool[label]:
                raise SystemExit(
                    f"{side} split has no {name} embeddings for class {label}. "
                    "The subject split left a class empty - lower train.n_folds or "
                    "check the label distribution in the .npz files."
                )

    rng = np.random.default_rng(seed)
    eeg_out, gaze_out, gait_out, y_out = [], [], [], []
    for label in (0, 1):
        for _ in range(n_per_class):
            eeg_out.append(eeg_pool[label][rng.integers(len(eeg_pool[label]))])
            gaze_out.append(gaze_pool[label][rng.integers(len(gaze_pool[label]))])
            gait_out.append(gait_pool[label][rng.integers(len(gait_pool[label]))])
            y_out.append(label)
    idx = rng.permutation(len(y_out))
    return (np.array(eeg_out)[idx], np.array(gaze_out)[idx],
            np.array(gait_out)[idx], np.array(y_out)[idx])


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


def train_one_config(train_arrays, val_arrays, active_modalities, cfg, device, tag):
    """Train on pre-split arrays. The split happens upstream, by subject."""
    train_ds = TripletDataset(*train_arrays, active_modalities=active_modalities)
    val_ds = TripletDataset(*val_arrays, active_modalities=active_modalities)

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
        run_epoch(model, train_loader, optimizer, criterion, device, train=True)
        val_metrics = run_epoch(model, val_loader, optimizer, criterion, device, train=False)
        if val_metrics["f1"] > best_f1:
            best_f1, patience = val_metrics["f1"], 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= cfg["train"]["early_stop_patience"]:
                break

    model.load_state_dict(best_state)
    final_metrics = run_epoch(model, val_loader, optimizer, criterion, device, train=False)
    print_metrics(final_metrics, prefix=tag)

    os.makedirs(cfg["train"]["checkpoint_dir"], exist_ok=True)
    torch.save({"model_state": best_state, "cfg": cfg, "modalities": list(active_modalities)},
               os.path.join(cfg["train"]["checkpoint_dir"], f"{tag}.pt"))
    return final_metrics


def main(cfg_path):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    strategy = cfg.get("pairing_strategy", "label_matched_random")
    if strategy != "label_matched_random":
        raise SystemExit(
            f"pairing_strategy={strategy!r} is not implemented; "
            "only 'label_matched_random' exists."
        )

    seed = cfg["train"]["seed"]
    n_folds = cfg["train"]["n_folds"]
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    eeg_emb, eeg_lab, eeg_subj = load_embeddings(cfg["embeddings"]["eeg"])
    gaze_emb, gaze_lab, gaze_subj = load_embeddings(cfg["embeddings"]["gaze"])
    gait_emb, gait_lab, gait_subj = load_embeddings(cfg["embeddings"]["gait"])

    # EEG + gait are the same people -> one joint split so a subject cannot be
    # train-side in one and val-side in the other. Gaze is a separate cohort.
    shared_lab = np.concatenate([eeg_lab, gait_lab])
    shared_subj = np.concatenate([eeg_subj, gait_subj])
    shared_folds = subject_folds(shared_lab, shared_subj, n_folds, seed)
    gaze_folds = subject_folds(gaze_lab, gaze_subj, n_folds, seed)

    n_train_per_class = cfg["n_synthetic_samples_per_class"]
    n_val_per_class = max(1, round(n_train_per_class * 0.25))   # keeps the old 80/20 ratio

    modality_names = {"eeg": "EEG", "gaze": "Gaze", "gait": "Gait"}
    combos = []
    for r in range(1, 4):
        combos.extend(itertools.combinations(["eeg", "gaze", "gait"], r))

    results = {tag: [] for tag in ("+".join(modality_names[m] for m in c) for c in combos)}

    for fold in range(n_folds):
        shared_train, shared_val = shared_folds[fold]
        gaze_train, gaze_val = gaze_folds[fold]
        print(f"\n########## Fusion fold {fold + 1}/{n_folds} "
              f"(shared {len(shared_train)}/{len(shared_val)} subjects, "
              f"gaze {len(gaze_train)}/{len(gaze_val)}) ##########")

        sides = {}
        for side, shared_s, gaze_s, n_per_class, rng_seed in (
            ("train", shared_train, gaze_train, n_train_per_class, seed + 1000 * fold),
            ("val", shared_val, gaze_val, n_val_per_class, seed + 1000 * fold + 500),
        ):
            sides[side] = make_synthetic_triplets(
                build_pool(eeg_emb, eeg_lab, eeg_subj, shared_s),
                build_pool(gaze_emb, gaze_lab, gaze_subj, gaze_s),
                build_pool(gait_emb, gait_lab, gait_subj, shared_s),
                n_per_class=n_per_class, seed=rng_seed, side=side)

        for combo in combos:
            tag = "+".join(modality_names[m] for m in combo)
            print(f"\n===== Fusion ablation: {tag} (fold {fold}) =====")
            metrics = train_one_config(sides["train"], sides["val"], combo, cfg, device,
                                       f"{tag}_fold{fold}")
            results[tag].append(metrics)

    keys = ["accuracy", "f1", "roc_auc", "mcc"]
    print(f"\n===== Ablation Summary over {n_folds} subject-disjoint folds =====")
    for tag, fold_metrics in results.items():
        summary = "  ".join(
            f"{k}={np.mean([m[k] for m in fold_metrics]):.4f}+/-"
            f"{np.std([m[k] for m in fold_metrics]):.4f}"
            for k in keys
        )
        print(f"{tag:20s} {summary}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/fusion_config.yaml")
    args = parser.parse_args()
    main(args.config)
