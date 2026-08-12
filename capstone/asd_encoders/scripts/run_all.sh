#!/usr/bin/env bash
set -euo pipefail

# Always run from the package root, whatever the caller's cwd is - every path
# below (configs/, checkpoints/, embeddings/) is relative to it.
cd "$(dirname "$0")/.."

echo "Train EEG encoder"
python train_eeg.py --config configs/eeg_config.yaml

echo "Train Gaze encoder"
python train_gaze.py --config configs/gaze_config.yaml

echo "Train Gait encoder"
python train_gait.py --config configs/gait_config.yaml

# Out-of-fold: each fold's checkpoint emits only the subjects it never trained
# on, so the fusion stage never sees an in-sample encoder output.
echo "Extract frozen embeddings (out-of-fold)"
python extract_embeddings.py --modality eeg  --config configs/eeg_config.yaml  \
    --checkpoint checkpoints/eeg/fold*_best.pt  --out-of-fold \
    --out embeddings/eeg_embeddings.npz
python extract_embeddings.py --modality gaze --config configs/gaze_config.yaml \
    --checkpoint checkpoints/gaze/fold*_best.pt --out-of-fold \
    --out embeddings/gaze_embeddings.npz
python extract_embeddings.py --modality gait --config configs/gait_config.yaml \
    --checkpoint checkpoints/gait/fold*_best.pt --out-of-fold \
    --out embeddings/gait_embeddings.npz

echo "Train fusion transformer + ablations"
python train_fusion.py --config configs/fusion_config.yaml
