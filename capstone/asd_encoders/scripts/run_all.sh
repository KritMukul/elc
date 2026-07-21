
set -e

echo "Train EEG encoder"
python train_eeg.py --config configs/eeg_config.yaml

echo "Train Gaze encoder"
python train_gaze.py --config configs/gaze_config.yaml

echo "Train Gait encoder"
python train_gait.py --config configs/gait_config.yaml

echo "Extract frozen embeddings"
python extract_embeddings.py --modality eeg  --config configs/eeg_config.yaml  \
    --checkpoint checkpoints/eeg/fold0_best.pt  --out embeddings/eeg_embeddings.npz
python extract_embeddings.py --modality gaze --config configs/gaze_config.yaml \
    --checkpoint checkpoints/gaze/fold0_best.pt --out embeddings/gaze_embeddings.npz
python extract_embeddings.py --modality gait --config configs/gait_config.yaml \
    --checkpoint checkpoints/gait/fold0_best.pt --out embeddings/gait_embeddings.npz

echo "Train fusion transformer + ablations"
python train_fusion.py --config configs/fusion_config.yaml

