#!/usr/bin/env python3
"""
train_yolo.py — Download annotated schematic dataset + Train YOLOv11

Downloads a pre-annotated electronic schematic component detection dataset
from Roboflow Universe, then trains YOLOv11 on it.

Usage:
    # Option 1: Download from Roboflow and train
    python train_yolo.py --roboflow_key <YOUR_API_KEY> --epochs 100 --batch 16

    # Option 2: Use an existing local dataset
    python train_yolo.py --dataset /path/to/dataset.yaml --epochs 100 --batch 16

To get a Roboflow API key:
    1. Go to https://app.roboflow.com/ and sign up (free)
    2. Go to Settings > API Key
    3. Copy your private API key
"""

import argparse
import os
import sys


def parse_arguments():
    parser = argparse.ArgumentParser(description="Train YOLOv11 for Schematic Component Detection")
    parser.add_argument("--roboflow_key", type=str, default=None,
                        help="Roboflow API key to download the dataset")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Path to local dataset.yaml (skip Roboflow download)")
    parser.add_argument("--model", type=str, default="yolo11n.pt",
                        help="Base YOLO model (yolo11n.pt, yolo11s.pt, yolo11m.pt)")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16,
                        help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Input image size")
    parser.add_argument("--device", type=str, default="0",
                        help="CUDA device(s), e.g. '0' or '0,1'")
    parser.add_argument("--project", type=str, default="runs/detect",
                        help="Project directory for saving results")
    parser.add_argument("--name", type=str, default="component_detector",
                        help="Experiment name")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from last checkpoint")
    return parser.parse_args()


def download_roboflow_dataset(api_key, output_dir="roboflow_dataset"):
    """
    Download an electronic schematic component detection dataset from Roboflow.
    Uses the 'YoloV8 electrical schematics' dataset which contains:
    - Resistors, Capacitors, Inductors, Diodes, Voltage sources, etc.
    """
    try:
        from roboflow import Roboflow
    except ImportError:
        print("Installing roboflow...")
        os.system(f"{sys.executable} -m pip install roboflow")
        from roboflow import Roboflow

    rf = Roboflow(api_key=api_key)

    # Try multiple datasets in order of relevance
    datasets_to_try = [
        # (workspace, project, version) — electrical schematic datasets on Roboflow
        ("yolov8-hu2dz", "yolov8-electrical-schematics", 1),
        ("basic-electronics-components", "basic-electronic-component", 1),
        ("component-detection-kd0zu", "circuit-components", 1),
    ]

    for workspace, project_name, version in datasets_to_try:
        try:
            print(f"\nTrying dataset: {workspace}/{project_name} v{version}...")
            project = rf.workspace(workspace).project(project_name)
            dataset = project.version(version).download("yolov11", location=output_dir)
            print(f"  ✓ Downloaded to {output_dir}")

            # Find the dataset.yaml
            yaml_path = os.path.join(output_dir, "data.yaml")
            if not os.path.exists(yaml_path):
                # Some Roboflow exports use dataset.yaml
                for candidate in ["data.yaml", "dataset.yaml"]:
                    candidate_path = os.path.join(output_dir, candidate)
                    if os.path.exists(candidate_path):
                        yaml_path = candidate_path
                        break

            if os.path.exists(yaml_path):
                print(f"  Dataset config: {yaml_path}")
                return yaml_path
            else:
                print(f"  Warning: Could not find data.yaml in {output_dir}")
                # List contents to debug
                for f in os.listdir(output_dir):
                    print(f"    {f}")

        except Exception as e:
            print(f"  ✗ Failed: {e}")
            continue

    print("\nCould not download any dataset from Roboflow.")
    print("Please download manually from https://universe.roboflow.com/")
    print("Search for 'electrical schematics' and export in YOLOv11 format.")
    sys.exit(1)


def main():
    args = parse_arguments()

    # Step 1: Get dataset
    dataset_yaml = args.dataset
    if dataset_yaml is None:
        if args.roboflow_key is None:
            print("=" * 60)
            print("  No dataset provided!")
            print("=" * 60)
            print()
            print("You need either:")
            print("  1. A Roboflow API key to download a dataset:")
            print("     python train_yolo.py --roboflow_key YOUR_KEY")
            print()
            print("  2. A local dataset in YOLO format:")
            print("     python train_yolo.py --dataset /path/to/data.yaml")
            print()
            print("To get a free Roboflow API key:")
            print("  1. Sign up at https://app.roboflow.com/")
            print("  2. Go to Settings > API Key")
            print()
            print("Or download a dataset manually:")
            print("  1. Go to https://universe.roboflow.com/")
            print("  2. Search for 'electrical schematics'")
            print("  3. Export in YOLOv11 format")
            print("  4. Run: python train_yolo.py --dataset path/to/data.yaml")
            return

        dataset_yaml = download_roboflow_dataset(args.roboflow_key)

    # Verify dataset exists
    if not os.path.exists(dataset_yaml):
        print(f"Error: Dataset file '{dataset_yaml}' not found.")
        return

    # Step 2: Train
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Error: ultralytics not installed. Run: pip install ultralytics")
        return

    print()
    print("=" * 60)
    print("  YOLO Component Detector Training")
    print("=" * 60)
    print(f"  Dataset:  {dataset_yaml}")
    print(f"  Model:    {args.model}")
    print(f"  Epochs:   {args.epochs}")
    print(f"  Batch:    {args.batch}")
    print(f"  ImgSize:  {args.imgsz}")
    print(f"  Device:   {args.device}")
    print("=" * 60)

    # Load base model
    print(f"\nLoading base model: {args.model}")
    model = YOLO(args.model)

    # Train
    print("\nStarting training...")
    results = model.train(
        data=dataset_yaml,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        project=args.project,
        name=args.name,
        resume=args.resume,
        # Augmentation settings optimized for schematics
        hsv_h=0.0,       # No hue augmentation (schematics are black/white)
        hsv_s=0.0,       # No saturation augmentation
        hsv_v=0.2,       # Slight brightness variation
        degrees=0.0,     # No rotation
        translate=0.1,   # Slight translation
        scale=0.3,       # Scale variation
        flipud=0.0,      # No vertical flip
        fliplr=0.0,      # No horizontal flip (schematics are directional)
        mosaic=0.5,       # Some mosaic augmentation
        patience=20,     # Early stopping patience
        save=True,
        save_period=10,
        verbose=True,
    )

    # Print results
    print()
    print("=" * 60)
    print("  Training Complete!")
    print("=" * 60)

    best_weights = os.path.join(args.project, args.name, "weights", "best.pt")
    if os.path.exists(best_weights):
        print(f"  Best weights: {best_weights}")
    else:
        for root, dirs, files in os.walk(os.path.join(args.project, args.name)):
            for f in files:
                if f == "best.pt":
                    best_weights = os.path.join(root, f)
                    break

    print(f"\n  Next step — run the SINA pipeline:")
    print(f"  python sina_pipeline.py \\")
    print(f"      --image_dir /path/to/images \\")
    print(f"      --yolo_weights {best_weights} \\")
    print(f"      --output_dir sina_graphs")


if __name__ == "__main__":
    main()
