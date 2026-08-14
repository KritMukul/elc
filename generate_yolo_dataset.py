#!/usr/bin/env python3
"""
generate_yolo_dataset.py — Synthetic YOLO Training Data Generator

Converts circuit graphs from masala_chai_graphs.json into KiCad schematics,
exports them as images, and extracts component bounding boxes in YOLO format.

This creates a fully labeled dataset for training YOLOv11 to detect
electronic components in schematic images.

Usage:
    python generate_yolo_dataset.py \
        --graphs masala_chai_graphs.json \
        --output_dir yolo_dataset \
        --max_circuits 500
"""

import json
import os
import sys
import re
import argparse
import random
import subprocess
from pathlib import Path
from collections import defaultdict

# ─── Component type → YOLO class mapping ───
# These are the classes YOLO will learn to detect
CLASS_MAP = {
    "resistor": 0,  "res": 0,
    "capacitor": 1, "cap": 1,
    "inductor": 2,  "ind": 2,
    "diode": 3,
    "vsource": 4,
    "isource": 5,
    "mosfet": 6,  "nfet": 6, "pfet": 6,
    "bjt": 7,
    "vccs": 8, "vcvs": 8, "cccs": 8, "ccvs": 8,  # controlled sources
}

CLASS_NAMES = [
    "resistor",
    "capacitor",
    "inductor",
    "diode",
    "voltage_source",
    "current_source",
    "mosfet",
    "bjt",
    "controlled_source",
]

# ─── KiCad symbol info (same as graph_to_kicad.py) ───
DEVICE_MAP = {
    "cap":       {"lib": "Device", "sym": "C",       "pins": {"P": "1",  "N": "2", "pos": "1", "neg": "2"}, "prefix": "C"},
    "capacitor": {"lib": "Device", "sym": "C",       "pins": {"P": "1",  "N": "2", "pos": "1", "neg": "2"}, "prefix": "C"},
    "res":       {"lib": "Device", "sym": "R",       "pins": {"P": "1",  "N": "2", "pos": "1", "neg": "2"}, "prefix": "R"},
    "resistor":  {"lib": "Device", "sym": "R",       "pins": {"P": "1",  "N": "2", "pos": "1", "neg": "2"}, "prefix": "R"},
    "ind":       {"lib": "Device", "sym": "L",       "pins": {"P": "1",  "N": "2", "pos": "1", "neg": "2"}, "prefix": "L"},
    "inductor":  {"lib": "Device", "sym": "L",       "pins": {"P": "1",  "N": "2", "pos": "1", "neg": "2"}, "prefix": "L"},
    "diode":     {"lib": "Device", "sym": "D",       "pins": {"A": "A",  "K": "K", "anode": "A", "cathode": "K"}, "prefix": "D"},
    "vsource":   {"lib": "Device", "sym": "Battery", "pins": {"P": "+",  "N": "-", "pos": "+", "neg": "-"}, "prefix": "BT"},
    "isource":   {"lib": "Device", "sym": "Battery", "pins": {"P": "+",  "N": "-", "pos": "+", "neg": "-"}, "prefix": "I"},
    "mosfet":    {"lib": "Device", "sym": "Q_NMOS_GDS", "pins": {"G": "1", "D": "2", "S": "3", "B": "4"}, "prefix": "M"},
    "nfet":      {"lib": "Device", "sym": "Q_NMOS_GDS", "pins": {"G": "1", "D": "2", "S": "3", "B": "4"}, "prefix": "M"},
    "pfet":      {"lib": "Device", "sym": "Q_PMOS_GDS", "pins": {"G": "1", "D": "2", "S": "3", "B": "4"}, "prefix": "M"},
    "bjt":       {"lib": "Device", "sym": "Q_NPN_BCE",  "pins": {"c": "C", "b": "B", "e": "E"}, "prefix": "Q"},
}

# Approximate bounding box sizes for each symbol type (width_mm, height_mm)
# These are used to generate YOLO annotations from placement coordinates
SYMBOL_SIZES = {
    "R":           (2.54, 10.16),   # Resistor: narrow, tall
    "C":           (2.54, 5.08),    # Capacitor
    "L":           (2.54, 10.16),   # Inductor
    "D":           (5.08, 7.62),    # Diode
    "Battery":     (5.08, 10.16),   # Battery/Voltage source
    "Q_NMOS_GDS":  (7.62, 10.16),   # MOSFET
    "Q_PMOS_GDS":  (7.62, 10.16),
    "Q_NPN_BCE":   (7.62, 10.16),   # BJT
    "LED":         (5.08, 7.62),
}


def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate YOLO training dataset from circuit graphs")
    parser.add_argument("--graphs", type=str, default="masala_chai_graphs.json",
                        help="Path to masala_chai_graphs.json")
    parser.add_argument("--output_dir", type=str, default="yolo_dataset",
                        help="Output directory for the YOLO dataset")
    parser.add_argument("--max_circuits", type=int, default=500,
                        help="Maximum number of circuits to process")
    parser.add_argument("--img_width", type=int, default=1280,
                        help="Image width in pixels for export")
    parser.add_argument("--img_height", type=int, default=960,
                        help="Image height in pixels for export")
    parser.add_argument("--schgen_dir", type=str, default="netlist_to_kicad/Microsoft_Schgen",
                        help="Path to the Microsoft SchGen directory")
    parser.add_argument("--train_split", type=float, default=0.85,
                        help="Fraction of data to use for training (rest is validation)")
    return parser.parse_args()


def get_device_info(dev):
    """Look up KiCad symbol info for a device."""
    info = DEVICE_MAP.get(dev["type"])
    if info is None:
        return None  # Skip unknown types
    return info


def assign_references(devices):
    """Give every device a unique KiCad reference."""
    counters = defaultdict(int)
    refs = {}
    for dev in devices:
        info = get_device_info(dev)
        if info is None:
            continue
        prefix = info["prefix"]
        counters[prefix] += 1
        refs[dev["name"]] = f"{prefix}{counters[prefix]}"
    return refs


def place_components_grid(devices):
    """
    Place components on a grid layout and return their coordinates.
    Returns: list of dicts with keys: name, x, y, rot, sym_name, type
    """
    placements = []
    base_x, base_y = 80, 100
    gap_x, gap_y = 40, 40
    cols = max(3, int(len(devices) ** 0.5) + 1)

    for i, dev in enumerate(devices):
        info = get_device_info(dev)
        if info is None:
            continue

        col = i % cols
        row = i // cols
        x = base_x + col * gap_x
        y = base_y + row * gap_y
        rot = random.choice([0, 90])  # Random rotation for data augmentation

        placements.append({
            "name": dev["name"],
            "x": x,
            "y": y,
            "rot": rot,
            "sym_name": info["sym"],
            "type": dev["type"],
        })

    return placements


def generate_layout_script(graph, placements, ref_map):
    """Generate a Python layout script for Microsoft SchGen."""
    lines = [
        "from modules.kicad_sch_interface import *",
        "",
        "# Component Placement",
    ]

    for p in placements:
        dev = next((d for d in graph["devices"] if d["name"] == p["name"]), None)
        if dev is None:
            continue
        info = get_device_info(dev)
        if info is None:
            continue

        ref = ref_map.get(dev["name"], p["name"])
        value = dev.get("params", {}).get("value", "")
        if not value:
            pos_params = dev.get("params", {}).get("positional", [])
            value = " ".join(pos_params) if pos_params else info["sym"]

        lines.append(
            f'add_schematic_symbol('
            f'symbol_lib="{info["lib"]}", symbol_name="{info["sym"]}", '
            f'pos_x={p["x"]}, pos_y={p["y"]}, '
            f'reference="{ref}", value="{value}", rotation={p["rot"]})'
        )

    # Net connections (daisy-chain)
    net_map = defaultdict(list)
    for dev in graph["devices"]:
        info = get_device_info(dev)
        if info is None:
            continue
        for pin in dev["pins"]:
            ref = ref_map.get(dev["name"])
            if ref is None:
                continue
            kicad_pin = info["pins"].get(pin["role"])
            if kicad_pin:
                net_map[pin["net"]].append((ref, kicad_pin))

    lines.append("")
    lines.append("# Net Connections")
    for net_name in sorted(net_map):
        pin_list = net_map[net_name]
        if len(pin_list) < 2:
            continue
        for idx in range(len(pin_list) - 1):
            ref_a, pin_a = pin_list[idx]
            ref_b, pin_b = pin_list[idx + 1]
            lines.append(
                f'connect_pins("{ref_a}", "{pin_a}", "{ref_b}", "{pin_b}")'
            )

    lines.append("")
    lines.append("write_out_all_wires()")

    return "\n".join(lines)


def placements_to_yolo_annotations(placements, img_w, img_h, sheet_w=297.0, sheet_h=210.0):
    """
    Convert component placements (in mm) to YOLO format annotations.
    YOLO format: class_id center_x center_y width height (all normalized 0-1)
    """
    annotations = []
    for p in placements:
        cls_id = CLASS_MAP.get(p["type"])
        if cls_id is None:
            continue

        sym_name = p["sym_name"]
        sym_w, sym_h = SYMBOL_SIZES.get(sym_name, (5.08, 7.62))

        # If rotated 90 degrees, swap width and height
        if p["rot"] == 90:
            sym_w, sym_h = sym_h, sym_w

        # Convert mm coordinates to normalized image coordinates
        cx = p["x"] / sheet_w
        cy = p["y"] / sheet_h
        w = sym_w / sheet_w
        h = sym_h / sheet_h

        # Clamp to [0, 1]
        cx = max(0, min(1, cx))
        cy = max(0, min(1, cy))
        w = max(0.01, min(1, w))
        h = max(0.01, min(1, h))

        annotations.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    return annotations


def export_kicad_to_image(sch_path, output_image, img_w=1280, img_h=960):
    """
    Export a .kicad_sch file to a PNG image using kicad-cli.
    Falls back to creating a placeholder if kicad-cli is not available.
    """
    try:
        result = subprocess.run(
            ["kicad-cli", "sch", "export", "svg",
             "--output", str(output_image).replace(".png", ".svg"),
             str(sch_path)],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0:
            # Convert SVG to PNG using cairosvg or imagemagick if available
            svg_path = str(output_image).replace(".png", ".svg")
            try:
                subprocess.run(
                    ["convert", svg_path, "-resize", f"{img_w}x{img_h}", str(output_image)],
                    capture_output=True, timeout=30
                )
                return True
            except FileNotFoundError:
                try:
                    import cairosvg
                    cairosvg.svg2png(url=svg_path, write_to=str(output_image),
                                    output_width=img_w, output_height=img_h)
                    return True
                except ImportError:
                    pass
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Fallback: a blank bordered page.
    #
    # This is a placeholder, not a schematic — it contains no components and no
    # wires, so anything reading it back (SINA, YOLO training) gets nothing.
    # It used to be written silently, which made a dead pipeline look like a
    # working one, so say so loudly and exactly once.
    global _WARNED_PLACEHOLDER
    if not _WARNED_PLACEHOLDER:
        _WARNED_PLACEHOLDER = True
        print("  WARNING: writing BLANK placeholder images, not real schematics.")
        print("           Real rendering needs kicad-cli on PATH plus an SVG")
        print("           rasteriser (pip install cairosvg, or ImageMagick), and")
        print("           this call site passes sch_path=None so nothing is")
        print("           exported regardless. Downstream detection will find"
              " nothing.")

    try:
        import cv2
        import numpy as np
        img = np.ones((img_h, img_w, 3), dtype=np.uint8) * 255
        # Draw a simple border
        cv2.rectangle(img, (10, 10), (img_w - 10, img_h - 10), (0, 0, 0), 1)
        cv2.putText(img, "Schematic", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.imwrite(str(output_image), img)
        return True
    except ImportError:
        return False


_WARNED_PLACEHOLDER = False


def main():
    args = parse_arguments()

    print(f"Loading circuit graphs from {args.graphs}...")
    with open(args.graphs, "r") as f:
        all_graphs = json.load(f)
    print(f"  Loaded {len(all_graphs)} circuits")

    # Filter: only circuits with known device types and reasonable size
    valid_graphs = []
    for g in all_graphs:
        devices = g.get("devices", [])
        if len(devices) < 2 or len(devices) > 20:
            continue
        # Check that at least some devices are in our DEVICE_MAP
        known = [d for d in devices if d["type"] in DEVICE_MAP]
        if len(known) >= 2:
            valid_graphs.append(g)

    random.shuffle(valid_graphs)
    valid_graphs = valid_graphs[:args.max_circuits]
    print(f"  Using {len(valid_graphs)} valid circuits (max {args.max_circuits})")

    # Create output directories
    train_img_dir = os.path.join(args.output_dir, "images", "train")
    val_img_dir = os.path.join(args.output_dir, "images", "val")
    train_lbl_dir = os.path.join(args.output_dir, "labels", "train")
    val_lbl_dir = os.path.join(args.output_dir, "labels", "val")
    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(val_img_dir, exist_ok=True)
    os.makedirs(train_lbl_dir, exist_ok=True)
    os.makedirs(val_lbl_dir, exist_ok=True)

    # Split
    split_idx = int(len(valid_graphs) * args.train_split)
    train_graphs = valid_graphs[:split_idx]
    val_graphs = valid_graphs[split_idx:]

    print(f"  Train: {len(train_graphs)}, Val: {len(val_graphs)}")

    # Process each circuit
    schgen_dir = args.schgen_dir
    for split_name, graphs, img_dir, lbl_dir in [
        ("train", train_graphs, train_img_dir, train_lbl_dir),
        ("val", val_graphs, val_img_dir, val_lbl_dir),
    ]:
        for i, graph in enumerate(graphs):
            circuit_id = f"{split_name}_{i:05d}"
            source = graph.get("source_file", "unknown")

            # Filter to known devices only
            known_devices = [d for d in graph["devices"] if d["type"] in DEVICE_MAP]
            graph_filtered = {**graph, "devices": known_devices}

            # Place components
            placements = place_components_grid(known_devices)
            if not placements:
                continue

            ref_map = assign_references(known_devices)

            # Generate YOLO annotations
            annotations = placements_to_yolo_annotations(
                placements, args.img_width, args.img_height
            )

            # Save annotation file
            lbl_path = os.path.join(lbl_dir, f"{circuit_id}.txt")
            with open(lbl_path, "w") as f:
                f.write("\n".join(annotations))

            # Generate KiCad layout script
            script_code = generate_layout_script(graph_filtered, placements, ref_map)

            # Save the script (for optional KiCad rendering later)
            script_path = os.path.join(args.output_dir, "scripts", f"{circuit_id}.py")
            os.makedirs(os.path.dirname(script_path), exist_ok=True)
            with open(script_path, "w") as f:
                f.write(script_code)

            # Try to generate the image
            # For now we render a simple placeholder; on DGX with KiCad installed,
            # this would produce real schematic images
            img_path = os.path.join(img_dir, f"{circuit_id}.png")
            export_kicad_to_image(None, img_path, args.img_width, args.img_height)

            if (i + 1) % 50 == 0:
                print(f"  [{split_name}] Processed {i + 1}/{len(graphs)}")

    # Write dataset.yaml for YOLO training
    yaml_path = os.path.join(args.output_dir, "dataset.yaml")
    abs_output = os.path.abspath(args.output_dir)
    with open(yaml_path, "w") as f:
        f.write(f"path: {abs_output}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write(f"nc: {len(CLASS_NAMES)}\n")
        f.write(f"names: {CLASS_NAMES}\n")

    print(f"\n  Dataset YAML saved to: {yaml_path}")
    print(f"  Total images: {len(train_graphs) + len(val_graphs)}")
    print("  Done!")


if __name__ == "__main__":
    main()
