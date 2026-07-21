#!/usr/bin/env python3
"""
graph_to_kicad.py — Deterministic Graph JSON → KiCad Schematic.

Converts a circuit graph JSON directly into a KiCad schematic by generating
the Python layout script that the Microsoft SchGen interface expects.

Two layout modes:
  1. Grid layout (default) — simple horizontal placement
  2. GVAE layout (--use_gvae) — uses your trained Graph-VAE to predict
     component positions that resemble real schematics

Usage:
    # Grid layout (fast, always works)
    python graph_to_kicad.py --graph input_graph.json --project_name my_circuit

    # GVAE layout (needs torch + torch_geometric + weights)
    python graph_to_kicad.py --graph input_graph.json --project_name my_circuit \\
        --use_gvae --gvae_weights ../gvae_weights.pth
"""

import json
import argparse
import subprocess
import sys
import os
import math
from collections import defaultdict


# ═══════════════════════════════════════════════════════════════════
#  Device type → KiCad symbol mapping
# ═══════════════════════════════════════════════════════════════════

DEVICE_MAP = {
    # graph type   KiCad lib   symbol         graph-role → kicad-pin    ref-prefix
    "cap":       {"lib": "Device", "sym": "C",       "pins": {"P": "1",  "N": "2"},               "prefix": "C"},
    "capacitor": {"lib": "Device", "sym": "C",       "pins": {"P": "1",  "N": "2", "pos": "1", "neg": "2"}, "prefix": "C"},
    "res":       {"lib": "Device", "sym": "R",       "pins": {"P": "1",  "N": "2"},               "prefix": "R"},
    "resistor":  {"lib": "Device", "sym": "R",       "pins": {"P": "1",  "N": "2", "pos": "1", "neg": "2"}, "prefix": "R"},
    "ind":       {"lib": "Device", "sym": "L",       "pins": {"P": "1",  "N": "2"},               "prefix": "L"},
    "inductor":  {"lib": "Device", "sym": "L",       "pins": {"P": "1",  "N": "2", "pos": "1", "neg": "2"}, "prefix": "L"},
    "diode":     {"lib": "Device", "sym": "D",       "pins": {"A": "A",  "K": "K", "anode": "A", "cathode": "K"}, "prefix": "D"},
    "vsource":   {"lib": "Device", "sym": "Battery", "pins": {"P": "+",  "N": "-", "pos": "+", "neg": "-"}, "prefix": "BT"},
    "isource":   {"lib": "Device", "sym": "Battery", "pins": {"P": "+",  "N": "-", "pos": "+", "neg": "-"}, "prefix": "I"},
    "mosfet":    {"lib": "Device", "sym": "Q_NMOS_GDS", "pins": {"G": "1", "D": "2", "S": "3", "B": "4",
                  "g": "1", "d": "2", "s": "3", "b": "4", "gate": "1", "drain": "2", "source": "3", "bulk": "4"}, "prefix": "M"},
    "nfet":      {"lib": "Device", "sym": "Q_NMOS_GDS", "pins": {"G": "1", "D": "2", "S": "3", "B": "4"}, "prefix": "M"},
    "pfet":      {"lib": "Device", "sym": "Q_PMOS_GDS", "pins": {"G": "1", "D": "2", "S": "3", "B": "4"}, "prefix": "M"},
    "bjt":       {"lib": "Device", "sym": "Q_NPN_BCE", "pins": {"c": "C", "b": "B", "e": "E",
                  "collector": "C", "base": "B", "emitter": "E"}, "prefix": "Q"},
}

# Model-level overrides (checked before DEVICE_MAP)
MODEL_OVERRIDES = {
    "LED": {"lib": "Device", "sym": "LED", "pins": {"A": "A", "K": "K", "anode": "A", "cathode": "K"}, "prefix": "D"},
}

# Feature vector indices — MUST match training code in pipeline.py
COMP_TYPE_TO_FEAT = {
    "res": 0, "resistor": 0,
    "cap": 1, "capacitor": 1,
    "ind": 2, "inductor": 2,
    "mosfet": 3, "nfet": 3, "pfet": 3, "bjt": 3,
    "diode": 5,
    "isource": 4, "vsource": 4,  # mapped to 'U' category like training
    "_GND": 7,
    "_VCC": 8,
}
NUM_FEATURES = 9  # matches pipeline.py training (R,C,L,Q,U,D,J,GND,VCC)


# ═══════════════════════════════════════════════════════════════════
#  GVAE Model Definition (must match pipeline.py training arch)
# ═══════════════════════════════════════════════════════════════════

def _define_gvae():
    """Import torch lazily and define the model class."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import GATConv

    class SchematicGVAE(nn.Module):
        def __init__(self, in_channels, hidden_channels, latent_dim):
            super().__init__()
            self.conv1 = GATConv(in_channels, hidden_channels, heads=4, concat=False)
            self.conv_mu = GATConv(hidden_channels, latent_dim, heads=1, concat=False)
            self.conv_logvar = GATConv(hidden_channels, latent_dim, heads=1, concat=False)
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, 2),
            )

        def reparameterize(self, mu, logvar):
            return mu  # deterministic at inference

        def forward(self, x, edge_index):
            h = F.relu(self.conv1(x, edge_index))
            mu = self.conv_mu(h, edge_index)
            logvar = self.conv_logvar(h, edge_index)
            pred_pos = self.decoder(self.reparameterize(mu, logvar))
            return pred_pos, mu, logvar

    return SchematicGVAE


# ═══════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════

def load_graph(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def get_device_info(dev: dict) -> dict:
    """Look up KiCad symbol info — model overrides take priority."""
    model = dev.get("model", "").strip().upper()
    if model in MODEL_OVERRIDES:
        return MODEL_OVERRIDES[model]
    info = DEVICE_MAP.get(dev["type"])
    if info is None:
        print(f"  Warning: unknown device type '{dev['type']}', falling back to resistor.")
        return DEVICE_MAP["res"]
    return info


def detect_power_nets(devices: list):
    """Identify VDD / GND nets from voltage-source terminals."""
    vdd_nets, gnd_nets = set(), set()
    for dev in devices:
        if dev["type"] in ("vsource", "isource"):
            for pin in dev["pins"]:
                if pin["role"] in ("P", "pos", "+"):
                    vdd_nets.add(pin["net"])
                elif pin["role"] in ("N", "neg", "-"):
                    gnd_nets.add(pin["net"])
    return vdd_nets, gnd_nets


def assign_references(devices: list) -> dict:
    """Give every device a unique KiCad reference (R1, C1, BT1, …)."""
    counters = defaultdict(int)
    refs = {}
    for dev in devices:
        info = get_device_info(dev)
        prefix = info["prefix"]
        counters[prefix] += 1
        refs[dev["name"]] = f"{prefix}{counters[prefix]}"
    return refs


# ═══════════════════════════════════════════════════════════════════
#  GVAE Position Prediction
# ═══════════════════════════════════════════════════════════════════

def predict_positions_gvae(graph: dict, weights_path: str):
    """
    Convert graph JSON → PyG Data, run GVAE inference, return
    a list of (x_mm, y_mm) positions in **KiCad coordinates**
    for each device in graph["devices"], plus power symbol positions.

    Returns:
        device_positions: list of (x, y) for each device
        power_positions:  dict  {net_name: (x, y)} for VDD/GND symbols
    """
    import torch
    import numpy as np

    devices = graph["devices"]
    vdd_nets, gnd_nets = detect_power_nets(devices)

    # ── Build net → [node_index] map ──
    net_map = defaultdict(list)

    # Nodes: devices first, then power symbols
    num_devices = len(devices)
    node_types = []  # feature index for each node

    for i, dev in enumerate(devices):
        dev_type = dev["type"]
        model = dev.get("model", "").strip().upper()
        if model == "LED":
            feat_idx = COMP_TYPE_TO_FEAT.get("diode", 4)
        else:
            feat_idx = COMP_TYPE_TO_FEAT.get(dev_type, 4)
        node_types.append(feat_idx)

        for pin in dev["pins"]:
            net_map[pin["net"]].append(i)

    # Add power symbol nodes
    power_node_map = {}  # net_name → node_index
    node_idx = num_devices

    for net in sorted(vdd_nets):
        node_types.append(COMP_TYPE_TO_FEAT["_VCC"])
        power_node_map[net] = node_idx
        net_map[net].append(node_idx)
        node_idx += 1

    for net in sorted(gnd_nets):
        node_types.append(COMP_TYPE_TO_FEAT["_GND"])
        power_node_map[net] = node_idx
        net_map[net].append(node_idx)
        node_idx += 1

    total_nodes = len(node_types)

    # ── Feature vectors (one-hot, 9-dim) ──
    x_tensor = torch.zeros((total_nodes, NUM_FEATURES))
    for i, feat_idx in enumerate(node_types):
        x_tensor[i, feat_idx] = 1.0

    # ── Edge index from shared nets ──
    src, dst = [], []
    for net_name, indices in net_map.items():
        unique = list(set(indices))
        for i in range(len(unique)):
            for j in range(len(unique)):
                if i != j:
                    src.append(unique[i])
                    dst.append(unique[j])

    if not src:
        print("  Warning: no edges found, falling back to grid layout.")
        return None, None

    edge_index = torch.tensor([src, dst], dtype=torch.long)

    # ── Load model and run inference ──
    SchematicGVAE = _define_gvae()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SchematicGVAE(
        in_channels=NUM_FEATURES, hidden_channels=64, latent_dim=32
    ).to(device)

    print(f"  Loading GVAE weights from {weights_path} …")
    state = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    x_tensor = x_tensor.to(device)
    edge_index = edge_index.to(device)

    print("  Running GVAE inference …")
    with torch.no_grad():
        pred_pos, _, _ = model(x_tensor, edge_index)

    pos = pred_pos.cpu().numpy()

    # ── Normalize + scale to sheet ──
    np.random.seed(42)
    pos += np.random.normal(0, 1e-4, pos.shape)  # tie-breaking

    min_v = pos.min(axis=0)
    max_v = pos.max(axis=0)
    rng = max_v - min_v
    rng[rng < 1e-6] = 1.0
    pos = (pos - min_v) / rng

    SHEET_W, SHEET_H = 180.0, 110.0
    OFFSET_X, OFFSET_Y = 55.0, 55.0

    pos[:, 0] = pos[:, 0] * SHEET_W + OFFSET_X
    pos[:, 1] = pos[:, 1] * SHEET_H + OFFSET_Y

    # ── Collision resolution ──
    MIN_SPACING = 40.0
    for _ in range(150):
        moved = False
        for i in range(total_nodes):
            for j in range(i + 1, total_nodes):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = math.hypot(dx, dy)
                if dist < MIN_SPACING:
                    if dist < 0.001:
                        dx, dy = np.random.rand(), np.random.rand()
                        dist = math.hypot(dx, dy)
                    push = (MIN_SPACING - dist) / 2.0
                    px, py = (dx / dist) * push, (dy / dist) * push
                    pos[i, 0] += px;  pos[i, 1] += py
                    pos[j, 0] -= px;  pos[j, 1] -= py
                    moved = True
        if not moved:
            break

    # ── Split into device and power positions ──
    device_positions = [(float(pos[i, 0]), float(pos[i, 1])) for i in range(num_devices)]
    power_positions = {
        net: (float(pos[idx, 0]), float(pos[idx, 1]))
        for net, idx in power_node_map.items()
    }

    return device_positions, power_positions


# ═══════════════════════════════════════════════════════════════════
#  Layout Code Generation
# ═══════════════════════════════════════════════════════════════════

def generate_layout_code(graph: dict, device_positions=None, power_positions=None, args=None) -> str:
    """
    Generate a Python script using the SchGen API.

    Args:
        graph: the circuit graph JSON
        device_positions: if provided, list of (x_mm, y_mm) in KiCad coords
                          for each device in graph["devices"]
        power_positions:  if provided, dict {net_name: (x_mm, y_mm)} for power symbols
        args: command line arguments
    """
    devices = graph["devices"]
    vdd_nets, gnd_nets = detect_power_nets(devices)

    # net → [(dev_name, pin_role)]
    net_map = defaultdict(list)
    for dev in devices:
        for pin in dev["pins"]:
            net_map[pin["net"]].append((dev["name"], pin["role"]))

    ref_map  = assign_references(devices)
    info_map = {d["name"]: get_device_info(d) for d in devices}

    use_gvae = device_positions is not None
    A4_HEIGHT = 297.0  # mm — for Y-flip from KiCad coords to SchGen coords

    lines = [
        "from modules.kicad_sch_interface import *",
        "",
        "# ══════════════════════════════════════",
        "#  Component Placement",
        "# ══════════════════════════════════════",
    ]

    # Manual layout coordinates to match the hand-drawn sketch (Y-inverted due to SchGen REVERSE_Y)
    # Bottom pins of V0, C0, D0 are aligned near y=80 to create a straight bottom rail
    manual_coords = {
        "V0": {"x": 80, "y": 85, "rot": 0},       # Vsource on the left (vertical, bottom pin at ~80)
        "R0": {"x": 115, "y": 120, "rot": 90},     # Resistor on the top (horizontal, at y=120)
        "C0": {"x": 150, "y": 84, "rot": 0},       # Capacitor in the middle (vertical, bottom pin at ~80)
        "R1": {"x": 185, "y": 110, "rot": 180},    # Resistor on the right branch (vertical, top pin at ~115)
        "D0": {"x": 185, "y": 85, "rot": 90},      # LED on the right branch (vertical, pointing down, rot=90)
    }

    # Grid layout defaults
    base_x, base_y, gap = 80, 80, 40

    for i, dev in enumerate(devices):
        info = info_map[dev["name"]]
        ref  = ref_map[dev["name"]]

        value = dev.get("params", {}).get("value", "")
        if not value:
            value = dev.get("model", "") or info["sym"]

        if args.manual and dev["name"] in manual_coords:
            c = manual_coords[dev["name"]]
            lines.append(
                f'add_schematic_symbol('
                f'symbol_lib="{info["lib"]}", symbol_name="{info["sym"]}", '
                f'pos_x={c["x"]}, pos_y={c["y"]}, '
                f'reference="{ref}", value="{value}", rotation={c["rot"]})'
            )
        elif use_gvae:
            kicad_x, kicad_y = device_positions[i]
            # SchGen uses Y-up (REVERSE_Y_FLAG), so flip from KiCad Y-down
            sx = kicad_x
            sy = A4_HEIGHT - kicad_y
            lines.append(
                f'add_schematic_symbol('
                f'symbol_lib="{info["lib"]}", symbol_name="{info["sym"]}", '
                f'pos_x={sx:.2f}, pos_y={sy:.2f}, '
                f'reference="{ref}", value="{value}", rotation=0)'
            )
        else:
            x = base_x + i * gap
            y = base_y
            lines.append(
                f'add_schematic_symbol('
                f'symbol_lib="{info["lib"]}", symbol_name="{info["sym"]}", '
                f'pos_x={x}, pos_y={y}, '
                f'reference="{ref}", value="{value}", rotation=0)'
            )

    # ── Power Symbols ──
    pwr_n = 1
    vdd_pwr, gnd_pwr = {}, {}

    # Skip power symbols entirely in manual mode to maintain a clean closed loop like the hand drawing
    if not (args and args.manual):
        lines += [
            "",
            "# ══════════════════════════════════════",
            "#  Power Symbols",
            "# ══════════════════════════════════════",
        ]
        for net in sorted(vdd_nets):
            pwr_ref = f"#PWR{pwr_n:02d}"
            vdd_pwr[net] = pwr_ref

            if use_gvae and power_positions and net in power_positions:
                kx, ky = power_positions[net]
                sx, sy = kx, A4_HEIGHT - ky
                lines.append(
                    f'add_schematic_symbol(symbol_lib="power", symbol_name="VDD", '
                    f'pos_x={sx:.2f}, pos_y={sy:.2f}, '
                    f'reference="{pwr_ref}", value="VDD", rotation=0)'
                )
            else:
                lines.append(
                    f'add_schematic_symbol(symbol_lib="power", symbol_name="VDD", '
                    f'pos_x={base_x - 20}, pos_y={base_y - 30}, '
                    f'reference="{pwr_ref}", value="VDD", rotation=0)'
                )
            pwr_n += 1

        for net in sorted(gnd_nets):
            pwr_ref = f"#PWR{pwr_n:02d}"
            gnd_pwr[net] = pwr_ref

            if use_gvae and power_positions and net in power_positions:
                kx, ky = power_positions[net]
                sx, sy = kx, A4_HEIGHT - ky
                lines.append(
                    f'add_schematic_symbol(symbol_lib="power", symbol_name="GND", '
                    f'pos_x={sx:.2f}, pos_y={sy:.2f}, '
                    f'reference="{pwr_ref}", value="GND", rotation=0)'
                )
            else:
                lines.append(
                    f'add_schematic_symbol(symbol_lib="power", symbol_name="GND", '
                    f'pos_x={base_x - 20}, pos_y={base_y + 30}, '
                    f'reference="{pwr_ref}", value="GND", rotation=0)'
                )
            pwr_n += 1

    # ── Net Connections ──
    lines += [
        "",
        "# ══════════════════════════════════════",
        "#  Net Connections",
        "# ══════════════════════════════════════",
    ]

    # Map reference name to its X coordinate to sort connections logically from left to right
    ref_x_map = {}
    for i, dev in enumerate(devices):
        ref = ref_map[dev["name"]]
        if args and args.manual and dev["name"] in manual_coords:
            ref_x_map[ref] = manual_coords[dev["name"]]["x"]
        elif use_gvae:
            ref_x_map[ref] = device_positions[i][0]
        else:
            ref_x_map[ref] = base_x + i * gap

    for net_name in sorted(net_map):
        conns = net_map[net_name]
        pin_list = []

        for dev_name, role in conns:
            ref = ref_map[dev_name]
            kicad_pin = info_map[dev_name]["pins"].get(role)
            if kicad_pin is None:
                print(f"  Warning: unknown pin role '{role}' for {dev_name}, skipping.")
                continue
            pin_list.append((ref, kicad_pin))

        # Attach power symbols (placed at left, so give them a small X value)
        if net_name in vdd_pwr:
            ref = vdd_pwr[net_name]
            ref_x_map[ref] = base_x - 30
            pin_list.append((ref, "1"))
        if net_name in gnd_pwr:
            ref = gnd_pwr[net_name]
            ref_x_map[ref] = base_x - 30
            pin_list.append((ref, "1"))

        if len(pin_list) < 2:
            continue

        # Sort pin list by X position (left to right)
        pin_list.sort(key=lambda item: ref_x_map.get(item[0], 0))

        lines.append(f"# Net {net_name}")
        # Daisy-chain connection (connect adjacent components from left to right)
        for idx in range(len(pin_list) - 1):
            ref_a, pin_a = pin_list[idx]
            ref_b, pin_b = pin_list[idx + 1]
            lines.append(
                f'connect_pins("{ref_a}", "{pin_a}", '
                f'"{ref_b}", "{pin_b}")'
            )

    # ── Write wires ──
    lines += [
        "",
        "# ══════════════════════════════════════",
        "#  Finalize Wires",
        "# ══════════════════════════════════════",
        "write_out_all_wires()",
    ]

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Graph JSON → KiCad Schematic (deterministic, optional GVAE layout)"
    )
    parser.add_argument("--graph", required=True,
                        help="Path to input graph JSON file")
    parser.add_argument("--project_name", default="det_schematic",
                        help="Name of the output KiCad project folder")
    parser.add_argument("--use_gvae", action="store_true",
                        help="Use trained GVAE for intelligent component placement")
    parser.add_argument("--gvae_weights", default="../gvae_weights.pth",
                        help="Path to GVAE .pth weights file")
    parser.add_argument("--manual", action="store_true",
                        help="Use manual coordinates to perfectly match the hand-drawn sketch")
    args = parser.parse_args()

    if not os.path.exists(args.graph):
        print(f"Error: graph file '{args.graph}' not found.")
        sys.exit(1)

    graph = load_graph(args.graph)

    # ── Position prediction ──
    device_pos, power_pos = None, None

    if args.use_gvae:
        if not os.path.exists(args.gvae_weights):
            print(f"Warning: GVAE weights '{args.gvae_weights}' not found. "
                  f"Falling back to grid layout.")
        else:
            try:
                device_pos, power_pos = predict_positions_gvae(
                    graph, args.gvae_weights
                )
                if device_pos:
                    print(f"  ✓ GVAE predicted positions for "
                          f"{len(device_pos)} components")
            except Exception as e:
                print(f"  GVAE inference failed: {e}")
                print(f"  Falling back to grid layout.")
                device_pos, power_pos = None, None

    mode = "GVAE" if device_pos else "Grid"
    print(f"\n  Layout mode: {mode}")

    # ── Generate layout script ──
    code = generate_layout_code(graph, device_pos, power_pos, args)

    script = "output_layout_script.py"
    with open(script, "w") as f:
        f.write(code)

    print("\n─── Generated Layout Script ───")
    print(code)
    print("───────────────────────────────\n")

    # ── Compile via init_project.py ──
    print(f"Compiling KiCad project '{args.project_name}' …")
    subprocess.run([
        sys.executable, "init_project.py",
        args.project_name, script, "--overwrite",
    ])
    print("Done.")


if __name__ == "__main__":
    main()
