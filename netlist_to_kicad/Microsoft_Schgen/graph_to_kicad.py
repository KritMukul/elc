#!/usr/bin/env python3
"""
graph_to_kicad.py — Deterministic Graph JSON → KiCad Schematic.

Converts a circuit graph JSON (the AnalogToBi / ELC format) directly into
a KiCad schematic by generating the Python layout script that the Microsoft
SchGen interface expects, then running init_project.py to compile it.

No LLM is used — this is a pure rule-based conversion, so component values,
net connectivity, and symbol names are guaranteed to be correct.

Usage (on DGX):
    python graph_to_kicad.py --graph input_graph.json --project_name my_circuit
"""

import json
import argparse
import subprocess
import sys
import os
from collections import defaultdict


# ═══════════════════════════════════════════════════════════════════
#  Device type  →  KiCad symbol mapping
# ═══════════════════════════════════════════════════════════════════
DEVICE_MAP = {
    # type        lib       symbol          pin-role → kicad-pin   ref-prefix
    "cap":     {"lib": "Device", "sym": "C",       "pins": {"P": "1",  "N": "2"},              "prefix": "C"},
    "res":     {"lib": "Device", "sym": "R",       "pins": {"P": "1",  "N": "2"},              "prefix": "R"},
    "ind":     {"lib": "Device", "sym": "L",       "pins": {"P": "1",  "N": "2"},              "prefix": "L"},
    "diode":   {"lib": "Device", "sym": "D",       "pins": {"A": "A",  "K": "K"},              "prefix": "D"},
    "vsource": {"lib": "Device", "sym": "Battery", "pins": {"P": "+",  "N": "-"},              "prefix": "BT"},
    "isource": {"lib": "Device", "sym": "Battery", "pins": {"P": "+",  "N": "-"},              "prefix": "I"},
    "nfet":    {"lib": "Device", "sym": "Q_NMOS_GDS", "pins": {"G": "1", "D": "2", "S": "3"}, "prefix": "M"},
    "pfet":    {"lib": "Device", "sym": "Q_PMOS_GDS", "pins": {"G": "1", "D": "2", "S": "3"}, "prefix": "M"},
}

# Model-level overrides (checked first, before DEVICE_MAP)
MODEL_OVERRIDES = {
    "LED": {"lib": "Device", "sym": "LED", "pins": {"A": "A", "K": "K"}, "prefix": "D"},
}


# ───────────────────────────────────────────────────────────────────
#  Helpers
# ───────────────────────────────────────────────────────────────────
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
        print(f"Warning: unknown device type '{dev['type']}', falling back to resistor symbol.")
        return DEVICE_MAP["res"]
    return info


def detect_power_nets(devices: list):
    """Identify VDD / GND nets by looking at voltage-source terminals."""
    vdd_nets, gnd_nets = set(), set()
    for dev in devices:
        if dev["type"] == "vsource":
            for pin in dev["pins"]:
                if pin["role"] == "P":
                    vdd_nets.add(pin["net"])
                elif pin["role"] == "N":
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


# ───────────────────────────────────────────────────────────────────
#  Code generation
# ───────────────────────────────────────────────────────────────────
def generate_layout_code(graph: dict) -> str:
    """
    Produce a self-contained Python script that uses the SchGen interface
    (add_schematic_symbol, connect_pins, write_out_all_wires) to place
    and wire every component described in the graph JSON.
    """
    devices   = graph["devices"]
    vdd_nets, gnd_nets = detect_power_nets(devices)

    # net → list of (device_name, pin_role)
    net_map = defaultdict(list)
    for dev in devices:
        for pin in dev["pins"]:
            net_map[pin["net"]].append((dev["name"], pin["role"]))

    ref_map  = assign_references(devices)
    info_map = {d["name"]: get_device_info(d) for d in devices}

    lines = [
        "from modules.kicad_sch_interface import *",
        "",
        "# ══════════════════════════════════════",
        "#  Component Placement (grid layout)",
        "# ══════════════════════════════════════",
    ]

    # Layout parameters  (all values are in "units" — SchGen multiplies
    # integers by 1.27 mm to land on the 50-mil KiCad grid)
    base_x  = 80   # starting column
    base_y  = 80   # row for the main components
    gap     = 40   # horizontal spacing between components

    for i, dev in enumerate(devices):
        info  = info_map[dev["name"]]
        ref   = ref_map[dev["name"]]
        x     = base_x + i * gap
        y     = base_y

        value = dev.get("params", {}).get("value", "")
        if not value:
            value = dev.get("model", "") or info["sym"]

        lines.append(
            f'add_schematic_symbol('
            f'symbol_lib="{info["lib"]}", symbol_name="{info["sym"]}", '
            f'pos_x={x}, pos_y={y}, '
            f'reference="{ref}", value="{value}", rotation=0)'
        )

    # ── Power symbols ──────────────────────────────────────────────
    lines += [
        "",
        "# ══════════════════════════════════════",
        "#  Power Symbols",
        "# ══════════════════════════════════════",
    ]
    pwr_n    = 1
    vdd_refs = {}        # net_name → power-symbol reference
    gnd_refs = {}

    for net in sorted(vdd_nets):
        pwr_ref = f"#PWR{pwr_n:02d}"
        vdd_refs[net] = pwr_ref
        vdd_x = base_x - 20
        vdd_y = base_y - 30
        lines.append(
            f'add_schematic_symbol(symbol_lib="power", symbol_name="VDD", '
            f'pos_x={vdd_x}, pos_y={vdd_y}, '
            f'reference="{pwr_ref}", value="VDD", rotation=0)'
        )
        pwr_n += 1

    for net in sorted(gnd_nets):
        pwr_ref = f"#PWR{pwr_n:02d}"
        gnd_refs[net] = pwr_ref
        gnd_x = base_x - 20
        gnd_y = base_y + 30
        lines.append(
            f'add_schematic_symbol(symbol_lib="power", symbol_name="GND", '
            f'pos_x={gnd_x}, pos_y={gnd_y}, '
            f'reference="{pwr_ref}", value="GND", rotation=0)'
        )
        pwr_n += 1

    # ── Net connections ────────────────────────────────────────────
    lines += [
        "",
        "# ══════════════════════════════════════",
        "#  Net Connections",
        "# ══════════════════════════════════════",
    ]

    for net_name in sorted(net_map):
        conns = net_map[net_name]

        # Build list of (kicad_ref, kicad_pin) on this net
        pin_list = []
        for dev_name, role in conns:
            ref      = ref_map[dev_name]
            kicad_pin = info_map[dev_name]["pins"][role]
            pin_list.append((ref, kicad_pin))

        # Attach power symbols that sit on this net
        if net_name in vdd_refs:
            pin_list.append((vdd_refs[net_name], "1"))
        if net_name in gnd_refs:
            pin_list.append((gnd_refs[net_name], "1"))

        if len(pin_list) < 2:
            continue

        lines.append(f"# Net {net_name}")
        anchor_ref, anchor_pin = pin_list[0]
        for ref, pin in pin_list[1:]:
            lines.append(
                f'connect_pins("{anchor_ref}", "{anchor_pin}", '
                f'"{ref}", "{pin}")'
            )

    # ── Write wires ────────────────────────────────────────────────
    lines += [
        "",
        "# ══════════════════════════════════════",
        "#  Finalize Wires",
        "# ══════════════════════════════════════",
        "write_out_all_wires()",
    ]

    return "\n".join(lines)


# ───────────────────────────────────────────────────────────────────
#  Main
# ───────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Deterministic: Graph JSON → KiCad Schematic (no LLM)"
    )
    parser.add_argument("--graph", required=True,
                        help="Path to the input graph JSON file")
    parser.add_argument("--project_name", default="det_schematic",
                        help="Name of the output KiCad project folder")
    args = parser.parse_args()

    if not os.path.exists(args.graph):
        print(f"Error: graph file '{args.graph}' not found.")
        sys.exit(1)

    graph = load_graph(args.graph)
    code  = generate_layout_code(graph)

    script = "output_layout_script.py"
    with open(script, "w") as f:
        f.write(code)

    print("─── Generated Layout Script ───")
    print(code)
    print("───────────────────────────────\n")

    print(f"Compiling KiCad project '{args.project_name}' …")
    subprocess.run([
        sys.executable, "init_project.py",
        args.project_name, script, "--overwrite",
    ])
    print("Done.")


if __name__ == "__main__":
    main()
