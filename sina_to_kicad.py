#!/usr/bin/env python3
"""
sina_to_kicad.py — SINA graph JSON → KiCad schematic with electrically connected nets.

Why this module exists
----------------------
The old ``--sina`` path in ``netlist_to_kicad/Microsoft_Schgen/graph_to_kicad.py``
replayed SINA's *image pixel* wire segments straight into the .kicad_sch file.
That can never connect anything:

  * KiCad only bonds a wire to a pin when the wire endpoint sits on the pin's
    exact coordinate. Scaled pixel coordinates land on the symbol's centre —
    inside the body, touching no pin.
  * Symbols and wires went through two different Y flips (``297 - y`` in
    graph_to_kicad, then ``210 - y`` again inside kicad_sch_interface, which the
    raw wires skipped entirely), so symbols ended up at negative Y while their
    own wires sat 300 mm away.
  * Nothing was snapped to KiCad's 1.27 mm grid and no junctions were emitted,
    so even a visually touching T-connection stayed electrically open.

So the pixel geometry is kept only as a *placement hint* and thrown away as
*routing* data. Connectivity is rebuilt from the netlist instead:

  1. clean the netlist so each net lands on its own terminal of each device
  2. place every device, using the SINA bbox for relative position + orientation
  3. work out each pin's exact coordinate from the .kicad_sym geometry
  4. route orthogonal wires between those coordinates, on KiCad's 1.27 mm grid
  5. drop junction dots wherever three or more conductors of one net meet
  6. verify every net is a single electrical node before writing the file

Step 6 re-derives connectivity under KiCad's own rules and reports per net;
``--verify`` additionally cross-checks against ``kicad-cli``'s netlist export,
which runs the same engine eeschema does. A non-zero exit means the schematic
really is broken, not merely ugly.

Usage
-----
    python sina_to_kicad.py --graph sina_out/foo_graph.json --output foo.kicad_sch
    python sina_to_kicad.py --graph foo_graph.json --output foo.kicad_sch --verify
"""

import argparse
import heapq
import json
import math
import os
import shutil
import subprocess
import sys
from collections import defaultdict

try:
    import kicad_sch_api as ksa
except ImportError:  # pragma: no cover - dependency check happens in main()
    ksa = None


# ═══════════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════════

GRID = 1.27                 # mm — KiCad's default schematic grid (50 mil)
STUB = 2 * GRID             # mm — how far a wire escapes a pin before turning
MIN_PITCH = 8 * GRID        # mm — minimum centre-to-centre spacing of symbols
BODY_MARGIN = GRID          # mm — clearance kept around a symbol body

# KiCad's A4 sheet is landscape: 297 x 210 mm. The margin has to clear half a
# symbol body plus its escape stub, or wires run off the sheet.
SHEET_W, SHEET_H = 297.0, 210.0
AREA = (30.0, 35.0, SHEET_W - 30.0, SHEET_H - 35.0)   # x0, y0, x1, y1

GROUND_NET_NAMES = {"0", "gnd", "ground", "vss", "agnd", "dgnd"}


# ═══════════════════════════════════════════════════════════════════
#  Device type → KiCad symbol
# ═══════════════════════════════════════════════════════════════════
#
#  axis  = which way the pins point at rotation 0 in the KiCad library.
#          "V" = pins above/below the body, "H" = pins left/right.
#          R/C/L/Battery are drawn vertically, D/LED horizontally.
#  pins  = KiCad pin identifiers in terminal order. Verified against the
#          installed KiCad 10 Device library (Q_NMOS_GDS, which the old
#          DEVICE_MAP used, does not exist there — it is Q_NMOS).

DEVICE_MAP = {
    "resistor":     {"lib_id": "Device:R_US",      "prefix": "R",  "axis": "V", "pins": ["1", "2"]},
    "capacitor":    {"lib_id": "Device:C",         "prefix": "C",  "axis": "V", "pins": ["1", "2"]},
    "capacitor_p":  {"lib_id": "Device:C_Polarized", "prefix": "C", "axis": "V", "pins": ["1", "2"]},
    "inductor":     {"lib_id": "Device:L",         "prefix": "L",  "axis": "V", "pins": ["1", "2"]},
    "diode":        {"lib_id": "Device:D",         "prefix": "D",  "axis": "H", "pins": ["1", "2"]},
    "led":          {"lib_id": "Device:LED",       "prefix": "D",  "axis": "H", "pins": ["1", "2"]},
    "vsource":      {"lib_id": "Device:Battery",   "prefix": "BT", "axis": "V", "pins": ["1", "2"]},
    "isource":      {"lib_id": "Simulation_SPICE:IDC", "prefix": "I", "axis": "V", "pins": ["1", "2"]},
    "nmos":         {"lib_id": "Device:Q_NMOS",    "prefix": "Q",  "axis": "V", "pins": ["G", "D", "S"]},
    "pmos":         {"lib_id": "Device:Q_PMOS",    "prefix": "Q",  "axis": "V", "pins": ["G", "D", "S"]},
    "npn":          {"lib_id": "Device:Q_NPN",     "prefix": "Q",  "axis": "V", "pins": ["B", "C", "E"]},
    "pnp":          {"lib_id": "Device:Q_PNP",     "prefix": "Q",  "axis": "V", "pins": ["B", "C", "E"]},
}

# Everything SINA / YOLO / the SPICE parser might call a part, folded onto the
# canonical keys above.
TYPE_ALIASES = {
    "res": "resistor", "r": "resistor", "resistor.adjustable": "resistor",
    "potentiometer": "resistor", "rheostat": "resistor",
    "cap": "capacitor", "c": "capacitor", "capacitor.unpolarized": "capacitor",
    "capacitor.polarized": "capacitor_p", "electrolytic": "capacitor_p",
    "ind": "inductor", "l": "inductor", "coil": "inductor",
    "d": "diode", "diode.light_emitting": "led", "diode.zener": "diode",
    "lamp": "led",
    "voltage.dc": "vsource", "voltage.ac": "vsource", "vdc": "vsource",
    "v": "vsource", "battery": "vsource", "voltage_src": "vsource",
    "voltage_source": "vsource", "probe.voltage": "vsource",
    "current.dc": "isource", "i": "isource", "idc": "isource",
    "current_src": "isource", "current_source": "isource",
    "dependent_source": "isource", "vccs": "isource",
    "gnd": "gnd", "ground": "gnd", "vss": "gnd", "earth": "gnd",
    "mosfet": "nmos", "nfet": "nmos", "transistor.mosfet": "nmos",
    "pfet": "pmos", "m": "nmos",
    "bjt": "npn", "transistor.bjt": "npn", "q": "npn",
}

# SINA emits P/N roles; SPICE-derived graphs may use words. Index into
# DEVICE_MAP[...]["pins"].
ROLE_ORDER = {
    "p": 0, "pos": 0, "+": 0, "a": 0, "anode": 0, "1": 0,
    "n": 1, "neg": 1, "-": 1, "k": 1, "cathode": 1, "2": 1,
    "g": 0, "gate": 0, "b": 0, "base": 0,
    "d": 1, "drain": 1, "c": 1, "collector": 1,
    "s": 2, "source": 2, "e": 2, "emitter": 2, "3": 2,
}


def canonical_type(raw: str):
    """
    Fold an arbitrary detector/parser class name onto a DEVICE_MAP key.

    Returns None when nothing matches, so the caller can say so out loud. This
    used to fall through to "resistor" silently, which is how a detected current
    source and a ground symbol both turned into resistors in a schematic that
    otherwise verified clean — eight resistors where the netlist had six.
    """
    t = (raw or "").strip().lower().replace(" ", "_").replace("-", "_")
    if t in DEVICE_MAP or t == "gnd":
        return t
    if t in TYPE_ALIASES:
        return TYPE_ALIASES[t]
    for alias, canon in TYPE_ALIASES.items():          # substring fallback
        if alias in t and len(alias) > 2:
            return canon
    return None


def is_ground_net(name: str) -> bool:
    return str(name).strip().lower() in GROUND_NET_NAMES


# ═══════════════════════════════════════════════════════════════════
#  Symbol geometry — read straight from the installed KiCad libraries
# ═══════════════════════════════════════════════════════════════════
#
#  kicad_sch_api's list_component_pins() reports the wrong pin coordinates for
#  rotated symbols: at 90° and 270° a two-pin part comes back with pin 1 and
#  pin 2 swapped, and a three-pin part lands nowhere near its real pins. Verified
#  against kicad-cli's own netlist export, which is what eeschema itself uses.
#
#  So the pin coordinates are derived here instead, from the .kicad_sym file plus
#  KiCad's placement transform (symbol files use Y-up, schematics use Y-down):
#
#      0°    (X + px, Y - py)        180°   (X - px, Y + py)
#      90°   (X - py, Y - px)        270°   (X + py, Y + px)

_SYMBOL_CACHE = {}


def _sexp_parse(text: str):
    """Minimal S-expression reader — enough for .kicad_sym files."""
    tokens, i, n = [], 0, len(text)
    stack, current = [], []
    while i < n:
        ch = text[i]
        if ch == "(":
            stack.append(current)
            current = []
            i += 1
        elif ch == ")":
            done = current
            current = stack.pop() if stack else []
            current.append(done)
            i += 1
        elif ch == '"':
            j = i + 1
            buf = []
            while j < n and text[j] != '"':
                if text[j] == "\\":
                    buf.append(text[j + 1])
                    j += 2
                else:
                    buf.append(text[j])
                    j += 1
            current.append("".join(buf))
            i = j + 1
        elif ch.isspace():
            i += 1
        else:
            j = i
            while j < n and not text[j].isspace() and text[j] not in "()\"":
                j += 1
            current.append(text[i:j])
            i = j
    tokens = current
    return tokens[0] if len(tokens) == 1 else tokens


def symbol_lib_dir() -> str:
    """Locate the installed KiCad symbol library directory."""
    env = os.environ.get("KICAD_SYMBOL_DIR")
    if env and os.path.isdir(env):
        return env
    roots = [
        r"C:\Program Files\KiCad", r"C:\Program Files (x86)\KiCad",
        "/usr/share/kicad", "/usr/local/share/kicad",
        "/Applications/KiCad/KiCad.app/Contents/SharedSupport",
    ]
    for root in roots:
        direct = os.path.join(root, "symbols")
        if os.path.isdir(direct):
            return direct
        if os.path.isdir(root):
            for ver in sorted(os.listdir(root), reverse=True):
                cand = os.path.join(root, ver, "share", "kicad", "symbols")
                if os.path.isdir(cand):
                    return cand
    raise RuntimeError(
        "Could not find the KiCad symbol libraries. "
        "Set KICAD_SYMBOL_DIR to the directory holding Device.kicad_sym."
    )


def load_symbol_pins(lib_id: str):
    """
    Return [(pin_number, (local_x, local_y))] for a "Lib:Symbol" id.

    Pin coordinates in a .kicad_sym file are already the connection points, in
    the symbol's own Y-up frame.
    """
    if lib_id in _SYMBOL_CACHE:
        return _SYMBOL_CACHE[lib_id]

    lib_name, sym_name = lib_id.split(":", 1)
    path = os.path.join(symbol_lib_dir(), lib_name + ".kicad_sym")
    if not os.path.exists(path):
        raise RuntimeError(f"symbol library not found: {path}")

    with open(path, encoding="utf-8") as f:
        tree = _sexp_parse(f.read())

    target = None
    for node in tree:
        if isinstance(node, list) and node and node[0] == "symbol" and node[1] == sym_name:
            target = node
            break
    if target is None:
        raise RuntimeError(f"symbol '{sym_name}' not found in {path}")

    pins = []

    def walk(node):
        for child in node:
            if not isinstance(child, list) or not child:
                continue
            if child[0] == "pin":
                at = num = None
                for item in child:
                    if isinstance(item, list) and item:
                        if item[0] == "at":
                            at = (float(item[1]), float(item[2]))
                        elif item[0] == "number":
                            num = item[1]
                if at is not None and num is not None:
                    pins.append((num, at))
            else:
                walk(child)

    walk(target)
    _SYMBOL_CACHE[lib_id] = pins
    return pins


def pin_position(lib_id: str, number: str, x: float, y: float, rotation: int):
    """Absolute schematic coordinate of one pin of a placed symbol."""
    for num, (px, py) in load_symbol_pins(lib_id):
        if num != number:
            continue
        rot = int(rotation) % 360
        if rot == 0:
            return (q(x + px), q(y - py))
        if rot == 90:
            return (q(x - py), q(y - px))
        if rot == 180:
            return (q(x - px), q(y + py))
        if rot == 270:
            return (q(x + py), q(y + px))
        raise ValueError(f"unsupported rotation {rotation}")
    raise KeyError(f"pin {number} not found on {lib_id}")


# ═══════════════════════════════════════════════════════════════════
#  Geometry helpers
# ═══════════════════════════════════════════════════════════════════

def q(v: float) -> float:
    """Quantise to the 2 decimals KiCad stores, so equality comparisons hold."""
    return round(float(v), 2)


def key(pt) -> tuple:
    return (q(pt[0]), q(pt[1]))


def snap(v: float) -> float:
    return q(round(v / GRID) * GRID)


def on_segment(pt, seg, eps=1e-6) -> bool:
    """True if pt lies on the axis-aligned segment seg = (a, b), endpoints included."""
    (ax, ay), (bx, by) = seg
    px, py = pt
    if abs(ax - bx) < eps:                       # vertical
        return abs(px - ax) < eps and min(ay, by) - eps <= py <= max(ay, by) + eps
    if abs(ay - by) < eps:                       # horizontal
        return abs(py - ay) < eps and min(ax, bx) - eps <= px <= max(ax, bx) + eps
    # diagonal — should not happen, but handle it so the verifier stays honest
    cross = (px - ax) * (by - ay) - (py - ay) * (bx - ax)
    if abs(cross) > 1e-3:
        return False
    return (min(ax, bx) - eps <= px <= max(ax, bx) + eps
            and min(ay, by) - eps <= py <= max(ay, by) + eps)


def is_interior(pt, seg, eps=1e-6) -> bool:
    """True if pt lies on seg but is not one of its endpoints."""
    return on_segment(pt, seg, eps) and key(pt) not in (key(seg[0]), key(seg[1]))


# ═══════════════════════════════════════════════════════════════════
#  Stage 1 — netlist normalisation
# ═══════════════════════════════════════════════════════════════════

def normalize_graph(graph: dict, verbose=True) -> tuple:
    """
    Turn a raw SINA graph into a netlist that can actually be drawn.

    SINA appends one pin per net that happens to touch a device's bounding box,
    tagged only "P" or "N". That produces two failure modes the emitter used to
    walk straight into:

      * a 2-terminal device carrying three or more nets
      * two different nets both tagged "P", which then both map to KiCad pin "1"
        and silently short together

    Both are resolved here by assigning each attached net a *distinct* terminal,
    ordered along the device's dominant axis, and dropping the surplus.

    Returns (devices, net_to_pins) where each device is a dict with keys
    name / ref / lib_id / type / value / bbox / axis / terminals, and
    net_to_pins maps net name → [(ref, kicad_pin_id)].
    """
    raw_devices = graph.get("devices", [])
    # SINA writes nets as {"name": ..., "wires": [...]}; SPICE-derived graphs
    # write them as bare name strings.
    nets_by_name = {}
    for n in graph.get("nets", []):
        if isinstance(n, dict):
            nets_by_name[n.get("name")] = n
        else:
            nets_by_name[n] = {"name": n, "wires": []}

    counters = defaultdict(int)
    devices = []
    net_to_pins = defaultdict(list)
    ground_nets = set()
    warnings = []

    for dev in raw_devices:
        ctype = canonical_type(dev.get("type", ""))
        if ctype is None:
            warnings.append(
                f"{dev.get('name', '?')}: unrecognised type "
                f"'{dev.get('type')}' — drawing it as a resistor"
            )
            ctype = "resistor"

        # A detected ground symbol is not a part to place, it is a statement
        # about the net it sits on. Marking the net gets a power:GND at every
        # pin of it, and KiCad ties all of those together globally.
        if ctype == "gnd":
            for pin in dev.get("pins", []):
                if pin.get("net") is not None:
                    ground_nets.add(pin["net"])
            continue

        info = DEVICE_MAP[ctype]
        n_term = len(info["pins"])

        counters[info["prefix"]] += 1
        ref = f'{info["prefix"]}{counters[info["prefix"]]}'

        bbox = dev.get("bbox")
        axis = infer_axis(dev, bbox or [0, 0, 10, 10], nets_by_name)

        # ── collect the distinct nets this device touches ──
        seen, attached = set(), []
        for pin in dev.get("pins", []):
            net = pin.get("net")
            if net is None or net in seen:
                continue                       # same net twice = one terminal
            seen.add(net)
            attached.append((net, pin.get("role", "")))

        if len(attached) > n_term:
            dropped = [n for n, _ in attached[n_term:]]
            warnings.append(
                f"{dev.get('name', ref)} ({ctype}) touched {len(attached)} nets "
                f"but has {n_term} terminals — dropping {dropped}"
            )
            attached = attached[:n_term]

        # ── give each net its own terminal ──
        # Prefer the terminal the role asks for; fall back to the first free one
        # so two "P" roles can never collide on the same KiCad pin.
        assignment = [None] * n_term
        deferred = []
        for net, role in attached:
            want = ROLE_ORDER.get(str(role).strip().lower())
            if want is not None and want < n_term and assignment[want] is None:
                assignment[want] = net
            else:
                deferred.append((net, role))
        for net, role in deferred:
            free = next((i for i in range(n_term) if assignment[i] is None), None)
            if free is None:
                warnings.append(f"{ref}: no free terminal left for net {net}")
                continue
            assignment[free] = net
            if role:
                warnings.append(
                    f"{ref}: role '{role}' already taken, net {net} moved to "
                    f"pin {info['pins'][free]}"
                )

        terminals = []
        for idx, net in enumerate(assignment):
            if net is None:
                continue
            kicad_pin = info["pins"][idx]
            terminals.append({"pin": kicad_pin, "net": net})
            net_to_pins[net].append((ref, kicad_pin))

        value = dev.get("params", {}).get("value") or dev.get("model") or ""

        devices.append({
            "name": dev.get("name", ref),
            "ref": ref,
            "type": ctype,
            "lib_id": info["lib_id"],
            "value": value,
            "bbox": bbox,
            "axis": axis,
            "terminals": terminals,
        })

    # A net carrying a ground symbol needs no second device pin — the power
    # symbol is its other end — so those are not dangling.
    dangling = [n for n, pins in net_to_pins.items()
                if len(pins) < 2 and n not in ground_nets]
    for n in dangling:
        warnings.append(f"net {n} has only {len(net_to_pins[n])} pin — left unrouted")

    if verbose:
        for w in warnings:
            print(f"  ! {w}")

    return devices, dict(net_to_pins), ground_nets


def infer_axis(dev: dict, bbox, nets_by_name: dict) -> str:
    """
    Decide whether a part should sit horizontally or vertically.

    An elongated bounding box settles it on its own — it is a direct measurement
    of how the symbol was drawn, so a box two and a half times taller than it is
    wide is a vertical part and nothing else needs asking. Only a roughly square
    box is ambiguous, and there the wire nodes SINA stores break the tie:
    whichever way the attached nets approach is the way the pins must point.

    Consulting the wires first got this backwards. A capacitor drawn 12 x 33 px
    came out horizontal because its nets happened to approach from the side,
    which put it across the rail it was supposed to hang from.
    """
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0

    w, h = x2 - x1, y2 - y1
    if w > h * 1.4:
        return "H"
    if h > w * 1.4:
        return "V"

    horiz = vert = 0
    for pin in dev.get("pins", []):
        net = nets_by_name.get(pin.get("net"))
        if not net:
            continue
        best, best_d = None, float("inf")
        for wire in net.get("wires", []):
            if len(wire) < 4:
                continue
            for px, py in ((wire[0], wire[1]), (wire[2], wire[3])):
                d = math.hypot(px - cx, py - cy)
                if d < best_d:
                    best_d, best = d, (px, py)
        if best is None:
            continue
        if abs(best[0] - cx) > abs(best[1] - cy):
            horiz += 1
        else:
            vert += 1

    if horiz > vert:
        return "H"
    if vert > horiz:
        return "V"

    return "H" if w > h else "V"


# ═══════════════════════════════════════════════════════════════════
#  Stage 2 — placement
# ═══════════════════════════════════════════════════════════════════

def plan_placement(devices: list) -> None:
    """
    Map SINA pixel positions onto the sheet, in place (adds x / y / rotation).

    KiCad schematic Y grows downward, exactly like image pixel Y, so there is no
    flip here at all — the flips in the old code were the bug, not the fix.
    """
    if not devices:
        return

    # SPICE-derived graphs carry no pixel geometry at all — fall back to a grid
    # so those still produce a readable sheet.
    if not any(d.get("bbox") for d in devices):
        cols = max(1, int(math.ceil(math.sqrt(len(devices)))))
        for i, d in enumerate(devices):
            d["bbox"] = [(i % cols) * 100, (i // cols) * 100,
                         (i % cols) * 100 + 40, (i // cols) * 100 + 60]

    centers = []
    for d in devices:
        x1, y1, x2, y2 = d["bbox"]
        centers.append(((x1 + x2) / 2.0, (y1 + y2) / 2.0))

    px_min_x = min(c[0] for c in centers)
    px_max_x = max(c[0] for c in centers)
    px_min_y = min(c[1] for c in centers)
    px_max_y = max(c[1] for c in centers)

    ax0, ay0, ax1, ay1 = AREA
    span_x, span_y = px_max_x - px_min_x, px_max_y - px_min_y

    # Aspect-preserving fit, then blown up (within the sheet) until neighbouring
    # parts are at least MIN_PITCH apart, so symbols never overlap their wires.
    fit = min((ax1 - ax0) / span_x if span_x > 1e-6 else float("inf"),
              (ay1 - ay0) / span_y if span_y > 1e-6 else float("inf"))
    if not math.isfinite(fit):
        fit = 1.0

    closest = float("inf")
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            d = math.hypot(centers[i][0] - centers[j][0], centers[i][1] - centers[j][1])
            if 1e-6 < d < closest:
                closest = d
    scale = fit
    if math.isfinite(closest) and closest * scale < MIN_PITCH:
        scale = min(fit, MIN_PITCH / closest) if fit >= MIN_PITCH / closest else fit

    off_x = ax0 + max(0.0, ((ax1 - ax0) - span_x * scale) / 2.0)
    off_y = ay0 + max(0.0, ((ay1 - ay0) - span_y * scale) / 2.0)

    for d, (cx, cy) in zip(devices, centers):
        d["x"] = snap(off_x + (cx - px_min_x) * scale)
        d["y"] = snap(off_y + (cy - px_min_y) * scale)
        default_axis = DEVICE_MAP[d["type"]]["axis"]
        # 3-terminal parts have no meaningful "axis" — leave them upright.
        if len(DEVICE_MAP[d["type"]]["pins"]) > 2:
            d["rotation"] = 0
        else:
            d["rotation"] = 0 if d["axis"] == default_axis else 90

    _separate(devices)

    for d in devices:                       # keep everything on the sheet
        d["x"] = snap(min(max(d["x"], AREA[0]), AREA[2]))
        d["y"] = snap(min(max(d["y"], AREA[1]), AREA[3]))


def _separate(devices: list, iterations: int = 60) -> None:
    """Push overlapping symbols apart along the axis of least overlap."""
    for _ in range(iterations):
        moved = False
        for i in range(len(devices)):
            for j in range(i + 1, len(devices)):
                a, b = devices[i], devices[j]
                dx, dy = b["x"] - a["x"], b["y"] - a["y"]
                if abs(dx) >= MIN_PITCH or abs(dy) >= MIN_PITCH:
                    continue
                push = GRID
                if abs(dx) >= abs(dy):
                    s = 1.0 if dx >= 0 else -1.0
                    a["x"] = snap(a["x"] - s * push)
                    b["x"] = snap(b["x"] + s * push)
                else:
                    s = 1.0 if dy >= 0 else -1.0
                    a["y"] = snap(a["y"] - s * push)
                    b["y"] = snap(b["y"] + s * push)
                moved = True
        if not moved:
            return


# ═══════════════════════════════════════════════════════════════════
#  Stage 3 — routing
# ═══════════════════════════════════════════════════════════════════

def escape_direction(pin_pt, body_center):
    """Unit direction a wire must leave a pin, so it never crosses the body."""
    dx = pin_pt[0] - body_center[0]
    dy = pin_pt[1] - body_center[1]
    if abs(dx) >= abs(dy):
        return (1.0 if dx > 0 else -1.0, 0.0) if abs(dx) > 1e-6 else (0.0, -1.0)
    return (0.0, 1.0 if dy > 0 else -1.0)


def body_boxes(placed_pins: dict, centers: dict) -> list:
    """
    Conservative keep-out box per symbol, derived from its real pin coordinates.

    kicad_sch_api exposes no bounding box, but the pin extents plus a margin
    bound the body closely enough to steer wires around it — which matters
    because a wire crossing a body also crosses its pins.
    """
    boxes = []
    for ref, pts in placed_pins.items():
        coords = [p for (_, p) in pts]
        if not coords:
            continue
        cx, cy = centers.get(ref, coords[0])
        xs = [p[0] for p in coords] + [cx]
        ys = [p[1] for p in coords] + [cy]
        boxes.append((min(xs) - BODY_MARGIN, min(ys) - BODY_MARGIN,
                      max(xs) + BODY_MARGIN, max(ys) + BODY_MARGIN))
    return boxes


def _path_segments(points):
    """Turn a polyline into segments, dropping zero-length hops."""
    segs = []
    for a, b in zip(points, points[1:]):
        if key(a) != key(b):
            segs.append((key(a), key(b)))
    return segs


#  Routing is a maze search on KiCad's own 1.27 mm lattice, because the rules
#  that decide whether two things are connected are all lattice-local:
#
#    * a wire crossing a symbol body also crosses its pins, and a wire lying on
#      a pin bonds to it — so bodies are blocked cells
#    * two nets sharing an endpoint, or running along the same line, become one
#      net — so another net's wires block *travel along them* and its corners
#      block as cells
#    * two nets crossing at right angles stay separate unless a junction says
#      otherwise — so a perpendicular crossing is left free
#
#  Encoding those three rules directly is both simpler and stricter than the
#  L-shape / trunk heuristics it replaces, which could always be cornered into
#  a layout where every candidate shape shorted something.

DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))
TURN_PENALTY = 4            # prefer long straight runs over staircases
LANE_PENALTY = 2            # prefer a lane no other net is already using
SEARCH_PAD = 20             # lattice cells of slack around the routed area


def idx(v: float) -> int:
    return int(round(v / GRID))


def cell_of(pt) -> tuple:
    return (idx(pt[0]), idx(pt[1]))


def point_of(cell) -> tuple:
    return (q(cell[0] * GRID), q(cell[1] * GRID))


def _blocked_cells(boxes) -> set:
    """Lattice cells sitting inside a symbol body (its pins included)."""
    blocked = set()
    for x0, y0, x1, y1 in boxes:
        for i in range(idx(x0) + 1, idx(x1)):
            for j in range(idx(y0) + 1, idx(y1)):
                p = point_of((i, j))
                if x0 < p[0] < x1 and y0 < p[1] < y1:
                    blocked.add((i, j))
    return blocked


def _walk_cells(seg):
    """Every lattice cell along an axis-aligned segment, endpoints included."""
    a, b = cell_of(seg[0]), cell_of(seg[1])
    di = (b[0] > a[0]) - (b[0] < a[0])
    dj = (b[1] > a[1]) - (b[1] < a[1])
    if di and dj:                       # diagonal fallback — should not occur
        return [a, b]
    cells = [a]
    cur = a
    while cur != b:
        cur = (cur[0] + di, cur[1] + dj)
        cells.append(cur)
        if len(cells) > 5000:
            break
    return cells


def _occupancy(foreign):
    """
    Turn other nets' wires into obstacles.

    Three different restrictions, matching three different KiCad behaviours:

      nodes   their corners and ends — landing on one shares a point, which is
              a short, so those cells are closed outright
      along   the unit edges they cover — running along a wire fuses with it
      noturn  every cell they pass through — crossing at right angles is fine
              and stays allowed, but *turning* on one leaves an endpoint sitting
              on their wire, which is a T-connection and therefore a short
    """
    nodes, along, noturn = set(), set(), set()
    busy_rows, busy_cols = set(), set()
    for seg in foreign:
        cells = _walk_cells(seg)
        nodes.add(cells[0])
        nodes.add(cells[-1])
        noturn.update(cells)
        for a, b in zip(cells, cells[1:]):
            along.add(frozenset((a, b)))
            if a[1] == b[1]:
                busy_rows.add(a[1])
            else:
                busy_cols.add(a[0])
    return nodes, along, noturn - nodes, busy_rows, busy_cols


def _maze_route(starts, targets, blocked, blocked_edges, no_turn, bounds,
                busy_rows=(), busy_cols=()):
    """
    Cheapest lattice path from any start cell to any target cell.

    Dijkstra over (cell, incoming direction) so a turn can be priced — and so a
    turn can be forbidden outright on a cell belonging to another net. The
    result is a Manhattan route with as few bends as the obstacles allow, or
    None when the net is fenced in.
    """
    i0, j0, i1, j1 = bounds
    target_set = set(targets)
    start_set = set(starts)
    if not target_set or not start_set:
        return None

    heap = []
    best = {}
    for s in start_set:
        for d in range(len(DIRS)):
            heap.append((0, s, d))
            best[(s, d)] = 0
    heapq.heapify(heap)
    came = {}

    while heap:
        cost, cell, d = heapq.heappop(heap)
        if best.get((cell, d), None) != cost:
            continue
        if cell in target_set and cell not in start_set:
            path = [cell]
            state = (cell, d)
            while state in came:
                state = came[state]
                path.append(state[0])
            path.reverse()
            return path

        turning_here = cell in no_turn
        for nd, (di, dj) in enumerate(DIRS):
            if turning_here and nd != d and cell not in start_set:
                continue                      # would leave a T on another net
            nxt = (cell[0] + di, cell[1] + dj)
            if not (i0 <= nxt[0] <= i1 and j0 <= nxt[1] <= j1):
                continue
            if nxt in blocked and nxt not in target_set:
                continue
            if nxt in no_turn and nxt in target_set:
                continue                      # our net would end on theirs
            if frozenset((cell, nxt)) in blocked_edges:
                continue
            step = 1 + (TURN_PENALTY if nd != d else 0)
            # Sharing a line with another net is legal but hard to read: two
            # rails on the same row, separated only by a gap, look like one
            # wire. Nudge the route onto its own lane when one is free.
            if (dj == 0 and nxt[1] in busy_rows) or (di == 0 and nxt[0] in busy_cols):
                step += LANE_PENALTY
            ncost = cost + step
            if ncost < best.get((nxt, nd), float("inf")):
                best[(nxt, nd)] = ncost
                came[(nxt, nd)] = (cell, d)
                heapq.heappush(heap, (ncost, nxt, nd))

    return None


def _cells_to_segments(cells):
    """Compress a lattice path into the fewest axis-aligned segments."""
    if not cells or len(cells) < 2:
        return []
    pts = [cells[0]]
    for prev, cur, nxt in zip(cells, cells[1:], cells[2:]):
        if (cur[0] - prev[0], cur[1] - prev[1]) != (nxt[0] - cur[0], nxt[1] - cur[1]):
            pts.append(cur)
    pts.append(cells[-1])
    return _path_segments([point_of(c) for c in pts])


def route_net(stub_points, boxes, blocked_cells=None, foreign=()):
    """
    Wire one net together, given where each of its pins has already escaped to.

    foreign is every segment belonging to another net — including all of their
    escape stubs, which is why those are planned for the whole sheet before any
    routing starts. Returns axis-aligned segments that touch no body and no
    other net.
    """
    if len(stub_points) < 2:
        return []

    if blocked_cells is None:
        blocked_cells = _blocked_cells(boxes)
    foreign_nodes, foreign_edges, no_turn, busy_rows, busy_cols = _occupancy(foreign)
    blocked = blocked_cells | foreign_nodes

    segments = []
    stubs = [cell_of(p) for p in stub_points]

    reached = {stubs[0]}
    for stub in stubs[1:]:
        if stub in reached:
            continue
        cells = list(reached) + [stub]
        i_vals = [c[0] for c in cells]
        j_vals = [c[1] for c in cells]
        bounds = (max(min(i_vals) - SEARCH_PAD, 2), max(min(j_vals) - SEARCH_PAD, 2),
                  min(max(i_vals) + SEARCH_PAD, idx(SHEET_W) - 2),
                  min(max(j_vals) + SEARCH_PAD, idx(SHEET_H) - 2))

        path = _maze_route([stub], reached, blocked, foreign_edges, no_turn,
                           bounds, busy_rows, busy_cols)
        if path is None:      # retry with the whole sheet before giving up
            path = _maze_route([stub], reached, blocked, foreign_edges, no_turn,
                               (2, 2, idx(SHEET_W) - 2, idx(SHEET_H) - 2),
                               busy_rows, busy_cols)
        if path is None:
            # Nothing legal exists — draw it anyway so the net is visible and
            # let the verifier report it rather than dropping it silently.
            segments += _path_segments([point_of(stub), point_of(next(iter(reached)))])
            reached.add(stub)
            continue

        segments += _cells_to_segments(path)
        reached.update(path)

    return segments


def plan_escapes(pin_entries, blocked_cells):
    """
    Walk every pin out to open lattice, for the whole sheet, before any routing.

    Doing this per net as it was routed left the first net free to lay a wire
    exactly where a later net's pin would escape to — the two then shared a
    point, which is a short, and no amount of obstacle avoidance during the
    later net's own routing could undo it. Reserving all the escapes up front
    removes that ordering dependency.

    pin_entries is [(net, point, escape_direction)]. Returns
    ({net: [stub points]}, {net: [stub segments]}).
    """
    stubs, segments = defaultdict(list), defaultdict(list)
    claimed = set()

    for net, pt, escape in pin_entries:
        stub_pt, legs = _escape(pt, escape, blocked_cells | claimed)
        stubs[net].append(stub_pt)
        segments[net].extend(legs)
        for seg in legs:
            claimed.update(_walk_cells(seg))
        claimed.add(cell_of(stub_pt))

    return dict(stubs), dict(segments)


def _escape(pin_pt, escape, blocked, max_steps=10):
    """
    Step out of a pin along its own direction until the end cell is clear.

    Leaving along the pin keeps the wire off the symbol body; walking until the
    cell is free keeps it off whatever else is already there.
    """
    cur = cell_of(pin_pt)
    di, dj = int(escape[0]), int(escape[1])
    best = None
    for step in range(1, max_steps + 1):
        nxt = (cur[0] + di * step, cur[1] + dj * step)
        if nxt in blocked:
            continue
        best = nxt
        if step >= 2:                      # 2 cells clears the body margin
            break
    if best is None:
        best = (cur[0] + di * 2, cur[1] + dj * 2)
    return point_of(best), _path_segments([key(pin_pt), point_of(best)])


# ═══════════════════════════════════════════════════════════════════
#  Stage 4 — junctions
# ═══════════════════════════════════════════════════════════════════

def compute_junctions(net_segments: dict) -> tuple:
    """
    Work out where KiCad needs a junction dot, and where two nets touch.

    KiCad's rules: wires crossing without a junction are *not* connected, three
    or more wire ends meeting need a dot, and a wire end landing in the middle of
    another wire needs a dot too. Everything the router emits obeys the first
    rule, so this only has to place dots — and shout if two different nets ever
    meet, which would be a short.

    Returns (junction points, list of (net_a, net_b, point) shorts).
    """
    endpoints = defaultdict(set)        # point → {net names}
    ends_here = defaultdict(int)        # point → number of wire ends
    for net, segs in net_segments.items():
        for seg in segs:
            for pt in seg:
                endpoints[key(pt)].add(net)
                ends_here[key(pt)] += 1

    junctions, shorts = set(), []

    for pt, nets in endpoints.items():
        if len(nets) > 1:
            a, b = sorted(nets)[:2]
            shorts.append((a, b, pt))
            continue
        net = next(iter(nets))
        touches_interior = any(
            is_interior(pt, seg) for seg in net_segments[net]
        )
        if ends_here[pt] >= 3 or touches_interior:
            junctions.add(pt)

    # A wire end of net A sitting inside a wire of net B is also a short.
    for pt, nets in endpoints.items():
        owner = next(iter(nets)) if len(nets) == 1 else None
        for net, segs in net_segments.items():
            if net == owner:
                continue
            if any(is_interior(pt, seg) for seg in segs):
                shorts.append((owner or "?", net, pt))
                break

    # Two nets running along the same line for any distance are shorted even
    # though no endpoint of either lands on the other.
    items = list(net_segments.items())
    for i, (net_a, segs_a) in enumerate(items):
        for net_b, segs_b in items[i + 1:]:
            for sa in segs_a:
                for sb in segs_b:
                    ov = _overlap_point(sa, sb)
                    if ov is not None:
                        shorts.append((net_a, net_b, ov))

    return junctions, shorts


def _overlap_point(sa, sb, eps=1e-6):
    """Midpoint of the shared stretch of two collinear segments, else None."""
    (ax1, ay1), (ax2, ay2) = sa
    (bx1, by1), (bx2, by2) = sb

    if abs(ax1 - ax2) < eps and abs(bx1 - bx2) < eps:            # both vertical
        if abs(ax1 - bx1) > eps:
            return None
        lo = max(min(ay1, ay2), min(by1, by2))
        hi = min(max(ay1, ay2), max(by1, by2))
        return (q(ax1), q((lo + hi) / 2)) if hi - lo > eps else None

    if abs(ay1 - ay2) < eps and abs(by1 - by2) < eps:            # both horizontal
        if abs(ay1 - by1) > eps:
            return None
        lo = max(min(ax1, ax2), min(bx1, bx2))
        hi = min(max(ax1, ax2), max(bx1, bx2))
        return (q((lo + hi) / 2), q(ay1)) if hi - lo > eps else None

    return None


# ═══════════════════════════════════════════════════════════════════
#  Stage 5 — verification
# ═══════════════════════════════════════════════════════════════════

class _UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, a):
        self.parent.setdefault(a, a)
        while self.parent[a] != a:
            self.parent[a] = self.parent[self.parent[a]]
            a = self.parent[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def verify_connectivity(net_segments, junctions, pin_points, net_to_pins,
                        ground_points=(), ground_nets=()):
    """
    Re-derive connectivity the way KiCad does and check it against the netlist.

    Everything lying on one wire segment is one node, but only pins, junctions
    and the segment's own endpoints count as contact points — a foreign wire end
    resting on a segment does not connect without a dot, which is exactly the
    trap the old emitter fell into.
    """
    uf = _UnionFind()
    all_segments = [(net, seg) for net, segs in net_segments.items() for seg in segs]

    pin_keys = {key(pt): (ref, pin) for (ref, pin), pt in pin_points.items()}
    junction_keys = {key(p) for p in junctions}

    for _net, seg in all_segments:
        a, b = key(seg[0]), key(seg[1])
        contacts = {a, b}
        for pk in pin_keys:
            if on_segment(pk, seg):
                contacts.add(pk)
        for jk in junction_keys:
            if on_segment(jk, seg):
                contacts.add(jk)
        contacts = list(contacts)
        for c in contacts[1:]:
            uf.union(contacts[0], c)

    # Every GND symbol is the same node wherever it sits on the sheet.
    ground_points = [key(p) for p in ground_points]
    for gp in ground_points[1:]:
        uf.union(ground_points[0], gp)

    results, ok = [], True
    for net, pins in sorted(net_to_pins.items()):
        if net in ground_nets and pins:
            # A ground net does not need a second device pin: the power symbol
            # on each of its pins is the other end, and KiCad ties every GND
            # symbol on the sheet together by name.
            results.append((net, "OK", f"{len(pins)} pin(s) to GND"))
            continue
        if len(pins) < 2:
            results.append((net, "DANGLING", f"{len(pins)} pin"))
            continue
        roots = set()
        missing = []
        for ref, pin in pins:
            pt = pin_points.get((ref, pin))
            if pt is None:
                missing.append(f"{ref}.{pin}")
                continue
            roots.add(uf.find(key(pt)))
        if missing:
            ok = False
            results.append((net, "FAIL", f"no placed pin for {', '.join(missing)}"))
        elif len(roots) == 1:
            results.append((net, "OK", f"{len(pins)} pins on one node"))
        else:
            ok = False
            results.append((net, "FAIL", f"{len(pins)} pins split across {len(roots)} nodes"))

    return ok, results


def verify_with_kicad_cli(sch_path, net_to_pins):
    """
    Cross-check against KiCad's own connectivity engine via kicad-cli.

    This is the ground truth: it runs the same code eeschema uses. Returns
    (ok, message) or (None, reason) when kicad-cli is unavailable.
    """
    cli = shutil.which("kicad-cli")
    if cli is None:
        for base in (r"C:\Program Files\KiCad", r"C:\Program Files (x86)\KiCad"):
            if not os.path.isdir(base):
                continue
            for ver in sorted(os.listdir(base), reverse=True):
                cand = os.path.join(base, ver, "bin", "kicad-cli.exe")
                if os.path.exists(cand):
                    cli = cand
                    break
            if cli:
                break
    if cli is None:
        return None, "kicad-cli not found — skipped ground-truth check"

    xml_path = os.path.splitext(sch_path)[0] + "_netlist.xml"
    proc = subprocess.run(
        [cli, "sch", "export", "netlist", "--format", "kicadxml",
         "-o", xml_path, sch_path],
        capture_output=True, text=True,
    )
    if proc.returncode != 0 or not os.path.exists(xml_path):
        return None, f"kicad-cli failed: {(proc.stderr or proc.stdout).strip()[:200]}"

    import xml.etree.ElementTree as ET
    root = ET.parse(xml_path).getroot()
    kicad_nets = []
    for net in root.find("nets") or []:
        nodes = {(n.get("ref"), n.get("pin")) for n in net.findall("node")}
        kicad_nets.append((net.get("name"), nodes))

    problems = []
    landed = {}
    for net, pins in sorted(net_to_pins.items()):
        if len(pins) < 2:
            continue
        want = {(r, p) for r, p in pins if not str(r).startswith("#")}
        if not want:
            continue
        hit = [name for name, nodes in kicad_nets if want <= nodes]
        if not hit:
            spread = [name for name, nodes in kicad_nets if nodes & want]
            problems.append(f"{net} split across {spread or 'nothing'}")
            continue
        # Two of our nets landing on one KiCad net means they got shorted.
        if hit[0] in landed:
            problems.append(f"{net} shorted to {landed[hit[0]]}")
        else:
            landed[hit[0]] = net

    if problems:
        return False, "; ".join(problems)
    return True, f"kicad-cli confirms all {len(landed)} nets, none shorted"


# ═══════════════════════════════════════════════════════════════════
#  Build
# ═══════════════════════════════════════════════════════════════════

def build_schematic(graph: dict, output_path: str, title="SINA circuit",
                    add_labels=False, verbose=True):
    """Run the whole pipeline and write the .kicad_sch. Returns a report dict."""
    if ksa is None:
        raise RuntimeError(
            "kicad_sch_api is required: pip install kicad-sch-api"
        )

    devices, net_to_pins, detected_grounds = normalize_graph(graph, verbose=verbose)
    if not devices:
        raise ValueError("graph contains no devices")

    plan_placement(devices)

    sch = ksa.create_schematic(title)

    # ── place, then take the position KiCad actually stored (it grid-snaps) ──
    placed_pins = {}
    pin_points = {}
    centers = {}
    for d in devices:
        comp = sch.components.add(
            d["lib_id"],
            reference=d["ref"],
            value=d["value"] or d["lib_id"].split(":")[-1],
            position=(d["x"], d["y"]),
            rotation=d["rotation"],
        )
        d["x"], d["y"] = q(comp.position.x), q(comp.position.y)
        centers[d["ref"]] = (d["x"], d["y"])

        pts = []
        for num, _local in load_symbol_pins(d["lib_id"]):
            pt = pin_position(d["lib_id"], num, d["x"], d["y"], d["rotation"])
            pts.append((num, pt))
            pin_points[(d["ref"], num)] = pt
        placed_pins[d["ref"]] = pts

    # ── ground gets a power symbol per pin instead of a routed net ──
    #
    # Ground is normally the biggest net on the sheet; dragging a wire from
    # every pin to every other one buries the schematic and gives the router its
    # hardest job for no benefit. Power symbols of the same name are connected
    # globally by KiCad, so a short stub to a GND symbol does the whole job.
    ground_nets = {n for n in net_to_pins if is_ground_net(n)} | detected_grounds
    ground_points = []
    gnd_stubs = []
    gnd_counter = 0
    gnd_local = load_symbol_pins("power:GND")[0]        # pin id and local offset
    gnd_rotation = {(0.0, 1.0): 0, (0.0, -1.0): 180, (1.0, 0.0): 270, (-1.0, 0.0): 90}

    for net in sorted(ground_nets):
        for ref, pin in list(net_to_pins[net]):
            pt = pin_points.get((ref, pin))
            if pt is None:
                continue
            gnd_counter += 1
            gref = f"#PWR{gnd_counter:02d}"
            ex, ey = escape_direction(pt, centers[ref])
            gpt = (q(pt[0] + ex * STUB), q(pt[1] + ey * STUB))
            # The GND symbol's pin sits at its own origin, so placing it at gpt
            # puts the pin exactly on the stub end whatever the rotation is.
            sch.components.add("power:GND", reference=gref, value="GND",
                               position=gpt, rotation=gnd_rotation.get((ex, ey), 0))
            centers[gref] = gpt
            pin_points[(gref, gnd_local[0])] = gpt
            placed_pins[gref] = [(gnd_local[0], gpt)]
            gnd_stubs.append((key(pt), gpt))
            ground_points.append(gpt)

    # ── route, biggest nets first so the crowded ones get the free lanes ──
    net_segments = {}
    if gnd_stubs:
        net_segments["<gnd>"] = list(gnd_stubs)
    boxes = body_boxes(placed_pins, centers)
    blocked_cells = _blocked_cells(boxes)

    order = sorted(((n, p) for n, p in net_to_pins.items() if n not in ground_nets),
                   key=lambda kv: (-len(kv[1]), str(kv[0])))
    order = [(n, p) for n, p in order if len(p) >= 2]

    # Reserve every pin's escape before routing anything — see plan_escapes.
    reserved = set()
    for seg in gnd_stubs:
        reserved.update(_walk_cells(seg))
    pin_entries = []
    for net, pins in order:
        for ref, pin in pins:
            pt = pin_points.get((ref, pin))
            if pt is None:
                continue
            pin_entries.append((net, pt, escape_direction(pt, centers.get(ref, pt))))
    net_stubs, escape_segments = plan_escapes(pin_entries, blocked_cells | reserved)

    for net, segs in escape_segments.items():
        net_segments.setdefault(net, []).extend(segs)

    for net, _pins in order:
        others = [seg for other, segs in net_segments.items() if other != net
                  for seg in segs]
        segs = route_net(net_stubs.get(net, []), boxes, blocked_cells, foreign=others)
        if segs:
            net_segments.setdefault(net, []).extend(segs)

    junctions, shorts = compute_junctions(net_segments)

    for net, segs in net_segments.items():
        for (x1, y1), (x2, y2) in segs:
            sch.wires.add(start=(x1, y1), end=(x2, y2))
    for jx, jy in sorted(junctions):
        sch.junctions.add(position=(jx, jy))

    if add_labels:
        for net, segs in net_segments.items():
            if not segs:
                continue
            (x1, y1), (x2, y2) = segs[-1]
            sch.add_label(text=net, position=(q((x1 + x2) / 2), q((y1 + y2) / 2)))

    ok, results = verify_connectivity(net_segments, junctions, pin_points,
                                      net_to_pins, ground_points, ground_nets)

    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    sch.save(output_path)

    return {
        "output": output_path,
        "devices": devices,
        "net_to_pins": net_to_pins,
        "segments": sum(len(s) for s in net_segments.values()),
        "junctions": len(junctions),
        "shorts": shorts,
        "ok": ok and not shorts,
        "results": results,
    }


# ═══════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="SINA graph JSON → KiCad schematic with connected nets"
    )
    parser.add_argument("--graph", required=True, help="SINA graph JSON")
    parser.add_argument("--output", default=None,
                        help="Output .kicad_sch (default: alongside the graph)")
    parser.add_argument("--title", default=None, help="Schematic title")
    parser.add_argument("--labels", action="store_true",
                        help="Also drop a net label on every routed net")
    parser.add_argument("--verify", action="store_true",
                        help="Cross-check the result with kicad-cli's netlist export")
    args = parser.parse_args()

    if not os.path.exists(args.graph):
        print(f"Error: graph file '{args.graph}' not found.")
        sys.exit(1)

    with open(args.graph) as f:
        graph = json.load(f)

    base = os.path.splitext(os.path.basename(args.graph))[0]
    output = args.output or os.path.join(os.path.dirname(os.path.abspath(args.graph)),
                                         base + ".kicad_sch")
    title = args.title or graph.get("source_file", base)

    print(f"Reading {args.graph}")
    report = build_schematic(graph, output, title=title,
                             add_labels=args.labels, verbose=True)

    print(f"\n  {len(report['devices'])} symbols, "
          f"{report['segments']} wire segments, {report['junctions']} junctions")

    print("\n  Net connectivity")
    for net, status, detail in report["results"]:
        mark = {"OK": "  ok  ", "FAIL": " FAIL ", "DANGLING": " open "}[status]
        print(f"    [{mark}] {net:<10} {detail}")

    for a, b, pt in report["shorts"]:
        print(f"    [ SHORT] {a} touches {b} at {pt}")

    print(f"\nWrote {output}")

    if args.verify:
        ok, msg = verify_with_kicad_cli(output, report["net_to_pins"])
        prefix = {True: "  ok  ", False: " FAIL ", None: " skip "}[ok]
        print(f"  [{prefix}] {msg}")
        if ok is False:
            sys.exit(2)

    if not report["ok"]:
        sys.exit(2)


if __name__ == "__main__":
    main()
