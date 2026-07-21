import json
import re
from pathlib import Path
from collections import defaultdict

ROOT = Path("/workspace/generation")
PARSED = ROOT / "data/parsed"
OUT = ROOT / "data/smart_layouts"
PRIORS_FILE = Path(
    "/workspace/layout_training/data/layout_priors.json"
)

OUT.mkdir(parents=True, exist_ok=True)

def load_json(p):
    with open(p) as f:
        return json.load(f)

def ctype(c):
    return str(c.get("type", "")).upper()

def pins(c):
    return c.get("pins", {})

def find_cross_coupled(mos):
    pairs = []

    for i, a in enumerate(mos):
        pa = pins(a)

        for b in mos[i+1:]:
            pb = pins(b)

            ga = pa.get("G")
            gb = pb.get("G")
            da = pa.get("D")
            db = pb.get("D")

            if ga and gb and da and db:
                if ga == db and gb == da:
                    pairs.append((a["name"], b["name"]))

    return pairs

def main():
    files = sorted(PARSED.glob("*_parsed.json"))

    if not files:
        raise SystemExit(
            "No parsed circuits. Run generation/src/01_parse_spice.py"
        )

    priors = {}
    if PRIORS_FILE.exists():
        priors = load_json(PRIORS_FILE).get("priors", {})

    for fp in files:
        data = load_json(fp)

        comps = data.get("components", [])

        mos = [
            c for c in comps
            if ctype(c) in ("NMOS", "PMOS", "MOSFET")
        ]

        passives = [
            c for c in comps
            if ctype(c) in (
                "RESISTOR",
                "CAPACITOR",
                "INDUCTOR"
            )
        ]

        sources = [
            c for c in comps
            if "SOURCE" in ctype(c)
        ]

        cross = find_cross_coupled(mos)

        placed = {}
        used = set()

        # --------------------------------------------------
        # Core MOS placement
        # PMOS upper, NMOS lower
        # --------------------------------------------------

        pmos = [c for c in mos if ctype(c) == "PMOS"]
        nmos = [c for c in mos if ctype(c) == "NMOS"]

        def place_row(arr, y):
            if not arr:
                return

            spacing = 6.0
            start = -(len(arr)-1)*spacing/2

            for i, c in enumerate(arr):
                placed[c["name"]] = {
                    "x": start + i*spacing,
                    "y": y,
                    "orientation": "vertical",
                    "region": "core"
                }

        place_row(pmos, 2.5)
        place_row(nmos, 8.0)

        # Cross-coupled pair should be mirrored
        for a, b in cross:
            if a in placed and b in placed:
                y = (
                    placed[a]["y"] +
                    placed[b]["y"]
                ) / 2

                placed[a]["x"] = -3.5
                placed[b]["x"] = 3.5
                placed[a]["y"] = y
                placed[b]["y"] = y

        # --------------------------------------------------
        # Passive loads:
        # inductors/resistors below core,
        # capacitors near right/output side
        # --------------------------------------------------

        inds = [
            c for c in passives
            if ctype(c) == "INDUCTOR"
        ]

        resistors = [
            c for c in passives
            if ctype(c) == "RESISTOR"
        ]

        caps = [
            c for c in passives
            if ctype(c) == "CAPACITOR"
        ]

        lower = inds + resistors

        if lower:
            spacing = 7.0
            start = -(len(lower)-1)*spacing/2

            for i, c in enumerate(lower):
                placed[c["name"]] = {
                    "x": start + i*spacing,
                    "y": 13.5,
                    "orientation": "vertical",
                    "region": "load"
                }

        for i, c in enumerate(caps):
            placed[c["name"]] = {
                "x": 8.0 + i*3,
                "y": 9.5 + i*2,
                "orientation": "vertical",
                "region": "load"
            }

        # --------------------------------------------------
        # Sources left side
        # --------------------------------------------------

        for i, c in enumerate(sources):
            placed[c["name"]] = {
                "x": -12.0,
                "y": 1.5 + i*3.0,
                "orientation": "vertical",
                "region": "bias"
            }

        # --------------------------------------------------
        # Anything not placed
        # --------------------------------------------------

        rest = [
            c for c in comps
            if c["name"] not in placed
        ]

        for i, c in enumerate(rest):
            placed[c["name"]] = {
                "x": 11.0,
                "y": 3.0 + i*3.0,
                "orientation": "vertical",
                "region": "other"
            }

        result = {
            "source_parsed": str(fp),
            "placement_method":
                "topology-aware analog schematic placement",
            "cross_coupled_pairs": cross,
            "components": placed
        }

        name = fp.name.replace(
            "_parsed.json",
            "_smart_layout.json"
        )

        out = OUT / name

        with open(out, "w") as f:
            json.dump(result, f, indent=2)

        print("="*72)
        print("SMART LAYOUT:", fp.name)
        print("="*72)

        for n, p in placed.items():
            print(
                f"{n:15s} "
                f"x={p['x']:6.1f} "
                f"y={p['y']:6.1f} "
                f"{p['region']}"
            )

        print("Cross-coupled:", cross)
        print("Saved:", out)

if __name__ == "__main__":
    main()