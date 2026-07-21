import json
import math
from pathlib import Path
from collections import defaultdict, Counter

ROOT = Path("/workspace/layout_training")
EXTRACTED = ROOT / "data/layout_extracted"
OUT = ROOT / "data/layout_priors.json"

def valid_box(c):
    if not c.get("visible", False):
        return False
    if float(c.get("confidence", 0) or 0) < 0.75:
        return False

    b = c.get("bbox_normalized")
    if not isinstance(b, list) or len(b) != 4:
        return False

    x1, y1, x2, y2 = b
    if not all(isinstance(v, (int, float)) for v in b):
        return False
    if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
        return False

    # Reject absurdly huge boxes
    if (x2-x1) > 0.55 or (y2-y1) > 0.55:
        return False

    return True

def norm_type(t):
    t = str(t or "").upper()

    if "MOS" in t or "FET" in t:
        return "MOSFET"
    if "BJT" in t or "NPN" in t or "PNP" in t:
        return "BJT"
    if "RES" in t:
        return "RESISTOR"
    if "CAP" in t:
        return "CAPACITOR"
    if "IND" in t:
        return "INDUCTOR"
    if "VOLT" in t:
        return "VOLTAGE_SOURCE"
    if "CURR" in t:
        return "CURRENT_SOURCE"
    if "DIODE" in t:
        return "DIODE"

    return t or "UNKNOWN"

def main():
    files = sorted(EXTRACTED.glob("sample_*.json"))

    stats = defaultdict(lambda: {
        "xs": [], "ys": [], "ws": [], "hs": []
    })

    accepted = 0
    rejected = 0

    for f in files:
        try:
            data = json.load(open(f))
            comps = data.get(
                "visual_extraction", {}
            ).get("components", [])

            for c in comps:
                if not valid_box(c):
                    rejected += 1
                    continue

                b = c["bbox_normalized"]
                x1, y1, x2, y2 = b

                typ = norm_type(
                    c.get("type") or
                    c.get("component_type")
                )

                stats[typ]["xs"].append((x1+x2)/2)
                stats[typ]["ys"].append((y1+y2)/2)
                stats[typ]["ws"].append(x2-x1)
                stats[typ]["hs"].append(y2-y1)

                accepted += 1

        except Exception as e:
            print("Skip", f.name, e)

    def mean(v, default):
        return sum(v)/len(v) if v else default

    priors = {}

    defaults = {
        "MOSFET": (0.50, 0.50, 0.12, 0.18),
        "BJT": (0.50, 0.50, 0.12, 0.18),
        "RESISTOR": (0.50, 0.50, 0.08, 0.18),
        "CAPACITOR": (0.65, 0.55, 0.10, 0.14),
        "INDUCTOR": (0.50, 0.70, 0.10, 0.18),
        "VOLTAGE_SOURCE": (0.18, 0.50, 0.12, 0.18),
        "CURRENT_SOURCE": (0.50, 0.35, 0.12, 0.18),
        "DIODE": (0.50, 0.50, 0.12, 0.15)
    }

    for typ, d in defaults.items():
        s = stats[typ]

        priors[typ] = {
            "mean_x": mean(s["xs"], d[0]),
            "mean_y": mean(s["ys"], d[1]),
            "mean_w": mean(s["ws"], d[2]),
            "mean_h": mean(s["hs"], d[3]),
            "samples": len(s["xs"])
        }

    result = {
        "description":
            "Masala-CHAI-derived/fallback schematic layout priors",
        "accepted_visual_labels": accepted,
        "rejected_visual_labels": rejected,
        "priors": priors,

        "analog_rules": {
            "power_top": True,
            "ground_bottom": True,
            "sources_left": True,
            "loads_near_output": True,
            "symmetric_pairs_mirrored": True,
            "cross_coupled_pairs_same_level": True,
            "signal_flow_left_to_right": True
        }
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)

    with open(OUT, "w") as f:
        json.dump(result, f, indent=2)

    print("="*72)
    print("LAYOUT PRIORS BUILT")
    print("="*72)
    print("Extracted files :", len(files))
    print("Accepted labels :", accepted)
    print("Rejected labels :", rejected)
    print("Saved           :", OUT)

    for k, v in priors.items():
        print(
            f"{k:18s} samples={v['samples']:4d} "
            f"mean=({v['mean_x']:.2f},{v['mean_y']:.2f})"
        )

if __name__ == "__main__":
    main()