import json
import math
from pathlib import Path

import cairosvg


ROOT = Path("/workspace/generation")

PARSED_DIR = ROOT / "data" / "parsed"
LAYOUT_DIR = ROOT / "data" / "smart_layouts"
OUTPUT_DIR = ROOT / "output"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# SVG HELPERS
# ============================================================

def line(x1, y1, x2, y2, w=2):
    return (
        f'<line x1="{x1}" y1="{y1}" '
        f'x2="{x2}" y2="{y2}" '
        f'stroke="black" stroke-width="{w}" '
        f'stroke-linecap="round"/>'
    )


def text(x, y, s, size=14, weight="normal", anchor="start"):
    s = str(s).replace("&", "&amp;").replace("<", "&lt;")
    return (
        f'<text x="{x}" y="{y}" '
        f'font-family="Arial, sans-serif" '
        f'font-size="{size}" '
        f'font-weight="{weight}" '
        f'text-anchor="{anchor}">{s}</text>'
    )


def circle(x, y, r=4, fill="black"):
    return (
        f'<circle cx="{x}" cy="{y}" r="{r}" '
        f'fill="{fill}"/>'
    )


def ground(x, y):
    s = []
    s.append(line(x, y, x, y + 10))
    s.append(line(x - 12, y + 10, x + 12, y + 10))
    s.append(line(x - 8, y + 15, x + 8, y + 15))
    s.append(line(x - 4, y + 20, x + 4, y + 20))
    return "\n".join(s)


# ============================================================
# COMPONENT SYMBOLS
# ============================================================

def draw_mos(name, typ, x, y):
    s = []

    # vertical channel
    s.append(line(x, y - 28, x, y + 28, 3))

    # gate
    gx = x - 25
    s.append(line(gx, y - 22, gx, y + 22, 2))
    s.append(line(gx - 20, y, gx, y, 2))

    # drain/source leads
    s.append(line(x, y - 28, x, y - 48))
    s.append(line(x, y + 28, x, y + 48))

    # body indication
    s.append(line(x + 14, y - 12, x + 14, y + 12))

    if typ == "PMOS":
        s.append(
            f'<circle cx="{gx + 7}" cy="{y}" '
            f'r="5" fill="white" stroke="black" '
            f'stroke-width="2"/>'
        )

    s.append(text(x + 28, y - 5, name, 15, "bold"))
    s.append(text(x + 28, y + 14, typ, 12))

    anchors = {
        "D": (x, y - 48),
        "S": (x, y + 48),
        "G": (gx - 20, y),
        "B": (x + 14, y)
    }

    return "\n".join(s), anchors


def draw_inductor(name, x, y):
    s = []

    s.append(line(x, y - 45, x, y - 30))

    yy = y - 30

    for i in range(4):
        y1 = yy + i * 15
        s.append(
            f'<path d="M {x} {y1} '
            f'C {x+18} {y1+3}, '
            f'{x+18} {y1+12}, '
            f'{x} {y1+15}" '
            f'fill="none" stroke="black" '
            f'stroke-width="2"/>'
        )

    s.append(line(x, y + 30, x, y + 45))

    s.append(text(x + 25, y, name, 15, "bold"))

    return "\n".join(s), {
        "1": (x, y - 45),
        "2": (x, y + 45)
    }


def draw_capacitor(name, x, y):
    s = []

    s.append(line(x, y - 45, x, y - 10))
    s.append(line(x - 18, y - 10, x + 18, y - 10, 3))
    s.append(line(x - 18, y + 10, x + 18, y + 10, 3))
    s.append(line(x, y + 10, x, y + 45))

    s.append(text(x + 28, y + 5, name, 15, "bold"))

    return "\n".join(s), {
        "1": (x, y - 45),
        "2": (x, y + 45)
    }


def draw_resistor(name, x, y):
    s = []

    s.append(line(x, y - 45, x, y - 30))

    pts = [
        (x, y - 30),
        (x - 10, y - 22),
        (x + 10, y - 14),
        (x - 10, y - 6),
        (x + 10, y + 2),
        (x - 10, y + 10),
        (x + 10, y + 18),
        (x, y + 30)
    ]

    p = " ".join(f"{a},{b}" for a, b in pts)

    s.append(
        f'<polyline points="{p}" fill="none" '
        f'stroke="black" stroke-width="2"/>'
    )

    s.append(line(x, y + 30, x, y + 45))
    s.append(text(x + 25, y, name, 15, "bold"))

    return "\n".join(s), {
        "1": (x, y - 45),
        "2": (x, y + 45)
    }


def draw_voltage(name, x, y):
    s = []

    s.append(line(x, y - 45, x, y - 25))

    s.append(
        f'<circle cx="{x}" cy="{y}" r="25" '
        f'fill="white" stroke="black" stroke-width="2"/>'
    )

    s.append(text(x, y - 5, "+", 18, "bold", "middle"))
    s.append(text(x, y + 15, "−", 18, "bold", "middle"))

    s.append(line(x, y + 25, x, y + 45))

    s.append(text(x + 35, y + 5, name, 13, "bold"))

    return "\n".join(s), {
        "+": (x, y - 45),
        "-": (x, y + 45)
    }


def draw_current(name, x, y):
    s = []

    s.append(line(x, y - 45, x, y - 25))

    s.append(
        f'<circle cx="{x}" cy="{y}" r="25" '
        f'fill="white" stroke="black" stroke-width="2"/>'
    )

    s.append(line(x, y - 12, x, y + 12, 2))

    s.append(
        f'<polygon points="'
        f'{x-6},{y+5} '
        f'{x+6},{y+5} '
        f'{x},{y+15}" fill="black"/>'
    )

    s.append(line(x, y + 25, x, y + 45))

    s.append(text(x + 35, y + 5, name, 13, "bold"))

    return "\n".join(s), {
        "+": (x, y - 45),
        "-": (x, y + 45)
    }


# ============================================================
# ROUTING
# ============================================================

def orthogonal_route(points, net, offset):
    if len(points) < 2:
        return ""

    s = []

    # Dedicated horizontal bus per net.
    avg_y = sum(p[1] for p in points) / len(points)

    bus_y = avg_y + offset

    min_x = min(p[0] for p in points)
    max_x = max(p[0] for p in points)

    s.append(line(min_x, bus_y, max_x, bus_y, 2))

    for x, y in points:
        s.append(line(x, y, x, bus_y, 2))
        s.append(circle(x, bus_y, 3))

    if net not in ("0", "VSS"):
        s.append(
            text(
                min_x + 5,
                bus_y - 6,
                net,
                11,
                "bold"
            )
        )

    return "\n".join(s)


# ============================================================
# MAIN
# ============================================================

def main():

    layouts = sorted(
        LAYOUT_DIR.glob("*_smart_layout.json")
    )

    if not layouts:
        raise SystemExit(
            "No smart layout JSON found. Run 06_smart_layout.py first."
        )

    for layout_file in layouts:

        base = layout_file.name.replace(
            "_smart_layout.json",
            ""
        )

        parsed_file = (
            PARSED_DIR /
            f"{base}_parsed.json"
        )

        if not parsed_file.exists():
            print("Missing:", parsed_file)
            continue

        with open(layout_file) as f:
            layout = json.load(f)

        with open(parsed_file) as f:
            parsed = json.load(f)

        components = parsed["components"]
        placements = layout["components"]

        # Canvas
        WIDTH = 1500
        HEIGHT = 1000

        SCALE_X = 55
        SCALE_Y = 55

        ORIGIN_X = 760
        ORIGIN_Y = 100

        svg = [
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{WIDTH}" height="{HEIGHT}" '
            f'viewBox="0 0 {WIDTH} {HEIGHT}">',
            '<rect width="100%" height="100%" fill="white"/>',

            text(
                WIDTH / 2,
                35,
                "Schematic reconstructed from generated SPICE",
                22,
                "bold",
                "middle"
            )
        ]

        anchors = {}

        # ----------------------------------------------------
        # DRAW COMPONENTS
        # ----------------------------------------------------

        for c in components:

            name = c["name"]
            typ = c["type"]

            if name not in placements:
                continue

            p = placements[name]

            x = ORIGIN_X + p["x"] * SCALE_X
            y = ORIGIN_Y + p["y"] * SCALE_Y

            if typ in ("NMOS", "PMOS"):

                drawing, a = draw_mos(
                    name, typ, x, y
                )

            elif typ == "INDUCTOR":

                drawing, a = draw_inductor(
                    name, x, y
                )

            elif typ == "CAPACITOR":

                drawing, a = draw_capacitor(
                    name, x, y
                )

            elif typ == "RESISTOR":

                drawing, a = draw_resistor(
                    name, x, y
                )

            elif typ == "VOLTAGE_SOURCE":

                drawing, a = draw_voltage(
                    name, x, y
                )

            elif typ == "CURRENT_SOURCE":

                drawing, a = draw_current(
                    name, x, y
                )

            else:
                continue

            svg.append(drawing)

            anchors[name] = a

        # ----------------------------------------------------
        # BUILD NET TERMINALS FROM ACTUAL SPICE PINS
        # ----------------------------------------------------

        nets = {}

        for c in components:

            name = c["name"]

            if name not in anchors:
                continue

            for pin, net in c.get("pins", {}).items():

                if pin not in anchors[name]:
                    continue

                nets.setdefault(
                    net, []
                ).append(
                    anchors[name][pin]
                )

        # ----------------------------------------------------
        # ROUTE
        # ----------------------------------------------------

        offsets = {}

        signal_index = 0

        for net, pts in nets.items():

            if len(pts) < 2:
                continue

            if net == "VDD":
                offset = -90

            elif net in ("VSS", "0"):
                offset = 100

            else:
                # stagger buses to reduce exact overlap
                offset = (
                    -45 +
                    (signal_index % 5) * 24
                )
                signal_index += 1

            svg.append(
                orthogonal_route(
                    pts,
                    net,
                    offset
                )
            )

        # ----------------------------------------------------
        # FOOTER
        # ----------------------------------------------------

        svg.append(
            text(
                30,
                HEIGHT - 25,
                f"Generated from {base}.spice",
                12
            )
        )

        svg.append("</svg>")

        svg_text = "\n".join(svg)

        svg_file = (
            OUTPUT_DIR /
            f"{base}_smart.svg"
        )

        png_file = (
            OUTPUT_DIR /
            f"{base}_smart.png"
        )

        svg_file.write_text(
            svg_text,
            encoding="utf-8"
        )

        cairosvg.svg2png(
            bytestring=svg_text.encode(),
            write_to=str(png_file),
            output_width=WIDTH * 2,
            output_height=HEIGHT * 2
        )

        print("=" * 72)
        print("SMART SCHEMATIC RENDERED")
        print("=" * 72)
        print("Input :", parsed_file)
        print("Layout:", layout_file)
        print("SVG   :", svg_file)
        print("PNG   :", png_file)


if __name__ == "__main__":
    main()