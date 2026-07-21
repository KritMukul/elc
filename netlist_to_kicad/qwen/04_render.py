import json
import math
from pathlib import Path

INPUT = "optimized_layout.json"

OUT_DIR = Path("output")
OUT_DIR.mkdir(exist_ok=True)

SVG_FILE = OUT_DIR / "qwen_analogtobi_0535.svg"
PNG_FILE = OUT_DIR / "qwen_analogtobi_0535.png"

data = json.loads(Path(INPUT).read_text())

components = {
    c["name"]: c
    for c in data["components"]
}

positions = data["positions"]

# --------------------------------------------------
# Canvas
# --------------------------------------------------

W = 1600
H = 1100

# Convert optimized coordinates to screen coordinates
SCALE = 150
CX = 700
CY = 480


def xy(name):
    p = positions[name]

    return (
        CX + p["x"] * SCALE,
        CY - p["y"] * SCALE
    )


svg = []


def add(s):
    svg.append(s)


add(
    f'<svg xmlns="http://www.w3.org/2000/svg" '
    f'width="{W}" height="{H}" '
    f'viewBox="0 0 {W} {H}">'
)

add("""
<rect width="100%" height="100%" fill="white"/>

<style>

text {
    font-family: Arial, Helvetica, sans-serif;
    fill: black;
}

.wire {
    stroke: black;
    stroke-width: 4;
    fill: none;
    stroke-linecap: round;
    stroke-linejoin: round;
}

.symbol {
    stroke: black;
    stroke-width: 4;
    fill: white;
}

.thin {
    stroke: black;
    stroke-width: 3;
    fill: none;
}

.node {
    fill: black;
}

.label {
    font-size: 25px;
    font-weight: bold;
}

.small {
    font-size: 19px;
}

.net {
    font-size: 21px;
    font-weight: bold;
}

</style>
""")


# --------------------------------------------------
# Drawing primitives
# --------------------------------------------------

def line(x1, y1, x2, y2, cls="wire"):

    add(
        f'<line x1="{x1}" y1="{y1}" '
        f'x2="{x2}" y2="{y2}" '
        f'class="{cls}"/>'
    )


def path(points, cls="wire"):

    pts = " ".join(
        f"{x},{y}"
        for x, y in points
    )

    add(
        f'<polyline points="{pts}" '
        f'class="{cls}"/>'
    )


def text(x, y, value, cls="small", anchor="middle"):

    value = (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )

    add(
        f'<text x="{x}" y="{y}" '
        f'text-anchor="{anchor}" '
        f'class="{cls}">{value}</text>'
    )


def dot(x, y, r=7):

    add(
        f'<circle cx="{x}" cy="{y}" '
        f'r="{r}" class="node"/>'
    )


def ground(x, y):

    line(x, y, x, y + 22)

    line(
        x - 25,
        y + 22,
        x + 25,
        y + 22,
        "thin"
    )

    line(
        x - 17,
        y + 31,
        x + 17,
        y + 31,
        "thin"
    )

    line(
        x - 9,
        y + 40,
        x + 9,
        y + 40,
        "thin"
    )


def supply_arrow(x, y, label):

    line(x, y + 35, x, y)

    path(
        [
            (x - 12, y + 12),
            (x, y),
            (x + 12, y + 12)
        ],
        "thin"
    )

    text(
        x,
        y - 14,
        label,
        "net"
    )


# --------------------------------------------------
# MOSFET symbol
#
# Return exact pin coordinates:
# D, G, S, B
# --------------------------------------------------

def mosfet(
    x,
    y,
    name,
    mos_type,
    model=""
):

    # Vertical MOS:
    # drain = bottom
    # source = top
    # gate = left

    source = (x, y - 70)
    drain = (x, y + 70)
    gate = (x - 95, y)
    bulk = (x + 75, y)

    # Channel
    line(
        x,
        y - 38,
        x,
        y + 38,
        "thin"
    )

    # Source/drain leads
    line(
        x,
        y - 70,
        x,
        y - 38,
        "thin"
    )

    line(
        x,
        y + 38,
        x,
        y + 70,
        "thin"
    )

    # Gate
    line(
        x - 58,
        y - 38,
        x - 58,
        y + 38,
        "thin"
    )

    line(
        x - 95,
        y,
        x - 58,
        y,
        "thin"
    )

    # Bulk
    line(
        x + 20,
        y,
        x + 75,
        y,
        "thin"
    )

    # PMOS gate bubble
    if mos_type == "PMOS":

        add(
            f'<circle cx="{x - 49}" '
            f'cy="{y}" r="9" '
            f'class="symbol"/>'
        )

    # Arrow-like bulk marker
    if mos_type == "NMOS":

        path(
            [
                (x + 28, y - 8),
                (x + 40, y),
                (x + 28, y + 8)
            ],
            "thin"
        )

    else:

        path(
            [
                (x + 42, y - 8),
                (x + 30, y),
                (x + 42, y + 8)
            ],
            "thin"
        )

    # Labels
    text(
        x + 100,
        y - 8,
        name,
        "label",
        "start"
    )

    text(
        x + 100,
        y + 22,
        mos_type,
        "small",
        "start"
    )

    return {
        "S": source,
        "D": drain,
        "G": gate,
        "B": bulk
    }


def inductor(
    x,
    y,
    name,
    value
):

    top = (x, y - 75)
    bottom = (x, y + 75)

    line(
        x,
        y - 75,
        x,
        y - 55,
        "thin"
    )

    # Four vertical loops
    start_y = y - 55

    for i in range(4):

        cy = start_y + 27 * i + 14

        add(
            f'<path d="M {x} {cy - 14} '
            f'C {x + 35} {cy - 14}, '
            f'{x + 35} {cy + 14}, '
            f'{x} {cy + 14}" '
            f'class="thin"/>'
        )

    line(
        x,
        y + 53,
        x,
        y + 75,
        "thin"
    )

    text(
        x + 55,
        y - 4,
        name,
        "label",
        "start"
    )

    text(
        x + 55,
        y + 25,
        value,
        "small",
        "start"
    )

    return {
        "1": top,
        "2": bottom
    }


def capacitor(
    x,
    y,
    name,
    value
):

    top = (x, y - 65)
    bottom = (x, y + 65)

    line(
        x,
        y - 65,
        x,
        y - 15,
        "thin"
    )

    line(
        x - 35,
        y - 15,
        x + 35,
        y - 15,
        "thin"
    )

    line(
        x - 35,
        y + 15,
        x + 35,
        y + 15,
        "thin"
    )

    line(
        x,
        y + 15,
        x,
        y + 65,
        "thin"
    )

    text(
        x + 55,
        y - 2,
        name,
        "label",
        "start"
    )

    text(
        x + 55,
        y + 28,
        value,
        "small",
        "start"
    )

    return {
        "1": top,
        "2": bottom
    }


# --------------------------------------------------
# Header
# --------------------------------------------------

text(
    W / 2,
    55,
    "analogtobi_0535",
    "label"
)

text(
    W / 2,
    85,
    "Schematic reconstructed from generated SPICE",
    "small"
)


# --------------------------------------------------
# Draw components
# --------------------------------------------------

pins = {}

for name, comp in components.items():

    if name not in positions:
        continue

    x, y = xy(name)

    t = comp["type"]

    if t in {"NMOS", "PMOS"}:

        pins[name] = mosfet(
            x,
            y,
            name,
            t,
            comp.get("model", "")
        )

    elif t == "INDUCTOR":

        pins[name] = inductor(
            x,
            y,
            name,
            comp.get("value", "")
        )

    elif t == "CAPACITOR":

        pins[name] = capacitor(
            x,
            y,
            name,
            comp.get("value", "")
        )


# --------------------------------------------------
# Helper for pin
# --------------------------------------------------

def P(component, pin):

    return pins[component][pin]


# --------------------------------------------------
# Key net routing
#
# These routes are derived from the parsed SPICE,
# with human-readable orthogonal layout.
# --------------------------------------------------

# ==================================================
# IB1 common source rail for XM0 + XM3
# ==================================================

if "XM0" in pins and "XM3" in pins:

    s0 = P("XM0", "S")
    s3 = P("XM3", "S")

    rail_y = min(
        s0[1],
        s3[1]
    ) - 55

    path([
        s0,
        (s0[0], rail_y),
        (s3[0], rail_y),
        s3
    ])

    mid_x = (
        s0[0] + s3[0]
    ) / 2

    dot(
        mid_x,
        rail_y
    )

    line(
        mid_x,
        rail_y,
        mid_x,
        rail_y - 55
    )

    text(
        mid_x + 15,
        rail_y - 25,
        "IB1",
        "net",
        "start"
    )


# ==================================================
# Cross-coupled PMOS
#
# XM0 D = net6, G = net1
# XM3 D = net1, G = net6
# ==================================================

if "XM0" in pins and "XM3" in pins:

    xm0_d = P("XM0", "D")
    xm0_g = P("XM0", "G")

    xm3_d = P("XM3", "D")
    xm3_g = P("XM3", "G")

    left_route_x = min(
        xm0_g[0],
        xm0_d[0]
    ) - 60

    right_route_x = max(
        xm3_d[0],
        xm3_g[0]
    ) + 60

    # net1:
    # XM3 drain -> XM0 gate

    route_y_1 = (
        xm0_g[1] + 25
    )

    path([
        xm3_d,
        (
            right_route_x,
            xm3_d[1]
        ),
        (
            right_route_x,
            route_y_1
        ),
        (
            xm0_g[0],
            route_y_1
        ),
        xm0_g
    ])

    # net6:
    # XM0 drain -> XM3 gate

    route_y_2 = (
        xm3_g[1] - 25
    )

    path([
        xm0_d,
        (
            left_route_x,
            xm0_d[1]
        ),
        (
            left_route_x,
            route_y_2
        ),
        (
            xm3_g[0],
            route_y_2
        ),
        xm3_g
    ])


# ==================================================
# Connect PMOS drain nodes to inductors
#
# XM0 D = net6 -> L1
# XM3 D = net1 -> L0
# ==================================================

if "XM0" in pins and "L1" in pins:

    a = P("XM0", "D")
    b = P("L1", "1")

    path([
        a,
        (b[0], a[1]),
        b
    ])

    dot(
        b[0],
        a[1]
    )

    text(
        b[0] - 15,
        a[1] - 15,
        "net6",
        "net",
        "end"
    )


if "XM3" in pins and "L0" in pins:

    a = P("XM3", "D")
    b = P("L0", "1")

    path([
        a,
        (b[0], a[1]),
        b
    ])

    dot(
        b[0],
        a[1]
    )

    text(
        b[0] + 15,
        a[1] - 15,
        "net1",
        "net",
        "start"
    )


# ==================================================
# NMOS gate connections
#
# XM4 G = net6
# XM1 G = net1
# ==================================================

if "XM4" in pins and "L1" in pins:

    g = P("XM4", "G")
    l = P("L1", "1")

    path([
        g,
        (l[0], g[1]),
        l
    ])


if "XM1" in pins and "L0" in pins:

    g = P("XM1", "G")
    l = P("L0", "1")

    path([
        g,
        (l[0], g[1]),
        l
    ])


# ==================================================
# VCONT1:
# XM1 D/S and XM4 D/S share VCONT1
# ==================================================

vcont_points = []

for m in ["XM4", "XM1"]:

    if m in pins:

        vcont_points.append(
            P(m, "S")
        )

        vcont_points.append(
            P(m, "D")
        )


if vcont_points:

    rail_y = max(
        p[1]
        for p in vcont_points
    ) + 100

    xs = [
        p[0]
        for p in vcont_points
    ]

    xmin = min(xs)
    xmax = max(xs)

    line(
        xmin,
        rail_y,
        xmax,
        rail_y
    )

    for p in vcont_points:

        path([
            p,
            (
                p[0],
                rail_y
            )
        ])

        dot(
            p[0],
            rail_y
        )

    text(
        (
            xmin + xmax
        ) / 2,
        rail_y + 38,
        "VCONT1 = 0.9 V",
        "net"
    )


# ==================================================
# Inductor bottoms -> VSS
# ==================================================

vss_bottoms = []

for l in ["L1", "L0"]:

    if l in pins:

        p = P(l, "2")

        vss_bottoms.append(p)


if vss_bottoms:

    rail_y = max(
        p[1]
        for p in vss_bottoms
    ) + 75

    xmin = min(
        p[0]
        for p in vss_bottoms
    )

    xmax = max(
        p[0]
        for p in vss_bottoms
    )

    line(
        xmin,
        rail_y,
        xmax,
        rail_y
    )

    for p in vss_bottoms:

        line(
            p[0],
            p[1],
            p[0],
            rail_y
        )

        dot(
            p[0],
            rail_y
        )

    gx = (
        xmin + xmax
    ) / 2

    ground(
        gx,
        rail_y
    )

    text(
        gx + 35,
        rail_y + 35,
        "VSS",
        "net",
        "start"
    )


# ==================================================
# CL: net1 -> ground
# ==================================================

if "CL" in pins:

    top = P("CL", "1")
    bottom = P("CL", "2")

    # Connect CL top to net1 / L0 upper node

    if "L0" in pins:

        n1 = P("L0", "1")

        path([
            top,
            (
                top[0],
                n1[1]
            ),
            n1
        ])

        dot(
            n1[0],
            n1[1]
        )

    ground(
        bottom[0],
        bottom[1]
    )


# ==================================================
# Bulk labels
# ==================================================

for m in ["XM0", "XM3"]:

    if m in pins:

        b = P(m, "B")

        line(
            b[0],
            b[1],
            b[0] + 45,
            b[1]
        )

        text(
            b[0] + 55,
            b[1] + 7,
            "VDD",
            "net",
            "start"
        )


for m in ["XM1", "XM4"]:

    if m in pins:

        b = P(m, "B")

        line(
            b[0],
            b[1],
            b[0] + 45,
            b[1]
        )

        text(
            b[0] + 55,
            b[1] + 7,
            "VSS",
            "net",
            "start"
        )


# --------------------------------------------------
# Footer
# --------------------------------------------------

text(
    40,
    H - 40,
    "Generated from qwen_analogtobi_0535.spice",
    "small",
    "start"
)

add("</svg>")

SVG_FILE.write_text(
    "\n".join(svg)
)

print("=" * 65)
print("SCHEMATIC SVG GENERATED")
print("=" * 65)

print("SVG:", SVG_FILE)

# --------------------------------------------------
# SVG -> high-resolution PNG
# --------------------------------------------------

try:

    import cairosvg

    cairosvg.svg2png(
        bytestring=SVG_FILE.read_bytes(),
        write_to=str(PNG_FILE),
        output_width=3200,
        output_height=2200
    )

    print("PNG:", PNG_FILE)

except Exception as e:

    print("\nSVG generated successfully.")
    print("PNG conversion failed:")
    print(e)

print("\nDone.")