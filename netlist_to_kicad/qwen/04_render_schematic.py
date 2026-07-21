import json
import html
from pathlib import Path

INPUT = "schematic_ir.json"

OUT = Path("output")
OUT.mkdir(exist_ok=True)

SVG = OUT / "qwen_analogtobi_0535.svg"
PNG = OUT / "qwen_analogtobi_0535.png"

data = json.loads(Path(INPUT).read_text())

components = {
    c["name"]: c
    for c in data["components"]
}

placement = data["placement"]
net_plan = data["nets"]

W = 1900
H = 1250

SCALE = 105
OX = 1000
OY = 130


def screen(name):
    p = placement[name]

    return (
        OX + p["x"] * SCALE,
        OY + p["y"] * SCALE
    )


svg = []


def A(s):
    svg.append(s)


A(
    f'<svg xmlns="http://www.w3.org/2000/svg" '
    f'width="{W}" height="{H}" '
    f'viewBox="0 0 {W} {H}">'
)

A("""
<rect width="100%" height="100%" fill="white"/>

<style>
text {
    font-family: Arial, Helvetica, sans-serif;
    fill: #111;
}

.wire {
    stroke: #111;
    stroke-width: 3;
    fill: none;
    stroke-linecap: round;
    stroke-linejoin: round;
}

.symbol {
    stroke: #111;
    stroke-width: 3;
    fill: none;
}

.node {
    fill: #111;
}

.name {
    font-size: 22px;
    font-weight: bold;
}

.value {
    font-size: 16px;
}

.netlabel {
    font-size: 17px;
    font-weight: bold;
}

.title {
    font-size: 28px;
    font-weight: bold;
}
</style>
""")


def line(x1, y1, x2, y2, cls="wire"):
    A(
        f'<line x1="{x1}" y1="{y1}" '
        f'x2="{x2}" y2="{y2}" '
        f'class="{cls}"/>'
    )


def poly(points):
    pts = " ".join(
        f"{x},{y}"
        for x, y in points
    )

    A(
        f'<polyline points="{pts}" '
        f'class="wire"/>'
    )


def text(x, y, s, cls="value", anchor="middle"):
    s = html.escape(str(s))

    A(
        f'<text x="{x}" y="{y}" '
        f'class="{cls}" '
        f'text-anchor="{anchor}">{s}</text>'
    )


def circle(x, y, r=5):
    A(
        f'<circle cx="{x}" cy="{y}" '
        f'r="{r}" class="node"/>'
    )


def ground(x, y):
    line(x, y, x, y + 12)

    line(
        x - 18,
        y + 12,
        x + 18,
        y + 12,
        "symbol"
    )

    line(
        x - 12,
        y + 20,
        x + 12,
        y + 20,
        "symbol"
    )

    line(
        x - 6,
        y + 28,
        x + 6,
        y + 28,
        "symbol"
    )


def draw_mos(x, y, c):
    t = c["type"]

    # Source top, drain bottom, gate left, bulk right
    pins = {
        "S": (x, y - 52),
        "D": (x, y + 52),
        "G": (x - 72, y),
        "B": (x + 60, y)
    }

    # Channel
    line(
        x,
        y - 30,
        x,
        y + 30,
        "symbol"
    )

    line(
        x,
        y - 52,
        x,
        y - 30,
        "symbol"
    )

    line(
        x,
        y + 30,
        x,
        y + 52,
        "symbol"
    )

    # Gate
    line(
        x - 42,
        y - 30,
        x - 42,
        y + 30,
        "symbol"
    )

    line(
        x - 72,
        y,
        x - 42,
        y,
        "symbol"
    )

    # Bulk
    line(
        x + 12,
        y,
        x + 60,
        y,
        "symbol"
    )

    if t == "PMOS":
        A(
            f'<circle cx="{x - 34}" '
            f'cy="{y}" r="7" '
            f'fill="white" '
            f'stroke="#111" stroke-width="3"/>'
        )

    text(
        x + 80,
        y - 10,
        c["name"],
        "name",
        "start"
    )

    text(
        x + 80,
        y + 15,
        "PFET" if t == "PMOS" else "NFET",
        "value",
        "start"
    )

    params = c.get("params", {})

    if "W" in params:
        text(
            x + 80,
            y + 38,
            f"W = {params['W']}",
            "value",
            "start"
        )

    if "L" in params:
        text(
            x + 80,
            y + 60,
            f"L = {params['L']}",
            "value",
            "start"
        )

    return pins


def draw_inductor(x, y, c):
    pins = {
        "1": (x, y - 55),
        "2": (x, y + 55)
    }

    line(
        x,
        y - 55,
        x,
        y - 42,
        "symbol"
    )

    for i in range(4):
        yy = y - 42 + i * 21

        A(
            f'<path d="M {x} {yy} '
            f'C {x + 28} {yy}, '
            f'{x + 28} {yy + 21}, '
            f'{x} {yy + 21}" '
            f'class="symbol"/>'
        )

    line(
        x,
        y + 42,
        x,
        y + 55,
        "symbol"
    )

    text(
        x + 45,
        y - 5,
        c["name"],
        "name",
        "start"
    )

    text(
        x + 45,
        y + 20,
        c.get("value", ""),
        "value",
        "start"
    )

    return pins


def draw_capacitor(x, y, c):
    pins = {
        "1": (x, y - 45),
        "2": (x, y + 45)
    }

    line(
        x,
        y - 45,
        x,
        y - 10,
        "symbol"
    )

    line(
        x - 25,
        y - 10,
        x + 25,
        y - 10,
        "symbol"
    )

    line(
        x - 25,
        y + 10,
        x + 25,
        y + 10,
        "symbol"
    )

    line(
        x,
        y + 10,
        x,
        y + 45,
        "symbol"
    )

    text(
        x + 42,
        y - 4,
        c["name"],
        "name",
        "start"
    )

    text(
        x + 42,
        y + 22,
        c.get("value", ""),
        "value",
        "start"
    )

    return pins


def draw_resistor(x, y, c):
    pins = {
        "1": (x, y - 50),
        "2": (x, y + 50)
    }

    line(
        x,
        y - 50,
        x,
        y - 35,
        "symbol"
    )

    A(
        f'<rect x="{x - 14}" '
        f'y="{y - 35}" '
        f'width="28" height="70" '
        f'class="symbol"/>'
    )

    line(
        x,
        y + 35,
        x,
        y + 50,
        "symbol"
    )

    text(
        x + 35,
        y - 3,
        c["name"],
        "name",
        "start"
    )

    text(
        x + 35,
        y + 22,
        c.get("value", ""),
        "value",
        "start"
    )

    return pins


def draw_source(x, y, c):
    pins = {
        "+": (x, y - 48),
        "-": (x, y + 48)
    }

    A(
        f'<circle cx="{x}" cy="{y}" '
        f'r="34" class="symbol"/>'
    )

    if c["type"] == "CURRENT_SOURCE":
        line(
            x,
            y - 15,
            x,
            y + 15,
            "symbol"
        )

        poly([
            (x - 7, y + 6),
            (x, y + 15),
            (x + 7, y + 6)
        ])

    else:
        text(
            x,
            y - 7,
            "+",
            "name"
        )

        text(
            x,
            y + 20,
            "−",
            "name"
        )

    line(
        x,
        y - 48,
        x,
        y - 34,
        "symbol"
    )

    line(
        x,
        y + 34,
        x,
        y + 48,
        "symbol"
    )

    text(
        x - 50,
        y - 8,
        c["name"],
        "name",
        "end"
    )

    text(
        x - 50,
        y + 18,
        c.get("value", ""),
        "value",
        "end"
    )

    return pins


# =================================================
# Draw title
# =================================================

text(
    W / 2,
    42,
    "Schematic reconstructed from generated SPICE",
    "title"
)


# =================================================
# Draw components
# =================================================

pin_xy = {}

for name, p in placement.items():
    c = components.get(name)

    if not c:
        continue

    x, y = screen(name)

    if c["type"] in {"NMOS", "PMOS", "MOS"}:
        pin_xy[name] = draw_mos(x, y, c)

    elif c["type"] == "INDUCTOR":
        pin_xy[name] = draw_inductor(x, y, c)

    elif c["type"] == "CAPACITOR":
        pin_xy[name] = draw_capacitor(x, y, c)

    elif c["type"] == "RESISTOR":
        pin_xy[name] = draw_resistor(x, y, c)

    elif c["type"] in {
        "VOLTAGE_SOURCE",
        "CURRENT_SOURCE"
    }:
        pin_xy[name] = draw_source(x, y, c)


# =================================================
# Net-driven routing
# =================================================

def pin_position(component, pin):
    return pin_xy.get(
        component,
        {}
    ).get(pin)


for net, plan in net_plan.items():
    pts = []

    for conn in plan["connections"]:
        p = pin_position(
            conn["component"],
            conn["pin"]
        )

        if p:
            pts.append(
                (
                    conn["component"],
                    conn["pin"],
                    p
                )
            )

    if not pts:
        continue

    net_class = plan["class"]
    strategy = plan["strategy"]

    # ---------------------------------------------
    # Ground/VSS:
    # individual ground symbols avoid spaghetti
    # ---------------------------------------------

    if net_class == "ground":
        for _, _, (x, y) in pts:
            line(
                x,
                y,
                x,
                y + 18
            )

            ground(
                x,
                y + 18
            )

            text(
                x + 25,
                y + 42,
                net,
                "netlabel",
                "start"
            )

        continue

    # ---------------------------------------------
    # Power:
    # use named upward stubs
    # ---------------------------------------------

    if net_class == "power":
        for _, _, (x, y) in pts:
            line(
                x,
                y,
                x,
                y - 30
            )

            text(
                x,
                y - 38,
                net,
                "netlabel"
            )

        continue

    # ---------------------------------------------
    # High fanout / bias:
    # net labels rather than giant complete graph
    # ---------------------------------------------

    if strategy == "named_net":
        for _, _, (x, y) in pts:
            stub = 28

            line(
                x,
                y,
                x - stub,
                y
            )

            text(
                x - stub - 5,
                y + 6,
                net,
                "netlabel",
                "end"
            )

        continue

    # ---------------------------------------------
    # Small nets:
    # Orthogonal shared bus
    # ---------------------------------------------

    if len(pts) == 1:
        x, y = pts[0][2]

        text(
            x + 10,
            y - 10,
            net,
            "netlabel",
            "start"
        )

        continue

    xs = [
        p[2][0]
        for p in pts
    ]

    ys = [
        p[2][1]
        for p in pts
    ]

    bus_y = sum(ys) / len(ys)

    xmin = min(xs)
    xmax = max(xs)

    line(
        xmin,
        bus_y,
        xmax,
        bus_y
    )

    for _, _, (x, y) in pts:
        line(
            x,
            y,
            x,
            bus_y
        )

        circle(
            x,
            bus_y,
            4
        )

    text(
        xmin,
        bus_y - 10,
        net,
        "netlabel",
        "start"
    )


A("</svg>")

SVG.write_text(
    "\n".join(svg)
)

print("=" * 70)
print("SVG GENERATED")
print("=" * 70)
print(SVG)

try:
    import cairosvg

    cairosvg.svg2png(
        bytestring=SVG.read_bytes(),
        write_to=str(PNG),
        output_width=3800,
        output_height=2500
    )

    print("PNG GENERATED")
    print(PNG)

except Exception as e:
    print("PNG conversion failed:")
    print(e)