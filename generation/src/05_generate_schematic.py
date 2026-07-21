import json
import html
from pathlib import Path

# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

ROUTE_DIR = ROOT / "data" / "routes"
SVG_DIR = ROOT / "output" / "svg"

SVG_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# RENDER SETTINGS
# ============================================================

SCALE = 55.0

MARGIN = 100.0

SYMBOL_HALF_W = 0.75
SYMBOL_HALF_H = 0.85

WIRE_WIDTH = 2.2
SYMBOL_WIDTH = 2.2

JUNCTION_RADIUS = 0.075

FONT_SIZE = 13
SMALL_FONT_SIZE = 11

# SVG colors
WIRE_COLOR = "#111111"
SYMBOL_COLOR = "#111111"
TEXT_COLOR = "#111111"
NET_LABEL_COLOR = "#333333"
BG_COLOR = "#ffffff"


# ============================================================
# BASIC HELPERS
# ============================================================

def esc(text):
    return html.escape(str(text))


def svg_line(x1, y1, x2, y2, width=SYMBOL_WIDTH):
    return (
        f'<line x1="{x1:.2f}" y1="{y1:.2f}" '
        f'x2="{x2:.2f}" y2="{y2:.2f}" '
        f'stroke="{SYMBOL_COLOR}" '
        f'stroke-width="{width}" '
        f'stroke-linecap="round"/>'
    )


def svg_wire(x1, y1, x2, y2):
    return (
        f'<line x1="{x1:.2f}" y1="{y1:.2f}" '
        f'x2="{x2:.2f}" y2="{y2:.2f}" '
        f'stroke="{WIRE_COLOR}" '
        f'stroke-width="{WIRE_WIDTH}" '
        f'stroke-linecap="square"/>'
    )


def svg_circle(cx, cy, r, fill="none", width=SYMBOL_WIDTH):
    return (
        f'<circle cx="{cx:.2f}" cy="{cy:.2f}" '
        f'r="{r:.2f}" '
        f'fill="{fill}" '
        f'stroke="{SYMBOL_COLOR}" '
        f'stroke-width="{width}"/>'
    )


def svg_text(
    x,
    y,
    text,
    size=FONT_SIZE,
    anchor="start",
    weight="normal"
):
    return (
        f'<text x="{x:.2f}" y="{y:.2f}" '
        f'font-family="Arial, Helvetica, sans-serif" '
        f'font-size="{size}" '
        f'font-weight="{weight}" '
        f'text-anchor="{anchor}" '
        f'fill="{TEXT_COLOR}">'
        f'{esc(text)}</text>'
    )


# ============================================================
# COORDINATE TRANSFORM
# ============================================================

class Transform:

    def __init__(
        self,
        min_x,
        min_y,
        scale=SCALE,
        margin=MARGIN
    ):

        self.min_x = min_x
        self.min_y = min_y

        self.scale = scale
        self.margin = margin

    def x(self, value):

        return (
            (value - self.min_x)
            * self.scale
            + self.margin
        )

    def y(self, value):

        return (
            (value - self.min_y)
            * self.scale
            + self.margin
        )

    def p(self, x, y):

        return (
            self.x(x),
            self.y(y)
        )


# ============================================================
# FIND DRAWING BOUNDS
# ============================================================

def collect_all_points(data):

    xs = []
    ys = []

    # placements

    for p in data[
        "placements"
    ].values():

        xs.append(p["x"])
        ys.append(p["y"])

    # anchors

    for pins in data[
        "pin_anchors"
    ].values():

        for info in pins.values():

            p = info["point"]

            xs.append(p["x"])
            ys.append(p["y"])

    # routes

    for route in data[
        "routes"
    ].values():

        for seg in route[
            "segments"
        ]:

            xs.extend([
                seg["x1"],
                seg["x2"]
            ])

            ys.extend([
                seg["y1"],
                seg["y2"]
            ])

    if not xs:

        return (
            -10,
            10,
            -10,
            10
        )

    return (
        min(xs) - 2.0,
        max(xs) + 2.0,
        min(ys) - 2.0,
        max(ys) + 2.0
    )


# ============================================================
# COMPONENT MAP
# ============================================================

def component_map(data):

    return {

        c["name"]: c

        for c in (
            data[
                "electrical_truth"
            ][
                "components"
            ]
        )
    }


# ============================================================
# SYMBOL: MOSFET
# ============================================================

def draw_mos(
    name,
    component,
    placement,
    anchors,
    tr
):

    out = []

    cx, cy = tr.p(
        placement["x"],
        placement["y"]
    )

    pins = anchors.get(
        name,
        {}
    )

    mirror = placement.get(
        "mirror",
        False
    )

    # Get transformed pin positions

    def pin_xy(pin):

        p = pins[
            pin
        ][
            "point"
        ]

        return tr.p(
            p["x"],
            p["y"]
        )

    dx, dy = pin_xy("D")
    gx, gy = pin_xy("G")
    sx, sy = pin_xy("S")
    bx, by = pin_xy("B")

    # Main channel

    channel_top = (
        cy
        - 0.50 * SCALE
    )

    channel_bottom = (
        cy
        + 0.50 * SCALE
    )

    channel_x = cx

    out.append(
        svg_line(
            channel_x,
            channel_top,
            channel_x,
            channel_bottom
        )
    )

    # Drain

    out.append(
        svg_line(
            dx,
            dy,
            channel_x,
            channel_top
        )
    )

    # Source

    out.append(
        svg_line(
            sx,
            sy,
            channel_x,
            channel_bottom
        )
    )

    # Gate plate

    if not mirror:

        gate_plate_x = (
            cx
            - 0.28 * SCALE
        )

    else:

        gate_plate_x = (
            cx
            + 0.28 * SCALE
        )

    out.append(
        svg_line(
            gate_plate_x,
            cy - 0.42 * SCALE,
            gate_plate_x,
            cy + 0.42 * SCALE
        )
    )

    # Gate connection

    out.append(
        svg_line(
            gx,
            gy,
            gate_plate_x,
            cy
        )
    )

    # Body connection

    out.append(
        svg_line(
            bx,
            by,
            channel_x,
            cy
        )
    )

    # PMOS bubble on gate

    if component[
        "type"
    ] == "PMOS":

        bubble_x = (
            gate_plate_x
            + (
                0.10 * SCALE
                if not mirror
                else
                -0.10 * SCALE
            )
        )

        out.append(
            svg_circle(
                bubble_x,
                cy,
                0.09 * SCALE,
                fill=BG_COLOR
            )
        )

    # Arrow / type hint

    if component[
        "type"
    ] == "NMOS":

        marker = "N"

    elif component[
        "type"
    ] == "PMOS":

        marker = "P"

    else:

        marker = "M"

    # Component name

    label_x = (
        cx
        + (
            0.65 * SCALE
            if not mirror
            else
            -0.65 * SCALE
        )
    )

    anchor = (
        "start"
        if not mirror
        else
        "end"
    )

    out.append(
        svg_text(
            label_x,
            cy - 0.18 * SCALE,
            name,
            FONT_SIZE,
            anchor,
            "bold"
        )
    )

    out.append(
        svg_text(
            label_x,
            cy + 0.10 * SCALE,
            marker,
            SMALL_FONT_SIZE,
            anchor
        )
    )

    return out


# ============================================================
# SYMBOL: CAPACITOR
# ============================================================

def draw_capacitor(
    name,
    placement,
    anchors,
    tr
):

    out = []

    cx, cy = tr.p(
        placement["x"],
        placement["y"]
    )

    p1 = anchors[
        name
    ][
        "1"
    ][
        "point"
    ]

    p2 = anchors[
        name
    ][
        "2"
    ][
        "point"
    ]

    x1, y1 = tr.p(
        p1["x"],
        p1["y"]
    )

    x2, y2 = tr.p(
        p2["x"],
        p2["y"]
    )

    plate1_y = (
        cy
        - 0.12 * SCALE
    )

    plate2_y = (
        cy
        + 0.12 * SCALE
    )

    half_w = (
        0.38 * SCALE
    )

    out.append(
        svg_line(
            x1,
            y1,
            cx,
            plate1_y
        )
    )

    out.append(
        svg_line(
            cx - half_w,
            plate1_y,
            cx + half_w,
            plate1_y
        )
    )

    out.append(
        svg_line(
            cx - half_w,
            plate2_y,
            cx + half_w,
            plate2_y
        )
    )

    out.append(
        svg_line(
            cx,
            plate2_y,
            x2,
            y2
        )
    )

    out.append(
        svg_text(
            cx + 0.55 * SCALE,
            cy + 4,
            name,
            FONT_SIZE,
            "start",
            "bold"
        )
    )

    return out


# ============================================================
# SYMBOL: RESISTOR
# ============================================================

def draw_resistor(
    name,
    placement,
    anchors,
    tr
):

    out = []

    cx, cy = tr.p(
        placement["x"],
        placement["y"]
    )

    p1 = anchors[
        name
    ][
        "1"
    ][
        "point"
    ]

    p2 = anchors[
        name
    ][
        "2"
    ][
        "point"
    ]

    x1, y1 = tr.p(
        p1["x"],
        p1["y"]
    )

    x2, y2 = tr.p(
        p2["x"],
        p2["y"]
    )

    top = (
        cy
        - 0.50 * SCALE
    )

    bottom = (
        cy
        + 0.50 * SCALE
    )

    out.append(
        svg_line(
            x1,
            y1,
            cx,
            top
        )
    )

    # Zig-zag

    steps = 8

    points = []

    for i in range(
        steps + 1
    ):

        y = (
            top
            + (
                bottom - top
            )
            * i / steps
        )

        if i in {
            0,
            steps
        }:

            x = cx

        elif i % 2:

            x = (
                cx
                - 0.18 * SCALE
            )

        else:

            x = (
                cx
                + 0.18 * SCALE
            )

        points.append(
            (
                x,
                y
            )
        )

    path = (
        " ".join(
            f"{x:.2f},{y:.2f}"
            for x, y in points
        )
    )

    out.append(
        f'<polyline points="{path}" '
        f'fill="none" '
        f'stroke="{SYMBOL_COLOR}" '
        f'stroke-width="{SYMBOL_WIDTH}" '
        f'stroke-linejoin="round"/>'
    )

    out.append(
        svg_line(
            cx,
            bottom,
            x2,
            y2
        )
    )

    out.append(
        svg_text(
            cx + 0.45 * SCALE,
            cy + 4,
            name,
            FONT_SIZE,
            "start",
            "bold"
        )
    )

    return out


# ============================================================
# SYMBOL: INDUCTOR
# ============================================================

def draw_inductor(
    name,
    placement,
    anchors,
    tr
):

    out = []

    cx, cy = tr.p(
        placement["x"],
        placement["y"]
    )

    p1 = anchors[
        name
    ][
        "1"
    ][
        "point"
    ]

    p2 = anchors[
        name
    ][
        "2"
    ][
        "point"
    ]

    x1, y1 = tr.p(
        p1["x"],
        p1["y"]
    )

    x2, y2 = tr.p(
        p2["x"],
        p2["y"]
    )

    coil_top = (
        cy
        - 0.50 * SCALE
    )

    coil_bottom = (
        cy
        + 0.50 * SCALE
    )

    out.append(
        svg_line(
            x1,
            y1,
            cx,
            coil_top
        )
    )

    # Vertical inductor using alternating Bezier bulges

    loops = 4

    loop_h = (
        coil_bottom
        - coil_top
    ) / loops

    path = [
        f"M {cx:.2f} {coil_top:.2f}"
    ]

    for i in range(loops):

        ya = (
            coil_top
            + i * loop_h
        )

        yb = (
            ya
            + loop_h
        )

        ym = (
            ya
            + loop_h / 2
        )

        path.append(
            f"C "
            f"{cx + 0.32*SCALE:.2f} "
            f"{ya + 0.12*loop_h:.2f}, "
            f"{cx + 0.32*SCALE:.2f} "
            f"{yb - 0.12*loop_h:.2f}, "
            f"{cx:.2f} "
            f"{yb:.2f}"
        )

    out.append(
        f'<path d="{" ".join(path)}" '
        f'fill="none" '
        f'stroke="{SYMBOL_COLOR}" '
        f'stroke-width="{SYMBOL_WIDTH}"/>'
    )

    out.append(
        svg_line(
            cx,
            coil_bottom,
            x2,
            y2
        )
    )

    out.append(
        svg_text(
            cx + 0.50 * SCALE,
            cy + 4,
            name,
            FONT_SIZE,
            "start",
            "bold"
        )
    )

    return out


# ============================================================
# SYMBOL: VOLTAGE SOURCE
# ============================================================

def draw_voltage_source(
    name,
    placement,
    anchors,
    tr
):

    out = []

    cx, cy = tr.p(
        placement["x"],
        placement["y"]
    )

    plus = anchors[
        name
    ][
        "+"
    ][
        "point"
    ]

    minus = anchors[
        name
    ][
        "-"
    ][
        "point"
    ]

    px, py = tr.p(
        plus["x"],
        plus["y"]
    )

    mx, my = tr.p(
        minus["x"],
        minus["y"]
    )

    radius = (
        0.42 * SCALE
    )

    out.append(
        svg_line(
            px,
            py,
            cx,
            cy - radius
        )
    )

    out.append(
        svg_circle(
            cx,
            cy,
            radius,
            fill=BG_COLOR
        )
    )

    out.append(
        svg_line(
            cx,
            cy + radius,
            mx,
            my
        )
    )

    out.append(
        svg_text(
            cx,
            cy - 0.10 * SCALE,
            "+",
            FONT_SIZE,
            "middle",
            "bold"
        )
    )

    out.append(
        svg_text(
            cx,
            cy + 0.20 * SCALE,
            "−",
            FONT_SIZE,
            "middle",
            "bold"
        )
    )

    out.append(
        svg_text(
            cx + 0.60 * SCALE,
            cy + 4,
            name,
            FONT_SIZE,
            "start",
            "bold"
        )
    )

    return out


# ============================================================
# SYMBOL: CURRENT SOURCE
# ============================================================

def draw_current_source(
    name,
    placement,
    anchors,
    tr
):

    out = []

    cx, cy = tr.p(
        placement["x"],
        placement["y"]
    )

    plus = anchors[
        name
    ][
        "+"
    ][
        "point"
    ]

    minus = anchors[
        name
    ][
        "-"
    ][
        "point"
    ]

    px, py = tr.p(
        plus["x"],
        plus["y"]
    )

    mx, my = tr.p(
        minus["x"],
        minus["y"]
    )

    radius = (
        0.42 * SCALE
    )

    out.append(
        svg_line(
            px,
            py,
            cx,
            cy - radius
        )
    )

    out.append(
        svg_circle(
            cx,
            cy,
            radius,
            fill=BG_COLOR
        )
    )

    out.append(
        svg_line(
            cx,
            cy + radius,
            mx,
            my
        )
    )

    # Arrow

    out.append(
        svg_line(
            cx,
            cy + 0.20 * SCALE,
            cx,
            cy - 0.20 * SCALE
        )
    )

    arrow = (
        f"{cx:.2f},"
        f"{cy - 0.25*SCALE:.2f} "
        f"{cx - 0.10*SCALE:.2f},"
        f"{cy - 0.08*SCALE:.2f} "
        f"{cx + 0.10*SCALE:.2f},"
        f"{cy - 0.08*SCALE:.2f}"
    )

    out.append(
        f'<polygon points="{arrow}" '
        f'fill="{SYMBOL_COLOR}"/>'
    )

    out.append(
        svg_text(
            cx + 0.60 * SCALE,
            cy + 4,
            name,
            FONT_SIZE,
            "start",
            "bold"
        )
    )

    return out


# ============================================================
# SYMBOL: GENERIC BLOCK
# ============================================================

def draw_generic(
    name,
    component,
    placement,
    anchors,
    tr
):

    out = []

    cx, cy = tr.p(
        placement["x"],
        placement["y"]
    )

    half_w = (
        0.65 * SCALE
    )

    half_h = (
        0.70 * SCALE
    )

    out.append(
        f'<rect '
        f'x="{cx-half_w:.2f}" '
        f'y="{cy-half_h:.2f}" '
        f'width="{2*half_w:.2f}" '
        f'height="{2*half_h:.2f}" '
        f'fill="{BG_COLOR}" '
        f'stroke="{SYMBOL_COLOR}" '
        f'stroke-width="{SYMBOL_WIDTH}"/>'
    )

    out.append(
        svg_text(
            cx,
            cy - 3,
            name,
            FONT_SIZE,
            "middle",
            "bold"
        )
    )

    out.append(
        svg_text(
            cx,
            cy + 15,
            component["type"],
            SMALL_FONT_SIZE,
            "middle"
        )
    )

    # Connect anchors to block boundary approximately

    for (
        pin_name,
        info
    ) in anchors.get(
        name,
        {}
    ).items():

        p = info[
            "point"
        ]

        px, py = tr.p(
            p["x"],
            p["y"]
        )

        if px < cx:

            tx = (
                cx
                - half_w
            )

        else:

            tx = (
                cx
                + half_w
            )

        out.append(
            svg_line(
                px,
                py,
                tx,
                py
            )
        )

        out.append(
            svg_text(
                (
                    tx + 4
                    if px < cx
                    else
                    tx - 4
                ),
                py - 4,
                pin_name,
                SMALL_FONT_SIZE,
                (
                    "start"
                    if px < cx
                    else
                    "end"
                )
            )
        )

    return out


# ============================================================
# DRAW ALL COMPONENTS
# ============================================================

def draw_components(
    data,
    tr
):

    out = []

    components = component_map(
        data
    )

    placements = data[
        "placements"
    ]

    anchors = data[
        "pin_anchors"
    ]

    # Draw sources first, core devices later

    order_priority = {

        "VOLTAGE_SOURCE": 0,
        "CURRENT_SOURCE": 0,

        "RESISTOR": 1,
        "CAPACITOR": 1,
        "INDUCTOR": 1,

        "NMOS": 2,
        "PMOS": 2,
        "MOS": 2
    }

    names = sorted(

        placements.keys(),

        key=lambda name: (

            order_priority.get(
                components.get(
                    name,
                    {}
                ).get(
                    "type",
                    "UNKNOWN"
                ),
                10
            ),

            placements[name][
                "y"
            ],

            placements[name][
                "x"
            ]
        )
    )

    for name in names:

        if name not in components:
            continue

        component = components[
            name
        ]

        placement = placements[
            name
        ]

        ctype = component[
            "type"
        ]

        if ctype in {
            "NMOS",
            "PMOS",
            "MOS"
        }:

            out.extend(
                draw_mos(
                    name,
                    component,
                    placement,
                    anchors,
                    tr
                )
            )

        elif ctype == "CAPACITOR":

            out.extend(
                draw_capacitor(
                    name,
                    placement,
                    anchors,
                    tr
                )
            )

        elif ctype == "RESISTOR":

            out.extend(
                draw_resistor(
                    name,
                    placement,
                    anchors,
                    tr
                )
            )

        elif ctype == "INDUCTOR":

            out.extend(
                draw_inductor(
                    name,
                    placement,
                    anchors,
                    tr
                )
            )

        elif ctype == "VOLTAGE_SOURCE":

            out.extend(
                draw_voltage_source(
                    name,
                    placement,
                    anchors,
                    tr
                )
            )

        elif ctype == "CURRENT_SOURCE":

            out.extend(
                draw_current_source(
                    name,
                    placement,
                    anchors,
                    tr
                )
            )

        else:

            out.extend(
                draw_generic(
                    name,
                    component,
                    placement,
                    anchors,
                    tr
                )
            )

    return out


# ============================================================
# DRAW ROUTES
# ============================================================

def draw_routes(
    data,
    tr
):

    out = []

    for (
        net_name,
        route
    ) in data[
        "routes"
    ].items():

        out.append(
            f'<g id="net-{esc(net_name)}">'
        )

        for seg in route[
            "segments"
        ]:

            x1, y1 = tr.p(
                seg["x1"],
                seg["y1"]
            )

            x2, y2 = tr.p(
                seg["x2"],
                seg["y2"]
            )

            out.append(
                svg_wire(
                    x1,
                    y1,
                    x2,
                    y2
                )
            )

        out.append(
            "</g>"
        )

    return out


# ============================================================
# DRAW JUNCTIONS
# ============================================================

def draw_junctions(
    data,
    tr
):

    out = []

    seen = set()

    for route in data[
        "routes"
    ].values():

        for p in route[
            "junctions"
        ]:

            key = (
                round(
                    p["x"],
                    3
                ),
                round(
                    p["y"],
                    3
                )
            )

            if key in seen:
                continue

            seen.add(key)

            x, y = tr.p(
                p["x"],
                p["y"]
            )

            out.append(
                f'<circle '
                f'cx="{x:.2f}" '
                f'cy="{y:.2f}" '
                f'r="{JUNCTION_RADIUS*SCALE:.2f}" '
                f'fill="{WIRE_COLOR}"/>'
            )

    return out


# ============================================================
# DRAW NET LABELS
#
# We deliberately avoid labeling every terminal.
# Only one label per important bus/rail.
# ============================================================

def draw_net_labels(
    data,
    tr
):

    out = []

    for (
        net_name,
        route
    ) in data[
        "routes"
    ].items():

        if not route[
            "segments"
        ]:
            continue

        net_class = route.get(
            "class",
            "signal"
        )

        strategy = route.get(
            "strategy",
            ""
        )

        # Label important nets and shared buses.

        if (
            net_class
            not in {
                "power",
                "ground",
                "bias",
                "control"
            }
            and
            strategy
            != "shared_bus"
        ):

            continue

        # Choose first horizontal segment if possible.

        chosen = None

        for seg in route[
            "segments"
        ]:

            if abs(
                seg["y1"]
                - seg["y2"]
            ) < 1e-9:

                chosen = seg
                break

        if chosen is None:

            chosen = route[
                "segments"
            ][0]

        x = min(
            chosen["x1"],
            chosen["x2"]
        )

        y = min(
            chosen["y1"],
            chosen["y2"]
        )

        sx, sy = tr.p(
            x,
            y
        )

        out.append(
            svg_text(
                sx + 5,
                sy - 7,
                net_name,
                SMALL_FONT_SIZE,
                "start",
                "bold"
            )
        )

    return out


# ============================================================
# BUILD SVG
# ============================================================

def build_svg(data):

    (
        min_x,
        max_x,
        min_y,
        max_y
    ) = collect_all_points(
        data
    )

    tr = Transform(
        min_x,
        min_y
    )

    width = (
        (max_x - min_x)
        * SCALE
        + 2 * MARGIN
    )

    height = (
        (max_y - min_y)
        * SCALE
        + 2 * MARGIN
    )

    svg = []

    svg.append(
        '<?xml version="1.0" '
        'encoding="UTF-8"?>'
    )

    svg.append(
        f'<svg '
        f'xmlns="http://www.w3.org/2000/svg" '
        f'width="{width:.0f}" '
        f'height="{height:.0f}" '
        f'viewBox="0 0 '
        f'{width:.0f} {height:.0f}">'
    )

    # Background

    svg.append(
        f'<rect '
        f'x="0" y="0" '
        f'width="{width:.0f}" '
        f'height="{height:.0f}" '
        f'fill="{BG_COLOR}"/>'
    )

    # Title

    svg.append(
        svg_text(
            MARGIN,
            42,
            (
                "Generated schematic — "
                + data[
                    "source_file"
                ]
            ),
            18,
            "start",
            "bold"
        )
    )

    # --------------------------------------------------------
    # Layer 1: wires
    # --------------------------------------------------------

    svg.append(
        '<g id="wires">'
    )

    svg.extend(
        draw_routes(
            data,
            tr
        )
    )

    svg.append(
        '</g>'
    )

    # --------------------------------------------------------
    # Layer 2: symbols
    #
    # White symbol interiors help hide wires passing behind
    # circular/block symbols.
    # --------------------------------------------------------

    svg.append(
        '<g id="components">'
    )

    svg.extend(
        draw_components(
            data,
            tr
        )
    )

    svg.append(
        '</g>'
    )

    # --------------------------------------------------------
    # Layer 3: junctions
    # --------------------------------------------------------

    svg.append(
        '<g id="junctions">'
    )

    svg.extend(
        draw_junctions(
            data,
            tr
        )
    )

    svg.append(
        '</g>'
    )

    # --------------------------------------------------------
    # Layer 4: net labels
    # --------------------------------------------------------

    svg.append(
        '<g id="labels">'
    )

    svg.extend(
        draw_net_labels(
            data,
            tr
        )
    )

    svg.append(
        '</g>'
    )

    svg.append(
        '</svg>'
    )

    return "\n".join(
        svg
    )


# ============================================================
# PROCESS FILE
# ============================================================

def process_file(path):

    print(
        "\n"
        + "=" * 78
    )

    print(
        f"GENERATING SVG: "
        f"{path.name}"
    )

    print(
        "=" * 78
    )

    data = json.loads(
        path.read_text()
    )

    validation_errors = data.get(
        "validation_errors",
        []
    )

    if validation_errors:

        print(
            "\nWARNING: route file "
            "contains validation errors:"
        )

        for error in validation_errors:

            print(
                "  ",
                error
            )

    svg = build_svg(
        data
    )

    stem = path.stem

    if stem.endswith(
        "_routes"
    ):

        stem = stem[
            :-len("_routes")
        ]

    output_path = (
        SVG_DIR
        / f"{stem}.svg"
    )

    output_path.write_text(
        svg,
        encoding="utf-8"
    )

    print(
        f"Components : "
        f"{len(data['placements'])}"
    )

    print(
        f"Nets       : "
        f"{len(data['routes'])}"
    )

    total_segments = sum(

        len(
            route[
                "segments"
            ]
        )

        for route in (
            data[
                "routes"
            ].values()
        )
    )

    print(
        f"Wire segments : "
        f"{total_segments}"
    )

    print(
        f"\nSaved → "
        f"{output_path}"
    )

    return output_path


# ============================================================
# MAIN
# ============================================================

def main():

    files = sorted(
        ROUTE_DIR.glob(
            "*_routes.json"
        )
    )

    print(
        "=" * 78
    )

    print(
        "SCHEMATIC SVG GENERATOR"
    )

    print(
        "=" * 78
    )

    print(
        f"Route directory : "
        f"{ROUTE_DIR}"
    )

    print(
        f"Files found     : "
        f"{len(files)}"
    )

    if not files:

        print(
            "\nNo route files found."
        )

        print(
            "Run first:"
        )

        print(
            "python "
            "src/04_route_schematic.py"
        )

        return

    success = 0
    failed = 0

    for path in files:

        try:

            process_file(
                path
            )

            success += 1

        except Exception as e:

            failed += 1

            print(
                f"\nFAILED: "
                f"{path.name}"
            )

            print(
                type(e).__name__,
                ":",
                e
            )

    print(
        "\n"
        + "=" * 78
    )

    print(
        "SVG GENERATION COMPLETE"
    )

    print(
        "=" * 78
    )

    print(
        f"Successful : {success}"
    )

    print(
        f"Failed     : {failed}"
    )

    print(
        f"Output dir : "
        f"{SVG_DIR}"
    )


if __name__ == "__main__":

    main()