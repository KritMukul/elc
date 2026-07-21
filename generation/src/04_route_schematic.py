import json
from pathlib import Path
from collections import defaultdict

# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

LAYOUT_DIR = ROOT / "data" / "layouts"
ROUTE_DIR = ROOT / "data" / "routes"

ROUTE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# GEOMETRY CONSTANTS
#
# These are logical schematic coordinates.
# Rendering/scaling happens later.
# ============================================================

PIN_DX = 1.20
PIN_DY = 1.20

RAIL_MARGIN_X = 3.0
RAIL_MARGIN_Y = 3.0

BUS_OFFSET = 1.5
BUS_STEP = 0.75

EPS = 1e-9


# ============================================================
# BASIC HELPERS
# ============================================================

def point(x, y):
    return {
        "x": round(float(x), 3),
        "y": round(float(y), 3)
    }


def point_tuple(p):
    return (
        round(float(p["x"]), 3),
        round(float(p["y"]), 3)
    )


def same_point(a, b):
    return (
        abs(a["x"] - b["x"]) < EPS
        and
        abs(a["y"] - b["y"]) < EPS
    )


def segment(a, b):
    return {
        "x1": round(float(a["x"]), 3),
        "y1": round(float(a["y"]), 3),
        "x2": round(float(b["x"]), 3),
        "y2": round(float(b["y"]), 3)
    }


def normalize_segment(seg):
    """
    Normalize segment direction so duplicate detection is easy.
    """

    a = (
        round(seg["x1"], 3),
        round(seg["y1"], 3)
    )

    b = (
        round(seg["x2"], 3),
        round(seg["y2"], 3)
    )

    if a <= b:
        return a + b

    return b + a


def add_segment_unique(segments, seen, a, b):
    """
    Add only non-zero orthogonal segment.
    """

    if same_point(a, b):
        return

    if (
        abs(a["x"] - b["x"]) > EPS
        and
        abs(a["y"] - b["y"]) > EPS
    ):
        raise ValueError(
            f"Non-orthogonal segment requested: "
            f"{a} -> {b}"
        )

    seg = segment(a, b)

    key = normalize_segment(seg)

    if key not in seen:
        seen.add(key)
        segments.append(seg)


# ============================================================
# PIN GEOMETRY
#
# Placement gives component center.
# Here we define approximate symbol pin anchors.
#
# Actual renderer in 05 MUST use exactly these same anchors.
# ============================================================

def mos_pin_positions(cx, cy, mirror=False):
    """
    Vertical MOS symbol convention:

             D
             |
       G ----| transistor
             |
             S

    B placed on opposite side of gate.

    Mirror swaps gate/body sides.
    """

    if not mirror:

        return {
            "D": point(cx, cy - PIN_DY),
            "G": point(cx - PIN_DX, cy),
            "S": point(cx, cy + PIN_DY),
            "B": point(cx + PIN_DX, cy)
        }

    return {
        "D": point(cx, cy - PIN_DY),
        "G": point(cx + PIN_DX, cy),
        "S": point(cx, cy + PIN_DY),
        "B": point(cx - PIN_DX, cy)
    }


def vertical_two_pin_positions(cx, cy):
    return {
        "1": point(cx, cy - PIN_DY),
        "2": point(cx, cy + PIN_DY)
    }


def source_pin_positions(cx, cy):
    return {
        "+": point(cx, cy - PIN_DY),
        "-": point(cx, cy + PIN_DY)
    }


def diode_pin_positions(cx, cy):
    return {
        "A": point(cx, cy - PIN_DY),
        "K": point(cx, cy + PIN_DY)
    }


def bjt_pin_positions(cx, cy, mirror=False):

    if not mirror:

        return {
            "C": point(cx, cy - PIN_DY),
            "B": point(cx - PIN_DX, cy),
            "E": point(cx, cy + PIN_DY),
            "S": point(cx + PIN_DX, cy)
        }

    return {
        "C": point(cx, cy - PIN_DY),
        "B": point(cx + PIN_DX, cy),
        "E": point(cx, cy + PIN_DY),
        "S": point(cx - PIN_DX, cy)
    }


def generic_pin_positions(component, cx, cy):
    """
    Generic fallback for SUBCKT / unknown multi-pin devices.

    Pins are distributed on left/right sides.
    """

    pins = list(
        component.get(
            "pins",
            {}
        ).keys()
    )

    result = {}

    if not pins:
        return result

    left = pins[::2]
    right = pins[1::2]

    def ys_for(count):

        if count <= 1:
            return [cy]

        spacing = 1.0

        start = (
            cy
            - spacing * (count - 1) / 2
        )

        return [
            start + i * spacing
            for i in range(count)
        ]

    for pin_name, y in zip(
        left,
        ys_for(len(left))
    ):

        result[pin_name] = point(
            cx - PIN_DX,
            y
        )

    for pin_name, y in zip(
        right,
        ys_for(len(right))
    ):

        result[pin_name] = point(
            cx + PIN_DX,
            y
        )

    return result


def get_component_pin_positions(
    component,
    placement
):
    """
    Returns:
        {
            pin_name: {x, y}
        }

    IMPORTANT:
    05_generate_schematic.py must use this exact geometry.
    """

    cx = placement["x"]
    cy = placement["y"]

    mirror = placement.get(
        "mirror",
        False
    )

    ctype = component["type"]

    if ctype in {
        "NMOS",
        "PMOS",
        "MOS"
    }:

        return mos_pin_positions(
            cx,
            cy,
            mirror
        )

    if ctype in {
        "RESISTOR",
        "CAPACITOR",
        "INDUCTOR"
    }:

        return vertical_two_pin_positions(
            cx,
            cy
        )

    if ctype in {
        "VOLTAGE_SOURCE",
        "CURRENT_SOURCE"
    }:

        return source_pin_positions(
            cx,
            cy
        )

    if ctype == "DIODE":

        return diode_pin_positions(
            cx,
            cy
        )

    if ctype == "BJT":

        return bjt_pin_positions(
            cx,
            cy,
            mirror
        )

    return generic_pin_positions(
        component,
        cx,
        cy
    )


# ============================================================
# BUILD EXACT PIN ANCHOR TABLE
# ============================================================

def build_pin_anchors(layout):

    components = (
        layout[
            "electrical_truth"
        ][
            "components"
        ]
    )

    placements = layout[
        "placements"
    ]

    anchors = {}

    for component in components:

        name = component["name"]

        if name not in placements:
            continue

        pin_positions = (
            get_component_pin_positions(
                component,
                placements[name]
            )
        )

        anchors[name] = {}

        for (
            pin_name,
            net_name
        ) in component.get(
            "pins",
            {}
        ).items():

            if pin_name not in pin_positions:
                continue

            anchors[name][pin_name] = {

                "net":
                    net_name,

                "point":
                    pin_positions[
                        pin_name
                    ]
            }

    return anchors


# ============================================================
# GET NET TERMINALS
# ============================================================

def collect_net_terminals(
    net_name,
    net_data,
    anchors
):

    terminals = []

    for conn in net_data[
        "connections"
    ]:

        component = conn[
            "component"
        ]

        pin_name = conn[
            "pin"
        ]

        if component not in anchors:
            continue

        if pin_name not in anchors[
            component
        ]:
            continue

        anchor = anchors[
            component
        ][
            pin_name
        ]

        terminals.append({

            "component":
                component,

            "pin":
                pin_name,

            "net":
                net_name,

            "point":
                anchor["point"]
        })

    return terminals


# ============================================================
# ROUTING UTILITY:
# COMPRESS / CLEAN SEGMENTS
# ============================================================

def clean_segments(segments):
    """
    Remove exact duplicates and zero-length segments.

    We intentionally do NOT aggressively merge all collinear
    segments here because junction semantics matter.
    """

    result = []
    seen = set()

    for seg in segments:

        a = point(
            seg["x1"],
            seg["y1"]
        )

        b = point(
            seg["x2"],
            seg["y2"]
        )

        if same_point(a, b):
            continue

        key = normalize_segment(
            seg
        )

        if key in seen:
            continue

        seen.add(key)
        result.append(seg)

    return result


# ============================================================
# ROUTE STRATEGY 1:
# DIRECT ORTHOGONAL
# ============================================================

def route_direct_orthogonal(
    terminals
):

    segments = []
    junctions = []
    seen = set()

    if len(terminals) <= 1:

        return (
            segments,
            junctions
        )

    # Choose first terminal as anchor.

    root = terminals[0][
        "point"
    ]

    for terminal in terminals[1:]:

        target = terminal[
            "point"
        ]

        # Prefer horizontal first,
        # then vertical.

        bend = point(
            target["x"],
            root["y"]
        )

        add_segment_unique(
            segments,
            seen,
            root,
            bend
        )

        add_segment_unique(
            segments,
            seen,
            bend,
            target
        )

        if (
            not same_point(
                bend,
                root
            )
            and
            not same_point(
                bend,
                target
            )
        ):

            junctions.append(
                bend
            )

    return (
        clean_segments(
            segments
        ),
        unique_points(
            junctions
        )
    )


# ============================================================
# ROUTE STRATEGY 2:
# SHARED SIGNAL BUS
# ============================================================

def choose_signal_bus_y(
    terminals,
    bus_index=0
):

    ys = [
        t["point"]["y"]
        for t in terminals
    ]

    if not ys:
        return 0.0

    sorted_ys = sorted(ys)

    mid = sorted_ys[
        len(sorted_ys) // 2
    ]

    # Offset avoids running bus exactly
    # through symbol centers.

    direction = (
        -1
        if bus_index % 2 == 0
        else 1
    )

    layer = (
        bus_index // 2
        + 1
    )

    return (
        mid
        + direction
        * BUS_OFFSET
        * layer
    )


def route_shared_bus(
    terminals,
    bus_index=0
):

    segments = []
    junctions = []
    seen = set()

    if len(terminals) <= 1:

        return (
            segments,
            junctions
        )

    xs = [
        t["point"]["x"]
        for t in terminals
    ]

    bus_y = choose_signal_bus_y(
        terminals,
        bus_index
    )

    x_min = min(xs)
    x_max = max(xs)

    left = point(
        x_min,
        bus_y
    )

    right = point(
        x_max,
        bus_y
    )

    add_segment_unique(
        segments,
        seen,
        left,
        right
    )

    for terminal in terminals:

        p = terminal[
            "point"
        ]

        tap = point(
            p["x"],
            bus_y
        )

        add_segment_unique(
            segments,
            seen,
            p,
            tap
        )

        junctions.append(
            tap
        )

    return (
        clean_segments(
            segments
        ),
        unique_points(
            junctions
        )
    )


# ============================================================
# ROUTE STRATEGY 3:
# POWER / GROUND RAIL
# ============================================================

def route_horizontal_rail(
    terminals,
    rail_y
):

    segments = []
    junctions = []
    seen = set()

    if not terminals:

        return (
            segments,
            junctions
        )

    xs = [
        t["point"]["x"]
        for t in terminals
    ]

    x_min = min(xs) - 1.0
    x_max = max(xs) + 1.0

    left = point(
        x_min,
        rail_y
    )

    right = point(
        x_max,
        rail_y
    )

    add_segment_unique(
        segments,
        seen,
        left,
        right
    )

    for terminal in terminals:

        p = terminal[
            "point"
        ]

        tap = point(
            p["x"],
            rail_y
        )

        add_segment_unique(
            segments,
            seen,
            p,
            tap
        )

        junctions.append(
            tap
        )

    return (
        clean_segments(
            segments
        ),
        unique_points(
            junctions
        )
    )


# ============================================================
# ROUTE STRATEGY 4:
# NAMED BIAS / CONTROL BUS
# ============================================================

def choose_named_bus_x(
    terminals,
    layout_bounds,
    bus_index=0
):

    core_points = [

        t["point"]

        for t in terminals

        if t["point"]["x"] > (
            layout_bounds[
                "min_x"
            ]
            + 2.0
        )
    ]

    if core_points:

        x = min(
            p["x"]
            for p in core_points
        )

        return (
            x
            - BUS_OFFSET
            - bus_index
            * BUS_STEP
        )

    return (
        layout_bounds[
            "min_x"
        ]
        - BUS_OFFSET
        - bus_index
        * BUS_STEP
    )


def route_named_bus(
    terminals,
    layout_bounds,
    bus_index=0
):

    segments = []
    junctions = []
    seen = set()

    if len(terminals) <= 1:

        return (
            segments,
            junctions
        )

    bus_x = choose_named_bus_x(
        terminals,
        layout_bounds,
        bus_index
    )

    ys = [
        t["point"]["y"]
        for t in terminals
    ]

    y_min = min(ys)
    y_max = max(ys)

    top = point(
        bus_x,
        y_min
    )

    bottom = point(
        bus_x,
        y_max
    )

    add_segment_unique(
        segments,
        seen,
        top,
        bottom
    )

    for terminal in terminals:

        p = terminal[
            "point"
        ]

        tap = point(
            bus_x,
            p["y"]
        )

        add_segment_unique(
            segments,
            seen,
            p,
            tap
        )

        junctions.append(
            tap
        )

    return (
        clean_segments(
            segments
        ),
        unique_points(
            junctions
        )
    )


# ============================================================
# UNIQUE POINTS
# ============================================================

def unique_points(points):

    result = []
    seen = set()

    for p in points:

        key = point_tuple(p)

        if key in seen:
            continue

        seen.add(key)
        result.append(
            point(
                p["x"],
                p["y"]
            )
        )

    return result


# ============================================================
# RAIL POSITIONS
# ============================================================

def calculate_rail_positions(
    layout,
    anchors
):

    all_points = []

    for component in anchors.values():

        for pin in component.values():

            all_points.append(
                pin["point"]
            )

    if not all_points:

        return {
            "power_y": -5.0,
            "ground_y": 20.0
        }

    min_y = min(
        p["y"]
        for p in all_points
    )

    max_y = max(
        p["y"]
        for p in all_points
    )

    return {

        "power_y":
            round(
                min_y
                - RAIL_MARGIN_Y,
                3
            ),

        "ground_y":
            round(
                max_y
                + RAIL_MARGIN_Y,
                3
            )
    }


# ============================================================
# ROUTE ONE NET
# ============================================================

def route_net(
    net_name,
    net_data,
    hint,
    terminals,
    layout,
    signal_bus_index,
    named_bus_index,
    rail_positions
):

    net_class = hint.get(
        "class",
        "signal"
    )

    strategy = hint.get(
        "strategy",
        "shared_bus"
    )

    # --------------------------------------------------------
    # Power
    # --------------------------------------------------------

    if strategy == "power_rail":

        segments, junctions = (
            route_horizontal_rail(
                terminals,
                rail_positions[
                    "power_y"
                ]
            )
        )

        return {
            "net": net_name,
            "class": net_class,
            "strategy": strategy,
            "terminals": terminals,
            "segments": segments,
            "junctions": junctions,
            "rail_y":
                rail_positions[
                    "power_y"
                ]
        }

    # --------------------------------------------------------
    # Ground
    # --------------------------------------------------------

    if strategy == "ground_rail":

        segments, junctions = (
            route_horizontal_rail(
                terminals,
                rail_positions[
                    "ground_y"
                ]
            )
        )

        return {
            "net": net_name,
            "class": net_class,
            "strategy": strategy,
            "terminals": terminals,
            "segments": segments,
            "junctions": junctions,
            "rail_y":
                rail_positions[
                    "ground_y"
                ]
        }

    # --------------------------------------------------------
    # Bias / control
    # --------------------------------------------------------

    if strategy == "named_bus":

        segments, junctions = (
            route_named_bus(
                terminals,
                layout["bounds"],
                named_bus_index
            )
        )

        return {
            "net": net_name,
            "class": net_class,
            "strategy": strategy,
            "terminals": terminals,
            "segments": segments,
            "junctions": junctions
        }

    # --------------------------------------------------------
    # Small signal net
    # --------------------------------------------------------

    if strategy == "direct_orthogonal":

        segments, junctions = (
            route_direct_orthogonal(
                terminals
            )
        )

        return {
            "net": net_name,
            "class": net_class,
            "strategy": strategy,
            "terminals": terminals,
            "segments": segments,
            "junctions": junctions
        }

    # --------------------------------------------------------
    # Shared signal bus
    # --------------------------------------------------------

    segments, junctions = (
        route_shared_bus(
            terminals,
            signal_bus_index
        )
    )

    return {
        "net": net_name,
        "class": net_class,
        "strategy": "shared_bus",
        "terminals": terminals,
        "segments": segments,
        "junctions": junctions
    }


# ============================================================
# VALIDATION
# ============================================================

def validate_routes(
    layout,
    anchors,
    routes
):

    errors = []
    warnings = []

    components = (
        layout[
            "electrical_truth"
        ][
            "components"
        ]
    )

    nets = (
        layout[
            "electrical_truth"
        ][
            "nets"
        ]
    )

    # --------------------------------------------------------
    # Every parsed component pin must have anchor
    # --------------------------------------------------------

    for component in components:

        name = component["name"]

        for pin_name in component.get(
            "pins",
            {}
        ):

            if (
                name not in anchors
                or
                pin_name not in anchors[
                    name
                ]
            ):

                errors.append(
                    f"Missing pin anchor: "
                    f"{name}.{pin_name}"
                )

    # --------------------------------------------------------
    # Every electrical net must have route object
    # --------------------------------------------------------

    for net_name in nets:

        if net_name not in routes:

            errors.append(
                f"Missing route for net "
                f"{net_name}"
            )

    # --------------------------------------------------------
    # Terminal count must match electrical truth
    # --------------------------------------------------------

    for net_name, net_data in nets.items():

        expected = len(
            net_data[
                "connections"
            ]
        )

        actual = len(
            routes.get(
                net_name,
                {}
            ).get(
                "terminals",
                []
            )
        )

        if expected != actual:

            errors.append(
                f"Net {net_name}: "
                f"expected {expected} terminals, "
                f"routed {actual}"
            )

    # --------------------------------------------------------
    # Check all segments orthogonal
    # --------------------------------------------------------

    for net_name, route in routes.items():

        for seg in route[
            "segments"
        ]:

            horizontal = (
                abs(
                    seg["y1"]
                    - seg["y2"]
                )
                < EPS
            )

            vertical = (
                abs(
                    seg["x1"]
                    - seg["x2"]
                )
                < EPS
            )

            if not (
                horizontal
                or vertical
            ):

                errors.append(
                    f"Non-orthogonal wire "
                    f"on net {net_name}"
                )

    # --------------------------------------------------------
    # Warn about single-terminal nets
    # --------------------------------------------------------

    for net_name, route in routes.items():

        if len(
            route[
                "terminals"
            ]
        ) == 1:

            warnings.append(
                f"Single-terminal net: "
                f"{net_name}"
            )

    return (
        errors,
        warnings
    )


# ============================================================
# BUILD COMPLETE ROUTING PLAN
# ============================================================

def build_routes(layout):

    anchors = build_pin_anchors(
        layout
    )

    nets = (
        layout[
            "electrical_truth"
        ][
            "nets"
        ]
    )

    routing_hints = layout[
        "routing_hints"
    ]

    rail_positions = (
        calculate_rail_positions(
            layout,
            anchors
        )
    )

    routes = {}

    signal_bus_index = 0
    named_bus_index = 0

    # Stable ordering:
    # power, signal, bias/control, ground
    # so renderer has predictable structure.

    class_priority = {
        "power": 0,
        "input": 1,
        "output": 1,
        "signal": 2,
        "bias": 3,
        "control": 3,
        "ground": 4
    }

    ordered_nets = sorted(
        nets.keys(),
        key=lambda net: (
            class_priority.get(
                routing_hints.get(
                    net,
                    {}
                ).get(
                    "class",
                    "signal"
                ),
                10
            ),
            net
        )
    )

    for net_name in ordered_nets:

        net_data = nets[
            net_name
        ]

        hint = routing_hints.get(
            net_name,
            {
                "class": "signal",
                "strategy": "shared_bus"
            }
        )

        terminals = (
            collect_net_terminals(
                net_name,
                net_data,
                anchors
            )
        )

        strategy = hint.get(
            "strategy",
            "shared_bus"
        )

        route = route_net(
            net_name,
            net_data,
            hint,
            terminals,
            layout,
            signal_bus_index,
            named_bus_index,
            rail_positions
        )

        routes[
            net_name
        ] = route

        if strategy == "shared_bus":
            signal_bus_index += 1

        if strategy == "named_bus":
            named_bus_index += 1

    errors, warnings = (
        validate_routes(
            layout,
            anchors,
            routes
        )
    )

    return {

        "source_file":
            layout[
                "source_file"
            ],

        "routing_method":
            "orthogonal_net_router_v1",

        "pin_geometry": {

            "pin_dx":
                PIN_DX,

            "pin_dy":
                PIN_DY
        },

        "rail_positions":
            rail_positions,

        "placements":
            layout[
                "placements"
            ],

        "pin_anchors":
            anchors,

        "routes":
            routes,

        "motifs":
            layout.get(
                "motifs",
                {}
            ),

        "electrical_truth":
            layout[
                "electrical_truth"
            ],

        "validation_errors":
            errors,

        "validation_warnings":
            warnings
    }


# ============================================================
# PROCESS ONE FILE
# ============================================================

def process_file(path):

    print(
        "\n"
        + "=" * 78
    )

    print(
        f"ROUTING: {path.name}"
    )

    print(
        "=" * 78
    )

    layout = json.loads(
        path.read_text()
    )

    result = build_routes(
        layout
    )

    stem = path.stem

    if stem.endswith(
        "_layout"
    ):

        stem = stem[
            :-len("_layout")
        ]

    output_path = (
        ROUTE_DIR
        / f"{stem}_routes.json"
    )

    output_path.write_text(
        json.dumps(
            result,
            indent=2
        )
    )

    print(
        "\nPIN ANCHORS"
    )

    print(
        "-" * 78
    )

    for (
        component,
        pins
    ) in result[
        "pin_anchors"
    ].items():

        for (
            pin_name,
            info
        ) in pins.items():

            p = info[
                "point"
            ]

            print(
                f"{component:<14}"
                f"{pin_name:<4}"
                f"net={info['net']:<12}"
                f"x={p['x']:>7.2f} "
                f"y={p['y']:>7.2f}"
            )

    print(
        "\nNET ROUTES"
    )

    print(
        "-" * 78
    )

    for (
        net_name,
        route
    ) in result[
        "routes"
    ].items():

        print(
            f"{net_name:<15}"
            f"class={route['class']:<10}"
            f"strategy={route['strategy']:<20}"
            f"terminals="
            f"{len(route['terminals']):<3}"
            f"segments="
            f"{len(route['segments']):<3}"
            f"junctions="
            f"{len(route['junctions'])}"
        )

    print(
        "\nRAILS"
    )

    print(
        "-" * 78
    )

    print(
        "Power rail Y :",
        result[
            "rail_positions"
        ][
            "power_y"
        ]
    )

    print(
        "Ground rail Y:",
        result[
            "rail_positions"
        ][
            "ground_y"
        ]
    )

    if result[
        "validation_errors"
    ]:

        print(
            "\nVALIDATION ERRORS"
        )

        print(
            "-" * 78
        )

        for error in result[
            "validation_errors"
        ]:

            print(
                "ERROR:",
                error
            )

    else:

        print(
            "\nRoute validation: PASS"
        )

    if result[
        "validation_warnings"
    ]:

        print(
            "\nWARNINGS"
        )

        print(
            "-" * 78
        )

        for warning in result[
            "validation_warnings"
        ]:

            print(
                "WARNING:",
                warning
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
        LAYOUT_DIR.glob(
            "*_layout.json"
        )
    )

    print(
        "=" * 78
    )

    print(
        "PIN-AWARE ORTHOGONAL "
        "SCHEMATIC ROUTER"
    )

    print(
        "=" * 78
    )

    print(
        f"Layout directory : "
        f"{LAYOUT_DIR}"
    )

    print(
        f"Files found      : "
        f"{len(files)}"
    )

    if not files:

        print(
            "\nNo layout files found."
        )

        print(
            "Run:"
        )

        print(
            "python "
            "src/03_algorithmic_layout.py"
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
        "ROUTING COMPLETE"
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
        f"{ROUTE_DIR}"
    )


if __name__ == "__main__":

    main()