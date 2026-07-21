import json
import math
from pathlib import Path
from collections import defaultdict, deque

# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

GRAPH_DIR = ROOT / "data" / "graphs"
LAYOUT_DIR = ROOT / "data" / "layouts"

LAYOUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LAYOUT CONSTANTS
#
# Coordinates are logical grid coordinates, NOT pixels.
# Renderer will scale these later.
# ============================================================

X_SPACING = 4.0
Y_SPACING = 3.0

CENTER_X = 0.0

LEFT_BIAS_X = -12.0
RIGHT_AUX_X = 12.0

TOP_Y = 0.0
CORE_START_Y = 3.0

DEVICE_TYPES = {
    "NMOS",
    "PMOS",
    "MOS",
    "RESISTOR",
    "CAPACITOR",
    "INDUCTOR",
    "DIODE",
    "BJT",
    "SUBCKT"
}

SOURCE_TYPES = {
    "VOLTAGE_SOURCE",
    "CURRENT_SOURCE"
}


# ============================================================
# BASIC HELPERS
# ============================================================

def get_components(graph):
    return graph["electrical_truth"]["components"]


def get_nets(graph):
    return graph["electrical_truth"]["nets"]


def component_map(components):
    return {
        c["name"]: c
        for c in components
    }


def net_class_map(graph):
    result = {}

    for node in graph["nodes"]:
        if node["node_kind"] == "net":
            result[node["name"]] = node["net_class"]

    return result


def is_mos(component):
    return component["type"] in {
        "NMOS",
        "PMOS",
        "MOS"
    }


def is_passive(component):
    return component["type"] in {
        "RESISTOR",
        "CAPACITOR",
        "INDUCTOR",
        "DIODE"
    }


def get_pin_net(component, pin):
    return component.get("pins", {}).get(pin)


def connected_nets(component):
    return set(
        component.get(
            "pins",
            {}
        ).values()
    )


def non_supply_nets(component, net_classes):
    result = []

    for net in connected_nets(component):
        cls = net_classes.get(net, "signal")

        if cls not in {
            "power",
            "ground"
        }:
            result.append(net)

    return result


# ============================================================
# PLACEMENT STATE
# ============================================================

class PlacementState:

    def __init__(self):

        self.placements = {}

        self.used_positions = set()

        self.reasons = {}

        self.groups = {}

    def occupied(self, x, y):

        key = (
            round(x, 3),
            round(y, 3)
        )

        return key in self.used_positions

    def find_free_position(
        self,
        x,
        y,
        dx=X_SPACING,
        dy=Y_SPACING
    ):
        """
        Find nearby free grid position if requested
        coordinate is already occupied.
        """

        if not self.occupied(x, y):
            return x, y

        candidates = []

        for radius in range(1, 10):

            candidates.extend([
                (x + radius * dx, y),
                (x - radius * dx, y),
                (x, y + radius * dy),
                (x, y - radius * dy),

                (
                    x + radius * dx,
                    y + radius * dy
                ),

                (
                    x - radius * dx,
                    y + radius * dy
                )
            ])

            for cx, cy in candidates:

                if not self.occupied(cx, cy):
                    return cx, cy

        return x, y

    def place(
        self,
        name,
        x,
        y,
        orientation="vertical",
        mirror=False,
        region="core",
        reason="",
        force=False
    ):

        if (
            name in self.placements
            and not force
        ):
            return

        if not force:
            x, y = self.find_free_position(
                x,
                y
            )

        self.placements[name] = {

            "x": round(x, 3),

            "y": round(y, 3),

            "orientation":
                orientation,

            "mirror":
                mirror,

            "region":
                region
        }

        self.used_positions.add(
            (
                round(x, 3),
                round(y, 3)
            )
        )

        if reason:
            self.reasons[name] = reason

    def is_placed(self, name):

        return name in self.placements

    def get(self, name):

        return self.placements.get(name)


# ============================================================
# NET / DEVICE RELATIONSHIP HELPERS
# ============================================================

def devices_on_net(
    net_name,
    nets,
    by_name
):

    result = []

    net = nets.get(net_name)

    if not net:
        return result

    for connection in net["connections"]:

        name = connection["component"]

        if name in by_name:

            result.append(
                (
                    by_name[name],
                    connection["pin"]
                )
            )

    return result


def find_related_devices(
    branch_net,
    nets,
    by_name,
    exclude=None
):

    exclude = set(
        exclude or []
    )

    result = []

    for component, pin in devices_on_net(
        branch_net,
        nets,
        by_name
    ):

        if component["name"] in exclude:
            continue

        result.append(
            (
                component,
                pin
            )
        )

    return result


# ============================================================
# STEP 1:
# FIND MAJOR SYMMETRIC / CROSS-COUPLED CORE
# ============================================================

def place_cross_coupled_cores(
    graph,
    state,
    by_name
):

    motifs = graph["motifs"]

    cross_pairs = motifs.get(
        "cross_coupled_pairs",
        []
    )

    branch_columns = {}

    core_groups = []

    for index, pair in enumerate(
        cross_pairs
    ):

        names = pair["devices"]

        if len(names) != 2:
            continue

        a = by_name[names[0]]
        b = by_name[names[1]]

        # Deterministic ordering based on branch/drain net
        ordered = sorted(
            [a, b],
            key=lambda c:
                str(
                    get_pin_net(
                        c,
                        "D"
                    )
                )
        )

        left = ordered[0]
        right = ordered[1]

        center = (
            index * 12.0
        )

        left_x = (
            center
            - X_SPACING
        )

        right_x = (
            center
            + X_SPACING
        )

        if pair["device_type"] == "PMOS":

            y = CORE_START_Y

        elif pair["device_type"] == "NMOS":

            y = (
                CORE_START_Y
                + 2 * Y_SPACING
            )

        else:

            y = (
                CORE_START_Y
                + Y_SPACING
            )

        state.place(
            left["name"],
            left_x,
            y,
            orientation="vertical",
            mirror=False,
            region="core",
            reason=(
                "left device of "
                "cross-coupled pair"
            )
        )

        state.place(
            right["name"],
            right_x,
            y,
            orientation="vertical",
            mirror=True,
            region="core",
            reason=(
                "right device of "
                "cross-coupled pair"
            )
        )

        left_net = get_pin_net(
            left,
            "D"
        )

        right_net = get_pin_net(
            right,
            "D"
        )

        if left_net:

            branch_columns[
                left_net
            ] = left_x

        if right_net:

            branch_columns[
                right_net
            ] = right_x

        core_groups.append({

            "type":
                "cross_coupled",

            "devices": [
                left["name"],
                right["name"]
            ],

            "left_branch_net":
                left_net,

            "right_branch_net":
                right_net,

            "center_x":
                center
        })

    return (
        branch_columns,
        core_groups
    )


# ============================================================
# STEP 2:
# PLACE MOS DEVICES RELATED TO BRANCH NETS
# ============================================================

def place_branch_mos(
    state,
    components,
    branch_columns
):

    for component in components:

        name = component["name"]

        if state.is_placed(name):
            continue

        if not is_mos(component):
            continue

        pins = component.get(
            "pins",
            {}
        )

        candidates = []

        # Drain relation strongest
        if pins.get("D") in branch_columns:

            candidates.append(
                (
                    0,
                    pins["D"]
                )
            )

        # Gate relation next
        if pins.get("G") in branch_columns:

            candidates.append(
                (
                    1,
                    pins["G"]
                )
            )

        # Source relation weaker
        if pins.get("S") in branch_columns:

            candidates.append(
                (
                    2,
                    pins["S"]
                )
            )

        if not candidates:
            continue

        candidates.sort(
            key=lambda x: x[0]
        )

        branch_net = (
            candidates[0][1]
        )

        x = branch_columns[
            branch_net
        ]

        if component["type"] == "PMOS":

            y = (
                CORE_START_Y
                - Y_SPACING
            )

        elif component["type"] == "NMOS":

            y = (
                CORE_START_Y
                + 2 * Y_SPACING
            )

        else:

            y = (
                CORE_START_Y
                + Y_SPACING
            )

        state.place(
            name,
            x,
            y,
            orientation="vertical",
            region="core",
            reason=(
                f"MOS associated with "
                f"branch net {branch_net}"
            )
        )


# ============================================================
# STEP 3:
# SHARED-SOURCE GROUP ALIGNMENT
# ============================================================

def place_shared_source_groups(
    graph,
    state,
    by_name,
    branch_columns
):

    groups = graph[
        "motifs"
    ].get(
        "shared_source_groups",
        []
    )

    for group in groups:

        names = group[
            "devices"
        ]

        unplaced = [
            n for n in names
            if not state.is_placed(n)
        ]

        if not unplaced:
            continue

        placed = [
            n for n in names
            if state.is_placed(n)
        ]

        # If some group members already placed,
        # align remaining devices horizontally.

        if placed:

            anchor_positions = [
                state.get(n)
                for n in placed
            ]

            avg_y = sum(
                p["y"]
                for p in anchor_positions
            ) / len(anchor_positions)

            xs = [
                p["x"]
                for p in anchor_positions
            ]

            left_start = (
                min(xs)
                - X_SPACING
            )

            for i, name in enumerate(
                unplaced
            ):

                state.place(
                    name,
                    left_start
                    - i * X_SPACING,
                    avg_y,
                    region="core",
                    reason=(
                        "aligned with "
                        "shared-source group"
                    )
                )

        else:

            # No anchor exists.
            # Create symmetric row.

            count = len(names)

            start_x = (
                CENTER_X
                - (
                    (count - 1)
                    * X_SPACING
                    / 2
                )
            )

            component = (
                by_name[
                    names[0]
                ]
            )

            if (
                component["type"]
                == "PMOS"
            ):

                y = CORE_START_Y

            else:

                y = (
                    CORE_START_Y
                    + 2 * Y_SPACING
                )

            for i, name in enumerate(
                names
            ):

                state.place(
                    name,
                    start_x
                    + i * X_SPACING,
                    y,
                    region="core",
                    reason=(
                        "shared-source "
                        "symmetric group"
                    )
                )


# ============================================================
# STEP 4:
# PASSIVE SYMMETRY
# ============================================================

def place_symmetric_passives(
    graph,
    state,
    by_name,
    branch_columns
):

    groups = graph[
        "motifs"
    ].get(
        "passive_symmetry",
        []
    )

    for group in groups:

        names = group[
            "devices"
        ]

        if len(names) != 2:
            continue

        a = by_name[
            names[0]
        ]

        b = by_name[
            names[1]
        ]

        common = set(
            group.get(
                "common_nets",
                []
            )
        )

        def unique_branch_net(
            component
        ):

            candidates = [

                net

                for net in (
                    component
                    .get(
                        "pins",
                        {}
                    )
                    .values()
                )

                if net not in common
            ]

            if candidates:

                return candidates[0]

            return None

        a_net = unique_branch_net(a)
        b_net = unique_branch_net(b)

        # If each passive belongs to known
        # left/right branch, align vertically.

        if (
            a_net in branch_columns
            and
            b_net in branch_columns
        ):

            ax = branch_columns[
                a_net
            ]

            bx = branch_columns[
                b_net
            ]

            # Place below active core
            y = (
                CORE_START_Y
                + 4 * Y_SPACING
            )

            state.place(
                a["name"],
                ax,
                y,
                orientation="vertical",
                region="load",
                reason=(
                    f"symmetric passive on "
                    f"branch {a_net}"
                )
            )

            state.place(
                b["name"],
                bx,
                y,
                orientation="vertical",
                region="load",
                reason=(
                    f"symmetric passive on "
                    f"branch {b_net}"
                )
            )

        else:

            y = (
                CORE_START_Y
                + 4 * Y_SPACING
            )

            state.place(
                a["name"],
                -X_SPACING,
                y,
                region="load",
                reason=(
                    "symmetric passive pair"
                )
            )

            state.place(
                b["name"],
                X_SPACING,
                y,
                region="load",
                reason=(
                    "symmetric passive pair"
                )
            )


# ============================================================
# STEP 5:
# PLACE REMAINING PASSIVES NEAR THEIR MOST IMPORTANT NET
# ============================================================

def place_remaining_passives(
    state,
    components,
    nets,
    branch_columns,
    net_classes
):

    right_count = 0
    bottom_count = 0

    for component in components:

        name = component[
            "name"
        ]

        if state.is_placed(name):
            continue

        if not is_passive(component):
            continue

        cnets = list(
            connected_nets(
                component
            )
        )

        branch_matches = [

            net
            for net in cnets

            if net in branch_columns
        ]

        if branch_matches:

            branch = (
                branch_matches[0]
            )

            x = (
                branch_columns[
                    branch
                ]
                + 2.0
            )

            y = (
                CORE_START_Y
                + 3 * Y_SPACING
                + bottom_count
                * Y_SPACING
            )

            state.place(
                name,
                x,
                y,
                region="load",
                reason=(
                    f"passive connected "
                    f"to branch {branch}"
                )
            )

            bottom_count += 1

            continue

        # Ground-connected passive:
        # generally lower in schematic

        has_ground = any(

            net_classes.get(
                net
            ) == "ground"

            for net in cnets
        )

        if has_ground:

            x = (
                CENTER_X
                + right_count
                * X_SPACING
            )

            y = (
                CORE_START_Y
                + 5 * Y_SPACING
            )

            state.place(
                name,
                x,
                y,
                region="load",
                reason=(
                    "ground-connected "
                    "passive"
                )
            )

            right_count += 1

        else:

            state.place(
                name,
                RIGHT_AUX_X,
                CORE_START_Y
                + right_count
                * Y_SPACING,
                region="auxiliary",
                reason=(
                    "remaining passive"
                )
            )

            right_count += 1


# ============================================================
# STEP 6:
# PLACE SOURCES / BIAS / TESTBENCH ON LEFT
# ============================================================

def classify_source_role(
    component,
    net_classes
):

    nets = list(
        component.get(
            "pins",
            {}
        ).values()
    )

    classes = {
        net_classes.get(
            n,
            "signal"
        )
        for n in nets
    }

    if "power" in classes:

        return "power_source"

    if (
        "bias" in classes
        or
        "control" in classes
    ):

        return "bias_source"

    if (
        component["type"]
        == "CURRENT_SOURCE"
    ):

        return "bias_source"

    if "input" in classes:

        return "input_source"

    return "testbench_source"


def place_sources(
    state,
    components,
    net_classes
):

    sources = [

        c for c in components

        if c["type"]
        in SOURCE_TYPES
    ]

    # Stable semantic ordering

    role_priority = {

        "power_source": 0,

        "bias_source": 1,

        "input_source": 2,

        "testbench_source": 3
    }

    annotated = []

    for source in sources:

        role = classify_source_role(
            source,
            net_classes
        )

        annotated.append(
            (
                role_priority[
                    role
                ],
                role,
                source
            )
        )

    annotated.sort(
        key=lambda x: (
            x[0],
            x[2]["name"]
        )
    )

    for i, (
        _,
        role,
        source
    ) in enumerate(
        annotated
    ):

        y = (
            TOP_Y
            + i * Y_SPACING
        )

        state.place(
            source["name"],
            LEFT_BIAS_X,
            y,
            orientation="vertical",
            region="bias",
            reason=role
        )


# ============================================================
# STEP 7:
# PLACE REMAINING DEVICES
# ============================================================

def place_remaining_devices(
    state,
    components
):

    remaining = [

        c for c in components

        if not state.is_placed(
            c["name"]
        )
    ]

    # MOS first, then subcircuits,
    # then anything else.

    priority = {

        "PMOS": 0,
        "NMOS": 0,
        "MOS": 0,
        "BJT": 1,
        "SUBCKT": 2
    }

    remaining.sort(

        key=lambda c: (

            priority.get(
                c["type"],
                10
            ),

            c["name"]
        )

    )

    for i, component in enumerate(
        remaining
    ):

        state.place(
            component["name"],
            RIGHT_AUX_X,
            CORE_START_Y
            + i * Y_SPACING,
            region="auxiliary",
            reason=(
                "fallback placement "
                "for unclassified device"
            )
        )


# ============================================================
# STEP 8:
# DERIVE NET ROUTING HINTS
#
# Actual wires will be created in 04.
# ============================================================

def build_net_routing_hints(
    graph,
    state,
    net_classes
):

    nets = get_nets(graph)

    hints = {}

    for net_name, net in nets.items():

        cls = net_classes.get(
            net_name,
            "signal"
        )

        degree = net.get(
            "degree",
            0
        )

        connected_positions = []

        for conn in net[
            "connections"
        ]:

            name = conn[
                "component"
            ]

            if state.is_placed(name):

                p = state.get(
                    name
                )

                connected_positions.append({

                    "component":
                        name,

                    "pin":
                        conn["pin"],

                    "x":
                        p["x"],

                    "y":
                        p["y"]
                })

        # --------------------------------------------
        # Routing strategy
        # --------------------------------------------

        if cls == "power":

            strategy = "power_rail"

        elif cls == "ground":

            strategy = "ground_rail"

        elif cls in {
            "bias",
            "control"
        }:

            strategy = "named_bus"

        elif degree >= 6:

            strategy = "shared_bus"

        elif degree <= 3:

            strategy = "direct_orthogonal"

        else:

            strategy = "shared_bus"

        hints[
            net_name
        ] = {

            "class":
                cls,

            "degree":
                degree,

            "strategy":
                strategy,

            "connected_components":
                connected_positions
        }

    return hints


# ============================================================
# STEP 9:
# CALCULATE LAYOUT BOUNDS
# ============================================================

def calculate_bounds(state):

    if not state.placements:

        return {
            "min_x": 0,
            "max_x": 0,
            "min_y": 0,
            "max_y": 0
        }

    xs = [

        p["x"]

        for p in (
            state
            .placements
            .values()
        )
    ]

    ys = [

        p["y"]

        for p in (
            state
            .placements
            .values()
        )
    ]

    return {

        "min_x":
            min(xs),

        "max_x":
            max(xs),

        "min_y":
            min(ys),

        "max_y":
            max(ys)
    }


# ============================================================
# MAIN LAYOUT FUNCTION
# ============================================================

def build_layout(graph):

    components = get_components(
        graph
    )

    nets = get_nets(
        graph
    )

    by_name = component_map(
        components
    )

    net_classes = net_class_map(
        graph
    )

    state = PlacementState()

    # --------------------------------------------------------
    # 1. Strongest structural motif
    # --------------------------------------------------------

    (
        branch_columns,
        core_groups
    ) = place_cross_coupled_cores(

        graph,
        state,
        by_name
    )

    # --------------------------------------------------------
    # 2. Devices attached to branch nets
    # --------------------------------------------------------

    place_branch_mos(

        state,
        components,
        branch_columns
    )

    # --------------------------------------------------------
    # 3. Shared-source structures
    # --------------------------------------------------------

    place_shared_source_groups(

        graph,
        state,
        by_name,
        branch_columns
    )

    # --------------------------------------------------------
    # 4. Symmetric passive structures
    # --------------------------------------------------------

    place_symmetric_passives(

        graph,
        state,
        by_name,
        branch_columns
    )

    # --------------------------------------------------------
    # 5. Remaining passives
    # --------------------------------------------------------

    place_remaining_passives(

        state,
        components,
        nets,
        branch_columns,
        net_classes
    )

    # --------------------------------------------------------
    # 6. Sources / bias
    # --------------------------------------------------------

    place_sources(

        state,
        components,
        net_classes
    )

    # --------------------------------------------------------
    # 7. Anything still unclassified
    # --------------------------------------------------------

    place_remaining_devices(

        state,
        components
    )

    # --------------------------------------------------------
    # 8. Routing hints
    # --------------------------------------------------------

    routing_hints = (
        build_net_routing_hints(

            graph,
            state,
            net_classes
        )
    )

    # --------------------------------------------------------
    # 9. Final output
    # --------------------------------------------------------

    return {

        "source_file":
            graph["source_file"],

        "layout_method":
            "algorithmic_analog_v1",

        "coordinate_system": {

            "type":
                "logical_grid",

            "x_spacing":
                X_SPACING,

            "y_spacing":
                Y_SPACING
        },

        "placements":
            state.placements,

        "placement_reasons":
            state.reasons,

        "branch_columns":
            branch_columns,

        "core_groups":
            core_groups,

        "routing_hints":
            routing_hints,

        "bounds":
            calculate_bounds(
                state
            ),

        # Keep original exact electrical truth.
        # 04 must route from THIS,
        # never infer electrical connectivity
        # from visual proximity.

        "electrical_truth":
            graph[
                "electrical_truth"
            ],

        "motifs":
            graph[
                "motifs"
            ]
    }


# ============================================================
# VALIDATION
# ============================================================

def validate_layout(
    graph,
    layout
):

    errors = []

    components = get_components(
        graph
    )

    component_names = {

        c["name"]

        for c in components
    }

    placed_names = set(

        layout[
            "placements"
        ].keys()
    )

    # Every component must be placed

    missing = (
        component_names
        - placed_names
    )

    if missing:

        errors.append(

            "Unplaced components: "
            + ", ".join(
                sorted(missing)
            )

        )

    # No unknown placed components

    extra = (
        placed_names
        - component_names
    )

    if extra:

        errors.append(

            "Unknown placed components: "
            + ", ".join(
                sorted(extra)
            )

        )

    # Check duplicate coordinates

    positions = defaultdict(
        list
    )

    for (
        name,
        placement
    ) in layout[
        "placements"
    ].items():

        key = (

            placement["x"],
            placement["y"]

        )

        positions[key].append(
            name
        )

    duplicates = {

        pos: names

        for pos, names
        in positions.items()

        if len(names) > 1
    }

    if duplicates:

        for pos, names in (
            duplicates.items()
        ):

            errors.append(

                f"Position collision "
                f"{pos}: "
                + ", ".join(names)

            )

    return errors


# ============================================================
# PROCESS ONE GRAPH
# ============================================================

def process_file(path):

    print(
        "\n"
        + "=" * 76
    )

    print(
        f"ALGORITHMIC LAYOUT: "
        f"{path.name}"
    )

    print(
        "=" * 76
    )

    graph = json.loads(
        path.read_text()
    )

    layout = build_layout(
        graph
    )

    errors = validate_layout(
        graph,
        layout
    )

    layout[
        "validation_errors"
    ] = errors

    stem = path.stem

    if stem.endswith(
        "_graph"
    ):

        stem = stem[
            :-len("_graph")
        ]

    output_path = (

        LAYOUT_DIR
        / f"{stem}_layout.json"

    )

    output_path.write_text(

        json.dumps(
            layout,
            indent=2
        )

    )

    print(
        "\nFINAL COMPONENT PLACEMENT"
    )

    print(
        "-" * 76
    )

    ordered = sorted(

        layout[
            "placements"
        ].items(),

        key=lambda item: (

            item[1]["y"],
            item[1]["x"]
        )

    )

    for name, p in ordered:

        reason = (

            layout[
                "placement_reasons"
            ].get(
                name,
                ""
            )
        )

        print(

            f"{name:<15}"
            f"x={p['x']:>7.2f}  "
            f"y={p['y']:>7.2f}  "
            f"{p['region']:<12}"
            f"{p['orientation']:<10}"
            f"{reason}"

        )

    print(
        "\nBRANCH COLUMNS"
    )

    print(
        "-" * 76
    )

    if layout[
        "branch_columns"
    ]:

        for net, x in layout[
            "branch_columns"
        ].items():

            print(
                f"{net:<20}"
                f"x={x}"
            )

    else:

        print(
            "No explicit symmetric "
            "branch columns detected."
        )

    print(
        "\nROUTING STRATEGIES"
    )

    print(
        "-" * 76
    )

    for net, hint in (
        layout[
            "routing_hints"
        ].items()
    ):

        print(

            f"{net:<15}"
            f"class="
            f"{hint['class']:<10}"
            f"degree="
            f"{hint['degree']:<3}"
            f"strategy="
            f"{hint['strategy']}"

        )

    if errors:

        print(
            "\nLAYOUT VALIDATION ERRORS"
        )

        print(
            "-" * 76
        )

        for error in errors:

            print(
                "ERROR:",
                error
            )

    else:

        print(
            "\nLayout validation: PASS"
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

        GRAPH_DIR.glob(
            "*_graph.json"
        )

    )

    print(
        "=" * 76
    )

    print(
        "ANALOG CIRCUIT "
        "ALGORITHMIC SCHEMATIC LAYOUT"
    )

    print(
        "=" * 76
    )

    print(
        f"Graph directory : "
        f"{GRAPH_DIR}"
    )

    print(
        f"Files found     : "
        f"{len(files)}"
    )

    if not files:

        print(
            "\nNo graph files found."
        )

        print(
            "Run:"
        )

        print(
            "python "
            "src/02_build_circuit_graph.py"
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
        + "=" * 76
    )

    print(
        "LAYOUT GENERATION COMPLETE"
    )

    print(
        "=" * 76
    )

    print(
        f"Successful : {success}"
    )

    print(
        f"Failed     : {failed}"
    )

    print(
        f"Output dir : "
        f"{LAYOUT_DIR}"
    )


if __name__ == "__main__":

    main()