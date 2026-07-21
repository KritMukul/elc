import json
from pathlib import Path
from collections import Counter, defaultdict

# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

PARSED_DIR = ROOT / "data" / "parsed"
GRAPH_DIR = ROOT / "data" / "graphs"

GRAPH_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# DEVICE / PIN VOCABULARY
# ============================================================

DEVICE_TYPES = [
    "NMOS",
    "PMOS",
    "MOS",
    "RESISTOR",
    "CAPACITOR",
    "INDUCTOR",
    "VOLTAGE_SOURCE",
    "CURRENT_SOURCE",
    "DIODE",
    "BJT",
    "SUBCKT",
    "UNKNOWN"
]

DEVICE_TYPE_TO_ID = {
    name: i
    for i, name in enumerate(DEVICE_TYPES)
}


PIN_TYPES = [
    "D",
    "G",
    "S",
    "B",
    "1",
    "2",
    "+",
    "-",
    "A",
    "K",
    "C",
    "E",
    "UNKNOWN"
]

PIN_TYPE_TO_ID = {
    name: i
    for i, name in enumerate(PIN_TYPES)
}


# ============================================================
# NET CLASSIFICATION
# ============================================================

def classify_net(net_name):
    """
    Semantic classification of a net.

    These classes are useful later for schematic placement:
        VDD       -> power
        VSS / 0   -> ground
        IB1       -> bias
        VCONT1    -> control
        net1      -> signal
    """

    name = str(net_name).strip()
    upper = name.upper()

    # Ground
    if upper in {
        "0",
        "GND",
        "GROUND",
        "VSS",
        "VSSA",
        "VSSD"
    }:
        return "ground"

    # Positive supply
    if (
        upper == "VDD"
        or upper.startswith("VDD")
        or upper == "VCC"
        or upper.startswith("VCC")
    ):
        return "power"

    # Bias
    if (
        "BIAS" in upper
        or upper.startswith("IB")
        or upper.startswith("VB")
        or "VREF" in upper
    ):
        return "bias"

    # Control
    if (
        "VCONT" in upper
        or "CTRL" in upper
        or "CONTROL" in upper
    ):
        return "control"

    # Input
    if (
        upper.startswith("VIN")
        or upper.startswith("IN_")
        or upper in {"IN", "INPUT"}
    ):
        return "input"

    # Output
    if (
        upper.startswith("VOUT")
        or upper.startswith("OUT_")
        or upper in {"OUT", "OUTPUT"}
    ):
        return "output"

    return "signal"


NET_CLASSES = [
    "signal",
    "power",
    "ground",
    "bias",
    "control",
    "input",
    "output"
]

NET_CLASS_TO_ID = {
    name: i
    for i, name in enumerate(NET_CLASSES)
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def pin_type_id(pin):
    return PIN_TYPE_TO_ID.get(
        str(pin).upper(),
        PIN_TYPE_TO_ID["UNKNOWN"]
    )


def device_type_id(device_type):
    return DEVICE_TYPE_TO_ID.get(
        device_type,
        DEVICE_TYPE_TO_ID["UNKNOWN"]
    )


def get_component_degree(component, nets):
    """
    Number of unique nets touched by a component.
    """

    return len(
        set(
            component.get(
                "pins",
                {}
            ).values()
        )
    )


# ============================================================
# STRUCTURAL FEATURE EXTRACTION
# ============================================================

def detect_cross_coupled_pairs(components):
    """
    Detect MOS pair:

        A.G = B.D
        B.G = A.D

    Example:
        XM0.G = net1
        XM0.D = net6

        XM3.G = net6
        XM3.D = net1
    """

    mos = [
        c for c in components
        if c["type"] in {
            "NMOS",
            "PMOS",
            "MOS"
        }
    ]

    pairs = []

    for i in range(len(mos)):

        for j in range(i + 1, len(mos)):

            a = mos[i]
            b = mos[j]

            # Usually symmetry is strongest for same MOS type
            if a["type"] != b["type"]:
                continue

            ap = a.get("pins", {})
            bp = b.get("pins", {})

            required = {"D", "G"}

            if not (
                required.issubset(ap)
                and required.issubset(bp)
            ):
                continue

            if (
                ap["G"] == bp["D"]
                and
                bp["G"] == ap["D"]
            ):

                pairs.append({
                    "devices": [
                        a["name"],
                        b["name"]
                    ],

                    "device_type":
                        a["type"],

                    "branch_nets": [
                        ap["D"],
                        bp["D"]
                    ]
                })

    return pairs


def detect_shared_source_pairs(components):
    """
    Same-type MOS devices sharing source net.
    """

    mos = [
        c for c in components
        if c["type"] in {
            "NMOS",
            "PMOS",
            "MOS"
        }
    ]

    groups = defaultdict(list)

    for c in mos:

        source = (
            c.get("pins", {})
            .get("S")
        )

        if source is not None:

            key = (
                c["type"],
                source
            )

            groups[key].append(
                c["name"]
            )

    result = []

    for (
        device_type,
        source_net
    ), devices in groups.items():

        if len(devices) >= 2:

            result.append({
                "devices": devices,
                "device_type":
                    device_type,
                "source_net":
                    source_net
            })

    return result


def detect_shared_gate_groups(components):
    """
    MOS devices sharing gate net.
    Useful for mirrors / bias structures.
    """

    mos = [
        c for c in components
        if c["type"] in {
            "NMOS",
            "PMOS",
            "MOS"
        }
    ]

    groups = defaultdict(list)

    for c in mos:

        gate = (
            c.get("pins", {})
            .get("G")
        )

        if gate is not None:

            key = (
                c["type"],
                gate
            )

            groups[key].append(
                c["name"]
            )

    result = []

    for (
        device_type,
        gate_net
    ), devices in groups.items():

        if len(devices) >= 2:

            result.append({
                "devices": devices,
                "device_type":
                    device_type,
                "gate_net":
                    gate_net
            })

    return result


def detect_passive_symmetry(components):
    """
    Finds same-type passive devices sharing one net.

    Example:

        L1 net6 VSS
        L0 net1 VSS

    -> possible symmetric branches.
    """

    passives = [
        c for c in components
        if c["type"] in {
            "RESISTOR",
            "CAPACITOR",
            "INDUCTOR"
        }
    ]

    result = []

    for i in range(len(passives)):

        for j in range(
            i + 1,
            len(passives)
        ):

            a = passives[i]
            b = passives[j]

            if a["type"] != b["type"]:
                continue

            a_nets = set(
                a.get(
                    "pins",
                    {}
                ).values()
            )

            b_nets = set(
                b.get(
                    "pins",
                    {}
                ).values()
            )

            common = (
                a_nets
                & b_nets
            )

            if common:

                result.append({
                    "devices": [
                        a["name"],
                        b["name"]
                    ],

                    "device_type":
                        a["type"],

                    "common_nets":
                        sorted(common)
                })

    return result


# ============================================================
# BUILD BIPARTITE CIRCUIT GRAPH
# ============================================================

def build_graph(parsed):
    """
    Builds:

        Device nodes
              ↕
        typed pin edges
              ↕
          Net nodes

    We DO NOT create:

        XM0 -- XM1
        XM0 -- CL
        XM1 -- CL

    merely because they share a net.

    A multi-terminal electrical net remains ONE net node.
    """

    components = parsed["components"]
    nets = parsed["nets"]

    nodes = []

    edges = []

    device_node_ids = {}

    net_node_ids = {}

    node_id = 0

    # --------------------------------------------------------
    # DEVICE NODES
    # --------------------------------------------------------

    for component in components:

        name = component["name"]

        ctype = component.get(
            "type",
            "UNKNOWN"
        )

        pins = component.get(
            "pins",
            {}
        )

        node = {

            "id":
                node_id,

            "node_kind":
                "device",

            "name":
                name,

            "device_type":
                ctype,

            "device_type_id":
                device_type_id(
                    ctype
                ),

            "degree":
                get_component_degree(
                    component,
                    nets
                ),

            "pin_count":
                len(pins),

            "model":
                component.get(
                    "model"
                ),

            "value":
                component.get(
                    "value"
                ),

            "params":
                component.get(
                    "params",
                    {}
                )
        }

        nodes.append(
            node
        )

        device_node_ids[
            name
        ] = node_id

        node_id += 1

    # --------------------------------------------------------
    # NET NODES
    # --------------------------------------------------------

    for net_name, net_data in nets.items():

        net_class = classify_net(
            net_name
        )

        node = {

            "id":
                node_id,

            "node_kind":
                "net",

            "name":
                net_name,

            "net_class":
                net_class,

            "net_class_id":
                NET_CLASS_TO_ID[
                    net_class
                ],

            "degree":
                net_data[
                    "degree"
                ]
        }

        nodes.append(
            node
        )

        net_node_ids[
            net_name
        ] = node_id

        node_id += 1

    # --------------------------------------------------------
    # DEVICE <-> NET EDGES
    # --------------------------------------------------------

    edge_id = 0

    for component in components:

        device_name = (
            component["name"]
        )

        device_id = (
            device_node_ids[
                device_name
            ]
        )

        for (
            pin_name,
            net_name
        ) in component.get(
            "pins",
            {}
        ).items():

            if (
                net_name
                not in net_node_ids
            ):
                continue

            net_id = (
                net_node_ids[
                    net_name
                ]
            )

            edge = {

                "id":
                    edge_id,

                "source":
                    device_id,

                "target":
                    net_id,

                "device":
                    device_name,

                "net":
                    net_name,

                "pin":
                    pin_name,

                "pin_type_id":
                    pin_type_id(
                        pin_name
                    )
            }

            edges.append(
                edge
            )

            edge_id += 1

    # --------------------------------------------------------
    # STRUCTURAL MOTIFS
    # --------------------------------------------------------

    motifs = {

        "cross_coupled_pairs":
            detect_cross_coupled_pairs(
                components
            ),

        "shared_source_groups":
            detect_shared_source_pairs(
                components
            ),

        "shared_gate_groups":
            detect_shared_gate_groups(
                components
            ),

        "passive_symmetry":
            detect_passive_symmetry(
                components
            )
    }

    # --------------------------------------------------------
    # GRAPH STATISTICS
    # --------------------------------------------------------

    device_counts = Counter(

        c.get(
            "type",
            "UNKNOWN"
        )

        for c in components
    )

    net_class_counts = Counter(

        classify_net(net)

        for net in nets
    )

    statistics = {

        "device_nodes":
            len(components),

        "net_nodes":
            len(nets),

        "total_nodes":
            len(nodes),

        "edges":
            len(edges),

        "device_type_counts":
            dict(
                device_counts
            ),

        "net_class_counts":
            dict(
                net_class_counts
            )
    }

    return {

        "source_file":
            parsed[
                "source_file"
            ],

        "graph_type":
            "device_net_bipartite",

        "nodes":
            nodes,

        "edges":
            edges,

        "device_node_ids":
            device_node_ids,

        "net_node_ids":
            net_node_ids,

        "motifs":
            motifs,

        "statistics":
            statistics,

        # Keep exact original circuit information.
        # Later routing must NEVER guess connectivity.

        "electrical_truth": {

            "components":
                components,

            "nets":
                nets
        }
    }


# ============================================================
# VALIDATE GRAPH
# ============================================================

def validate_graph(graph):
    """
    Basic consistency checks.
    """

    errors = []

    nodes = graph["nodes"]
    edges = graph["edges"]

    valid_ids = {
        node["id"]
        for node in nodes
    }

    # Every edge must point to valid nodes

    for edge in edges:

        if (
            edge["source"]
            not in valid_ids
        ):

            errors.append(
                f"Invalid source node "
                f"in edge {edge['id']}"
            )

        if (
            edge["target"]
            not in valid_ids
        ):

            errors.append(
                f"Invalid target node "
                f"in edge {edge['id']}"
            )

    # Each electrical pin should correspond
    # to exactly one graph edge.

    expected_edges = 0

    for component in (
        graph[
            "electrical_truth"
        ][
            "components"
        ]
    ):

        expected_edges += len(
            component.get(
                "pins",
                {}
            )
        )

    if (
        expected_edges
        != len(edges)
    ):

        errors.append(

            f"Expected "
            f"{expected_edges} "
            f"pin edges but graph "
            f"contains {len(edges)}"

        )

    return errors


# ============================================================
# PROCESS ONE PARSED JSON
# ============================================================

def process_file(path):

    print(
        "\n"
        + "=" * 72
    )

    print(
        f"BUILDING GRAPH: "
        f"{path.name}"
    )

    print(
        "=" * 72
    )

    parsed = json.loads(
        path.read_text()
    )

    graph = build_graph(
        parsed
    )

    errors = validate_graph(
        graph
    )

    graph[
        "validation_errors"
    ] = errors

    # Remove "_parsed" from output stem

    stem = path.stem

    if stem.endswith(
        "_parsed"
    ):

        stem = stem[
            :-len("_parsed")
        ]

    output_path = (

        GRAPH_DIR
        / f"{stem}_graph.json"

    )

    output_path.write_text(

        json.dumps(
            graph,
            indent=2
        )

    )

    stats = graph[
        "statistics"
    ]

    print(
        f"Device nodes : "
        f"{stats['device_nodes']}"
    )

    print(
        f"Net nodes    : "
        f"{stats['net_nodes']}"
    )

    print(
        f"Total nodes  : "
        f"{stats['total_nodes']}"
    )

    print(
        f"Pin edges    : "
        f"{stats['edges']}"
    )

    print(
        "\nDevice types:"
    )

    for (
        device_type,
        count
    ) in stats[
        "device_type_counts"
    ].items():

        print(
            f"  "
            f"{device_type:<20}"
            f"{count}"
        )

    print(
        "\nNet classes:"
    )

    for (
        net_class,
        count
    ) in stats[
        "net_class_counts"
    ].items():

        print(
            f"  "
            f"{net_class:<20}"
            f"{count}"
        )

    print(
        "\nDetected motifs:"
    )

    motifs = graph[
        "motifs"
    ]

    print(
        "  Cross-coupled:"
    )

    for item in motifs[
        "cross_coupled_pairs"
    ]:

        print(
            "   ",
            item
        )

    print(
        "  Shared-source:"
    )

    for item in motifs[
        "shared_source_groups"
    ]:

        print(
            "   ",
            item
        )

    print(
        "  Shared-gate:"
    )

    for item in motifs[
        "shared_gate_groups"
    ]:

        print(
            "   ",
            item
        )

    print(
        "  Passive symmetry:"
    )

    for item in motifs[
        "passive_symmetry"
    ]:

        print(
            "   ",
            item
        )

    if errors:

        print(
            "\nVALIDATION ERRORS:"
        )

        for error in errors:

            print(
                "  ERROR:",
                error
            )

    else:

        print(
            "\nGraph validation: PASS"
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

        PARSED_DIR.glob(
            "*_parsed.json"
        )

    )

    print(
        "=" * 72
    )

    print(
        "PARSED CIRCUIT → "
        "BIPARTITE CIRCUIT GRAPH"
    )

    print(
        "=" * 72
    )

    print(
        f"Parsed directory: "
        f"{PARSED_DIR}"
    )

    print(
        f"Files found: "
        f"{len(files)}"
    )

    if not files:

        print(
            "\nNo parsed circuit files found."
        )

        print(
            "Run first:"
        )

        print(
            "python "
            "src/01_parse_spice.py"
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
        + "=" * 72
    )

    print(
        "GRAPH BUILD COMPLETE"
    )

    print(
        "=" * 72
    )

    print(
        f"Successful : {success}"
    )

    print(
        f"Failed     : {failed}"
    )

    print(
        f"Output dir : "
        f"{GRAPH_DIR}"
    )


if __name__ == "__main__":

    main()