import json
from pathlib import Path
from itertools import combinations

INPUT = "parsed_circuit.json"
OUTPUT = "circuit_analysis.json"

data = json.loads(Path(INPUT).read_text())

components = data["components"]
nets = data["nets"]

# --------------------------------------------------
# Separate actual circuit devices from testbench
# --------------------------------------------------

DUT_TYPES = {
    "NMOS",
    "PMOS",
    "RESISTOR",
    "CAPACITOR",
    "INDUCTOR"
}

SOURCE_TYPES = {
    "VOLTAGE_SOURCE",
    "CURRENT_SOURCE"
}

dut = [
    c for c in components
    if c["type"] in DUT_TYPES
]

sources = [
    c for c in components
    if c["type"] in SOURCE_TYPES
]

mosfets = [
    c for c in dut
    if c["type"] in {"NMOS", "PMOS"}
]

# --------------------------------------------------
# Detect cross-coupled MOS pairs
#
# A.G == B.D and B.G == A.D
# --------------------------------------------------

cross_coupled = []

for a, b in combinations(mosfets, 2):

    if a["type"] != b["type"]:
        continue

    ap = a["pins"]
    bp = b["pins"]

    if (
        ap["G"] == bp["D"]
        and
        bp["G"] == ap["D"]
    ):
        cross_coupled.append({
            "devices": [
                a["name"],
                b["name"]
            ],
            "type": a["type"],
            "nodes": [
                ap["D"],
                bp["D"]
            ]
        })

# --------------------------------------------------
# Detect common-source pairs
# --------------------------------------------------

common_source_pairs = []

for a, b in combinations(mosfets, 2):

    if a["type"] != b["type"]:
        continue

    if (
        a["pins"]["S"] == b["pins"]["S"]
        and
        a["name"] != b["name"]
    ):
        common_source_pairs.append({
            "devices": [
                a["name"],
                b["name"]
            ],
            "type": a["type"],
            "common_source":
                a["pins"]["S"]
        })

# --------------------------------------------------
# Detect symmetric/passive branches
#
# Example:
# net6 -- L1 -- VSS
# net1 -- L0 -- VSS
# --------------------------------------------------

passives = [
    c for c in dut
    if c["type"] in {
        "RESISTOR",
        "CAPACITOR",
        "INDUCTOR"
    }
]

passive_pairs = []

for a, b in combinations(passives, 2):

    if a["type"] != b["type"]:
        continue

    ap = a["pins"]
    bp = b["pins"]

    a_nodes = set(ap.values())
    b_nodes = set(bp.values())

    common = a_nodes.intersection(b_nodes)

    if common:

        passive_pairs.append({
            "devices": [
                a["name"],
                b["name"]
            ],
            "type": a["type"],
            "common_nodes":
                list(common)
        })

# --------------------------------------------------
# Determine important nets
# --------------------------------------------------

power_nets = []

for candidate in [
    "VDD",
    "VSS",
    "GND",
    "0"
]:
    if candidate in nets:
        power_nets.append(candidate)

# Nets ranked by number of electrical connections

net_importance = sorted(
    [
        {
            "net": net,
            "connections": len(conns)
        }
        for net, conns in nets.items()
    ],
    key=lambda x: x["connections"],
    reverse=True
)

# --------------------------------------------------
# Generate placement hints
# --------------------------------------------------

placement_hints = {}

for c in dut:

    name = c["name"]
    t = c["type"]

    if t == "PMOS":

        placement_hints[name] = {
            "region": "upper",
            "orientation": "vertical"
        }

    elif t == "NMOS":

        placement_hints[name] = {
            "region": "middle",
            "orientation": "vertical"
        }

    elif t == "INDUCTOR":

        placement_hints[name] = {
            "region": "lower",
            "orientation": "vertical"
        }

    elif t == "CAPACITOR":

        placement_hints[name] = {
            "region": "lower_or_side",
            "orientation": "vertical"
        }

    else:

        placement_hints[name] = {
            "region": "auto",
            "orientation": "auto"
        }

# --------------------------------------------------
# Save analysis
# --------------------------------------------------

analysis = {
    "source": data["source"],

    "dut_components": dut,

    "testbench_sources": sources,

    "nets": nets,

    "structures": {
        "cross_coupled_pairs":
            cross_coupled,

        "common_source_pairs":
            common_source_pairs,

        "passive_pairs":
            passive_pairs
    },

    "power_nets":
        power_nets,

    "net_importance":
        net_importance,

    "placement_hints":
        placement_hints
}

Path(OUTPUT).write_text(
    json.dumps(
        analysis,
        indent=2
    )
)

# --------------------------------------------------
# Console report
# --------------------------------------------------

print("=" * 65)
print("CIRCUIT ANALYSIS COMPLETE")
print("=" * 65)

print(
    f"\nDUT components: "
    f"{len(dut)}"
)

for c in dut:

    print(
        f"  {c['name']:<8}"
        f"{c['type']:<14}"
        f"{c['pins']}"
    )

print(
    f"\nTestbench/Bias sources: "
    f"{len(sources)}"
)

for c in sources:

    print(
        f"  {c['name']:<14}"
        f"{c['type']}"
    )

print("\nDetected cross-coupled pairs:")

if cross_coupled:

    for x in cross_coupled:

        print(
            "  ",
            x["devices"],
            "nodes =",
            x["nodes"]
        )

else:

    print("   None")

print("\nDetected common-source pairs:")

if common_source_pairs:

    for x in common_source_pairs:

        print(
            "  ",
            x["devices"],
            "source =",
            x["common_source"]
        )

else:

    print("   None")

print("\nPassive symmetric candidates:")

if passive_pairs:

    for x in passive_pairs:

        print(
            "  ",
            x["devices"],
            "common =",
            x["common_nodes"]
        )

else:

    print("   None")

print("\nImportant nets:")

for x in net_importance:

    print(
        f"  {x['net']:<12} "
        f"{x['connections']} connections"
    )

print(
    f"\nSaved: {OUTPUT}"
)