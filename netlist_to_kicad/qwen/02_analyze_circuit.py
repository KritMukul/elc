import json
from itertools import combinations
from pathlib import Path

INPUT = "parsed_circuit.json"
OUTPUT = "circuit_analysis.json"

data = json.loads(Path(INPUT).read_text())

components = data["components"]
nets = data["nets"]

by_name = {
    c["name"]: c
    for c in components
}

MOS_TYPES = {"NMOS", "PMOS", "MOS"}

CORE_TYPES = {
    "NMOS",
    "PMOS",
    "MOS",
    "RESISTOR",
    "CAPACITOR",
    "INDUCTOR",
    "DIODE",
    "SUBCKT"
}

SOURCE_TYPES = {
    "VOLTAGE_SOURCE",
    "CURRENT_SOURCE"
}

core = [
    c for c in components
    if c["type"] in CORE_TYPES
]

sources = [
    c for c in components
    if c["type"] in SOURCE_TYPES
]

mos = [
    c for c in core
    if c["type"] in MOS_TYPES
]


# -------------------------------------------------
# Net classification
# -------------------------------------------------

def normalize(net):
    return str(net).upper()


def classify_net(net):
    n = normalize(net)

    if n in {"0", "GND", "VSS", "VSSA", "VSSD"}:
        return "ground"

    if (
        n == "VDD"
        or n.startswith("VDD")
        or n.startswith("VCC")
    ):
        return "power"

    if (
        "BIAS" in n
        or n.startswith("IB")
        or "VCONT" in n
        or "VREF" in n
    ):
        return "bias"

    if (
        n.startswith("VIN")
        or n.startswith("IN")
    ):
        return "input"

    if (
        n.startswith("VOUT")
        or n.startswith("OUT")
    ):
        return "output"

    return "signal"


net_classes = {
    net: classify_net(net)
    for net in nets
}


# -------------------------------------------------
# Cross-coupled MOS pair
# A.G = B.D and B.G = A.D
# -------------------------------------------------

cross_coupled = []

for a, b in combinations(mos, 2):
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
            "devices": [a["name"], b["name"]],
            "device_type": a["type"],
            "left_right_nodes": [
                ap["D"],
                bp["D"]
            ]
        })


# -------------------------------------------------
# Shared-source MOS pair
# -------------------------------------------------

shared_source = []

for a, b in combinations(mos, 2):
    if a["type"] != b["type"]:
        continue

    if a["pins"]["S"] == b["pins"]["S"]:
        shared_source.append({
            "devices": [a["name"], b["name"]],
            "device_type": a["type"],
            "source_net": a["pins"]["S"]
        })


# -------------------------------------------------
# Shared-gate candidates / current-mirror candidates
# -------------------------------------------------

shared_gate = []

for a, b in combinations(mos, 2):
    if a["type"] != b["type"]:
        continue

    if a["pins"]["G"] == b["pins"]["G"]:
        shared_gate.append({
            "devices": [a["name"], b["name"]],
            "device_type": a["type"],
            "gate_net": a["pins"]["G"]
        })


current_mirrors = []

for pair in shared_gate:
    a = by_name[pair["devices"][0]]
    b = by_name[pair["devices"][1]]

    diode_connected = []

    for d in [a, b]:
        if d["pins"]["G"] == d["pins"]["D"]:
            diode_connected.append(d["name"])

    if diode_connected:
        current_mirrors.append({
            **pair,
            "diode_connected": diode_connected
        })


# -------------------------------------------------
# Passive symmetry candidates
# -------------------------------------------------

passives = [
    c for c in core
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

    av = set(a["pins"].values())
    bv = set(b["pins"].values())

    common = av & bv

    if common:
        passive_pairs.append({
            "devices": [a["name"], b["name"]],
            "device_type": a["type"],
            "common_nets": sorted(common)
        })


# -------------------------------------------------
# Source role classification
# -------------------------------------------------

source_roles = {}

for s in sources:
    connected = list(s["pins"].values())

    role = "testbench"

    if any(classify_net(n) == "power" for n in connected):
        role = "power_supply"

    elif any(classify_net(n) == "ground" for n in connected):
        other = [
            n for n in connected
            if classify_net(n) != "ground"
        ]

        if other:
            oc = classify_net(other[0])

            if oc == "bias":
                role = "bias_source"
            elif oc == "power":
                role = "power_supply"

    if s["type"] == "CURRENT_SOURCE":
        role = "bias_current"

    source_roles[s["name"]] = role


# -------------------------------------------------
# Important nets — NET CENTRIC, not pairwise graph
# -------------------------------------------------

net_info = {}

for net, conns in nets.items():
    net_info[net] = {
        "class": net_classes[net],
        "degree": len(conns),
        "connections": conns
    }


analysis = {
    "source": data["source"],
    "components": components,
    "core_components": core,
    "sources": sources,

    "nets": net_info,

    "structures": {
        "cross_coupled_pairs": cross_coupled,
        "shared_source_pairs": shared_source,
        "shared_gate_pairs": shared_gate,
        "current_mirrors": current_mirrors,
        "passive_pairs": passive_pairs
    },

    "source_roles": source_roles
}

Path(OUTPUT).write_text(
    json.dumps(analysis, indent=2)
)

print("=" * 70)
print("CIRCUIT ANALYSIS COMPLETE")
print("=" * 70)

print(f"\nCore components: {len(core)}")
print(f"Sources        : {len(sources)}")

print("\nCross-coupled pairs:")
for x in cross_coupled:
    print(" ", x)

print("\nShared-source pairs:")
for x in shared_source:
    print(" ", x)

print("\nCurrent mirrors:")
for x in current_mirrors:
    print(" ", x)

print("\nPassive symmetry candidates:")
for x in passive_pairs:
    print(" ", x)

print("\nNet classes:")
for net, info in net_info.items():
    print(
        f"  {net:<15}"
        f"{info['class']:<10}"
        f"degree={info['degree']}"
    )

print(f"\nSaved: {OUTPUT}")