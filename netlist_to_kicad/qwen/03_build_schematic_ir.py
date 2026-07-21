import json
from pathlib import Path

INPUT = "circuit_analysis.json"
OUTPUT = "schematic_ir.json"

data = json.loads(Path(INPUT).read_text())

core = data["core_components"]
sources = data["sources"]
nets = data["nets"]
structures = data["structures"]
source_roles = data["source_roles"]

by_name = {
    c["name"]: c
    for c in data["components"]
}

placement = {}
orientation = {}

used = set()


def place(name, x, y, orient="vertical"):
    placement[name] = {
        "x": x,
        "y": y
    }

    orientation[name] = orient
    used.add(name)


# =================================================
# 1. Strongest motif: cross-coupled pairs
# =================================================

cross_pairs = structures["cross_coupled_pairs"]

for pair_index, pair in enumerate(cross_pairs):
    a_name, b_name = pair["devices"]

    a = by_name[a_name]
    b = by_name[b_name]

    base_x = pair_index * 12

    # Deterministic left/right ordering
    # based on drain-node name
    ordered = sorted(
        [a, b],
        key=lambda c: c["pins"]["D"]
    )

    left = ordered[0]
    right = ordered[1]

    y = 2.0 if pair["device_type"] == "PMOS" else 0.0

    place(
        left["name"],
        base_x - 3.0,
        y
    )

    place(
        right["name"],
        base_x + 3.0,
        y
    )


# =================================================
# 2. Match MOS devices beneath/above branch nets
#
# If MOS gate/drain connects to one of the
# cross-coupled branch nets, align it with branch.
# =================================================

branch_columns = {}

for pair in cross_pairs:
    for dev_name in pair["devices"]:
        dev = by_name[dev_name]
        branch_net = dev["pins"]["D"]

        if dev_name in placement:
            branch_columns[branch_net] = (
                placement[dev_name]["x"]
            )


for c in core:
    if c["name"] in used:
        continue

    if c["type"] not in {"NMOS", "PMOS", "MOS"}:
        continue

    candidate_nets = [
        c["pins"].get("G"),
        c["pins"].get("D")
    ]

    matched_x = None

    for net in candidate_nets:
        if net in branch_columns:
            matched_x = branch_columns[net]
            break

    if matched_x is not None:
        if c["type"] == "NMOS":
            place(
                c["name"],
                matched_x,
                6.0
            )
        else:
            place(
                c["name"],
                matched_x,
                -2.0
            )


# =================================================
# 3. Symmetric passive pairs
# =================================================

for pair in structures["passive_pairs"]:
    a_name, b_name = pair["devices"]

    if a_name in used or b_name in used:
        continue

    a = by_name[a_name]
    b = by_name[b_name]

    # Try to align passive with its non-common net
    common = set(pair["common_nets"])

    def branch_net(comp):
        candidates = [
            n for n in comp["pins"].values()
            if n not in common
        ]

        return (
            candidates[0]
            if candidates
            else None
        )

    an = branch_net(a)
    bn = branch_net(b)

    ax = branch_columns.get(an, -3.0)
    bx = branch_columns.get(bn, 3.0)

    place(
        a_name,
        ax,
        9.5
    )

    place(
        b_name,
        bx,
        9.5
    )


# =================================================
# 4. Remaining passives
# =================================================

right_side_x = (
    max(
        [p["x"] for p in placement.values()],
        default=3.0
    )
    + 4.0
)

passive_count = 0

for c in core:
    if c["name"] in used:
        continue

    if c["type"] in {
        "CAPACITOR",
        "RESISTOR",
        "INDUCTOR",
        "DIODE"
    }:
        connected = list(c["pins"].values())

        matched = [
            n for n in connected
            if n in branch_columns
        ]

        if matched:
            x = branch_columns[matched[0]] + 3.5
        else:
            x = right_side_x

        y = 5.0 + passive_count * 2.5

        place(
            c["name"],
            x,
            y
        )

        passive_count += 1


# =================================================
# 5. Remaining core devices
# =================================================

remaining_count = 0

for c in core:
    if c["name"] in used:
        continue

    place(
        c["name"],
        right_side_x + 4,
        2 + remaining_count * 3
    )

    remaining_count += 1


# =================================================
# 6. Testbench / bias sources
# Place on left side
# =================================================

left_x = (
    min(
        [p["x"] for p in placement.values()],
        default=-3.0
    )
    - 6.0
)

source_count = 0

for s in sources:
    role = source_roles.get(
        s["name"],
        "testbench"
    )

    place(
        s["name"],
        left_x,
        1.0 + source_count * 3.0
    )

    source_count += 1


# =================================================
# 7. Net routing strategy
# =================================================

net_plan = {}

for net, info in nets.items():
    cls = info["class"]

    if cls == "power":
        strategy = "power_label"

    elif cls == "ground":
        strategy = "ground_label"

    elif cls == "bias":
        strategy = "named_net"

    elif info["degree"] >= 4:
        strategy = "named_net"

    else:
        strategy = "direct_or_bus"

    net_plan[net] = {
        "class": cls,
        "strategy": strategy,
        "connections": info["connections"]
    }


# =================================================
# Final schematic IR
# =================================================

result = {
    "source": data["source"],

    "components": data["components"],

    "placement": placement,

    "orientation": orientation,

    "nets": net_plan,

    "structures": structures,

    "render_policy": {
        "power_at_top": True,
        "ground_at_bottom": True,
        "orthogonal_wires": True,
        "show_testbench_sources": True,
        "show_device_parameters": True,
        "prefer_net_labels_for_high_fanout": True
    }
}

Path(OUTPUT).write_text(
    json.dumps(
        result,
        indent=2
    )
)

print("=" * 70)
print("SCHEMATIC IR GENERATED")
print("=" * 70)

print("\nPlacement:")

for name, p in placement.items():
    print(
        f"{name:<15}"
        f"x={p['x']:>6} "
        f"y={p['y']:>6}"
    )

print("\nNet routing strategies:")

for net, p in net_plan.items():
    print(
        f"{net:<15}"
        f"{p['class']:<10}"
        f"{p['strategy']}"
    )

print(f"\nSaved: {OUTPUT}")