import json
import torch
from pathlib import Path

INPUT = "circuit_analysis.json"
OUTPUT = "optimized_layout.json"

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 65)
print("GPU SCHEMATIC LAYOUT OPTIMIZER")
print("=" * 65)
print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

data = json.loads(Path(INPUT).read_text())

components = data["dut_components"]
nets = data["nets"]
structures = data["structures"]

names = [c["name"] for c in components]
idx = {name: i for i, name in enumerate(names)}

N = len(names)

# -------------------------------------------------
# Initial human-style placement
# -------------------------------------------------

initial = {
    # Cross-coupled PMOS pair
    "XM0": [-3.0, 3.0],
    "XM3": [ 3.0, 3.0],

    # NMOS devices below
    "XM4": [-3.0, 0.0],
    "XM1": [ 3.0, 0.0],

    # Symmetric inductors
    "L1": [-3.0, -3.0],
    "L0": [ 3.0, -3.0],

    # Load capacitor on output side
    "CL": [6.0, -1.5]
}

coords = []

for name in names:
    coords.append(
        initial.get(name, [0.0, 0.0])
    )

pos = torch.tensor(
    coords,
    dtype=torch.float32,
    device=device,
    requires_grad=True
)

# -------------------------------------------------
# Build electrical connectivity edges
# -------------------------------------------------

edges = set()

for net, connections in nets.items():

    dut_connections = [
        x["component"]
        for x in connections
        if x["component"] in idx
    ]

    for i in range(len(dut_connections)):

        for j in range(i + 1, len(dut_connections)):

            a = idx[dut_connections[i]]
            b = idx[dut_connections[j]]

            if a != b:
                edges.add(
                    tuple(sorted((a, b)))
                )

edges = list(edges)

print("\nElectrical placement edges:")

for a, b in edges:
    print(
        f"  {names[a]} <-> {names[b]}"
    )

# -------------------------------------------------
# Optimization
# -------------------------------------------------

optimizer = torch.optim.Adam(
    [pos],
    lr=0.03
)

initial_tensor = torch.tensor(
    coords,
    dtype=torch.float32,
    device=device
)

for step in range(2000):

    optimizer.zero_grad()

    # ---------------------------------------------
    # 1. Connected devices should remain nearby
    # ---------------------------------------------

    wire_loss = torch.tensor(
        0.0,
        device=device
    )

    for a, b in edges:

        dist = torch.norm(
            pos[a] - pos[b]
        )

        wire_loss += dist

    # ---------------------------------------------
    # 2. Prevent overlapping components
    # ---------------------------------------------

    overlap_loss = torch.tensor(
        0.0,
        device=device
    )

    for i in range(N):

        for j in range(i + 1, N):

            dist = torch.norm(
                pos[i] - pos[j]
            )

            overlap_loss += torch.relu(
                2.3 - dist
            ) ** 2

    # ---------------------------------------------
    # 3. Preserve human-designed starting structure
    # ---------------------------------------------

    anchor_loss = torch.mean(
        (pos - initial_tensor) ** 2
    )

    # ---------------------------------------------
    # 4. Symmetry constraints
    # ---------------------------------------------

    symmetry_loss = torch.tensor(
        0.0,
        device=device
    )

    symmetric_pairs = [
        ("XM0", "XM3"),
        ("XM4", "XM1"),
        ("L1", "L0")
    ]

    for left, right in symmetric_pairs:

        if left in idx and right in idx:

            a = pos[idx[left]]
            b = pos[idx[right]]

            # Same vertical level
            symmetry_loss += (
                a[1] - b[1]
            ) ** 2

            # Mirror around x = 0
            symmetry_loss += (
                a[0] + b[0]
            ) ** 2

    # ---------------------------------------------
    # Total objective
    # ---------------------------------------------

    loss = (
        0.15 * wire_loss
        + 8.0 * overlap_loss
        + 2.0 * anchor_loss
        + 5.0 * symmetry_loss
    )

    loss.backward()
    optimizer.step()

    if step % 250 == 0:

        print(
            f"Step {step:4d} | "
            f"Loss {loss.item():.4f}"
        )

# -------------------------------------------------
# Snap coordinates to clean schematic grid
# -------------------------------------------------

result = {}

final_pos = (
    pos.detach()
    .cpu()
    .numpy()
)

for i, name in enumerate(names):

    x = round(
        float(final_pos[i][0]) * 2
    ) / 2

    y = round(
        float(final_pos[i][1]) * 2
    ) / 2

    result[name] = {
        "x": x,
        "y": y
    }

# -------------------------------------------------
# Add structural information for renderer
# -------------------------------------------------

output = {
    "device":
        str(device),

    "positions":
        result,

    "structures":
        structures,

    "components":
        components,

    "nets":
        nets
}

Path(OUTPUT).write_text(
    json.dumps(
        output,
        indent=2
    )
)

print("\n" + "=" * 65)
print("FINAL PLACEMENT")
print("=" * 65)

for name, p in result.items():

    print(
        f"{name:<6} "
        f"x={p['x']:>5} "
        f"y={p['y']:>5}"
    )

print(
    f"\nSaved: {OUTPUT}"
)