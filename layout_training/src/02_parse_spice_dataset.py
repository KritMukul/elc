import json
import re
from pathlib import Path
from collections import Counter

# ============================================================
# PATHS
# ============================================================

WORKSPACE = Path("/workspace")
DATASET_DIR = WORKSPACE / "masala-chai-dataset-new"

MAPPING_FILE = DATASET_DIR / "data_mapping.json"

PROJECT_DIR = WORKSPACE / "layout_training"
DATA_DIR = PROJECT_DIR / "data"

OUTPUT_DIR = DATA_DIR / "parsed_spice"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_FILE = DATA_DIR / "spice_parse_summary.json"
MASTER_FILE = DATA_DIR / "parsed_dataset.json"


# ============================================================
# BASIC HELPERS
# ============================================================

def read_text(path):
    try:
        return path.read_text(
            encoding="utf-8",
            errors="ignore"
        )
    except Exception:
        return ""


def resolve_path(relative_path):
    if not relative_path:
        return None

    p = Path(relative_path)

    if p.is_absolute():
        return p

    return DATASET_DIR / p


def clean_lines(text):
    """
    Clean markdown wrappers, comments and blank lines,
    while preserving actual SPICE statements.
    """

    result = []

    for raw in text.splitlines():
        line = raw.strip()

        if not line:
            continue

        low = line.lower()

        if low in {
            "plaintext",
            "text",
            "spice",
            "netlist",
            "```",
            "```spice",
            "```text",
            "```plaintext",
        }:
            continue

        if line.startswith("*"):
            continue

        result.append(line)

    return result


# ============================================================
# VALUE / PARAMETER HELPERS
# ============================================================

def parse_params(tokens):
    params = {}

    for token in tokens:
        if "=" in token:
            k, v = token.split("=", 1)
            params[k.upper()] = v

    return params


def get_model_and_params(tokens, start_index):
    """
    Returns remaining non key=value tokens plus parameters.
    """

    remaining = tokens[start_index:]

    model_tokens = []
    param_tokens = []

    for t in remaining:
        if "=" in t:
            param_tokens.append(t)
        else:
            model_tokens.append(t)

    return model_tokens, parse_params(param_tokens)


# ============================================================
# COMPONENT PARSER
# ============================================================

def parse_component(line):
    tokens = line.split()

    if not tokens:
        return None

    name = tokens[0]
    prefix = name[0].upper()

    comp = {
        "name": name,
        "raw": line,
        "type": "UNKNOWN",
        "pins": {},
        "value": None,
        "model": None,
        "params": {}
    }

    # --------------------------------------------------------
    # RESISTOR
    # Rname n1 n2 value
    # --------------------------------------------------------

    if prefix == "R" and len(tokens) >= 4:
        comp["type"] = "RESISTOR"

        comp["pins"] = {
            "1": tokens[1],
            "2": tokens[2]
        }

        comp["value"] = tokens[3]

        comp["params"] = parse_params(tokens[4:])

        return comp

    # --------------------------------------------------------
    # CAPACITOR
    # Cname n1 n2 value
    # --------------------------------------------------------

    if prefix == "C" and len(tokens) >= 4:
        comp["type"] = "CAPACITOR"

        comp["pins"] = {
            "1": tokens[1],
            "2": tokens[2]
        }

        comp["value"] = tokens[3]

        comp["params"] = parse_params(tokens[4:])

        return comp

    # --------------------------------------------------------
    # INDUCTOR
    # Lname n1 n2 value
    # --------------------------------------------------------

    if prefix == "L" and len(tokens) >= 4:
        comp["type"] = "INDUCTOR"

        comp["pins"] = {
            "1": tokens[1],
            "2": tokens[2]
        }

        comp["value"] = tokens[3]

        comp["params"] = parse_params(tokens[4:])

        return comp

    # --------------------------------------------------------
    # INDEPENDENT VOLTAGE SOURCE
    # Vname + - ...
    # --------------------------------------------------------

    if prefix == "V" and len(tokens) >= 3:
        comp["type"] = "VOLTAGE_SOURCE"

        comp["pins"] = {
            "+": tokens[1],
            "-": tokens[2]
        }

        comp["value"] = " ".join(tokens[3:])

        return comp

    # --------------------------------------------------------
    # INDEPENDENT CURRENT SOURCE
    # Iname + - ...
    # --------------------------------------------------------

    if prefix == "I" and len(tokens) >= 3:
        comp["type"] = "CURRENT_SOURCE"

        comp["pins"] = {
            "+": tokens[1],
            "-": tokens[2]
        }

        comp["value"] = " ".join(tokens[3:])

        return comp

    # --------------------------------------------------------
    # DIODE
    # Dname anode cathode model ...
    # --------------------------------------------------------

    if prefix == "D" and len(tokens) >= 4:
        comp["type"] = "DIODE"

        comp["pins"] = {
            "A": tokens[1],
            "K": tokens[2]
        }

        comp["model"] = tokens[3]
        comp["params"] = parse_params(tokens[4:])

        return comp

    # --------------------------------------------------------
    # BJT
    # Qname C B E [S] model ...
    #
    # Most Masala-CHAI examples use:
    # Q1 collector base emitter MODEL
    # --------------------------------------------------------

    if prefix == "Q" and len(tokens) >= 5:
        comp["type"] = "BJT"

        # Handle common 3-terminal BJT form
        comp["pins"] = {
            "C": tokens[1],
            "B": tokens[2],
            "E": tokens[3]
        }

        comp["model"] = tokens[4]
        comp["params"] = parse_params(tokens[5:])

        return comp

    # --------------------------------------------------------
    # MOSFET
    # Mname D G S B model params...
    # --------------------------------------------------------

    if prefix == "M" and len(tokens) >= 6:
        comp["type"] = "MOSFET"

        comp["pins"] = {
            "D": tokens[1],
            "G": tokens[2],
            "S": tokens[3],
            "B": tokens[4]
        }

        comp["model"] = tokens[5]
        comp["params"] = parse_params(tokens[6:])

        return comp

    # --------------------------------------------------------
    # SUBCIRCUIT
    # Xname node1 node2 ... subckt_name
    # --------------------------------------------------------

    if prefix == "X" and len(tokens) >= 3:
        comp["type"] = "SUBCIRCUIT"

        # Find first parameter token if present
        param_start = len(tokens)

        for i in range(1, len(tokens)):
            if "=" in tokens[i]:
                param_start = i
                break

        non_params = tokens[1:param_start]

        if len(non_params) >= 1:
            comp["model"] = non_params[-1]

            node_tokens = non_params[:-1]

            comp["pins"] = {
                str(i + 1): node
                for i, node in enumerate(node_tokens)
            }

        comp["params"] = parse_params(tokens[param_start:])

        return comp

    # --------------------------------------------------------
    # VCVS
    # Ename n+ n- nc+ nc- gain
    # --------------------------------------------------------

    if prefix == "E" and len(tokens) >= 6:
        comp["type"] = "VCVS"

        comp["pins"] = {
            "+": tokens[1],
            "-": tokens[2],
            "C+": tokens[3],
            "C-": tokens[4]
        }

        comp["value"] = " ".join(tokens[5:])

        return comp

    # --------------------------------------------------------
    # VCCS
    # Gname n+ n- nc+ nc- gm
    # --------------------------------------------------------

    if prefix == "G" and len(tokens) >= 6:
        comp["type"] = "VCCS"

        comp["pins"] = {
            "+": tokens[1],
            "-": tokens[2],
            "C+": tokens[3],
            "C-": tokens[4]
        }

        comp["value"] = " ".join(tokens[5:])

        return comp

    # --------------------------------------------------------
    # CCCS
    # Fname n+ n- controlling_voltage_source gain
    # --------------------------------------------------------

    if prefix == "F" and len(tokens) >= 5:
        comp["type"] = "CCCS"

        comp["pins"] = {
            "+": tokens[1],
            "-": tokens[2]
        }

        comp["control"] = tokens[3]
        comp["value"] = " ".join(tokens[4:])

        return comp

    # --------------------------------------------------------
    # CCVS
    # Hname n+ n- controlling_voltage_source transresistance
    # --------------------------------------------------------

    if prefix == "H" and len(tokens) >= 5:
        comp["type"] = "CCVS"

        comp["pins"] = {
            "+": tokens[1],
            "-": tokens[2]
        }

        comp["control"] = tokens[3]
        comp["value"] = " ".join(tokens[4:])

        return comp

    return comp


# ============================================================
# NET GRAPH
# ============================================================

def build_nets(components):
    nets = {}

    for comp in components:
        for pin, net in comp["pins"].items():

            if net not in nets:
                nets[net] = []

            nets[net].append({
                "component": comp["name"],
                "pin": pin,
                "type": comp["type"]
            })

    return nets


# ============================================================
# SIMPLE TOPOLOGY FEATURES
# ============================================================

def build_topology_features(components, nets):
    type_counts = Counter(
        c["type"]
        for c in components
    )

    ground_names = {
        "0",
        "gnd",
        "GND",
        "vss",
        "VSS"
    }

    power_names = {
        "vdd",
        "VDD",
        "vcc",
        "VCC",
        "vee",
        "VEE"
    }

    ground_nets = [
        n for n in nets
        if n in ground_names
    ]

    power_nets = [
        n for n in nets
        if n in power_names
    ]

    high_degree_nets = sorted(
        [
            {
                "net": net,
                "degree": len(conns)
            }
            for net, conns in nets.items()
        ],
        key=lambda x: x["degree"],
        reverse=True
    )

    return {
        "component_count": len(components),
        "net_count": len(nets),
        "component_types": dict(type_counts),
        "ground_nets": ground_nets,
        "power_nets": power_nets,
        "high_degree_nets": high_degree_nets[:10]
    }


# ============================================================
# PARSE ONE SPICE FILE
# ============================================================

def parse_spice_file(path):
    text = read_text(path)
    lines = clean_lines(text)

    components = []
    directives = []
    unknown = []

    for line in lines:

        if line.startswith("."):
            directives.append(line)
            continue

        comp = parse_component(line)

        if comp is None:
            continue

        components.append(comp)

        if comp["type"] == "UNKNOWN":
            unknown.append(line)

    nets = build_nets(components)

    topology = build_topology_features(
        components,
        nets
    )

    return {
        "source_file": str(path),
        "components": components,
        "nets": nets,
        "directives": directives,
        "unknown_lines": unknown,
        "topology": topology
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 78)
    print("MASALA-CHAI SPICE → STRUCTURED CIRCUIT DATASET")
    print("=" * 78)

    with open(
        MAPPING_FILE,
        "r",
        encoding="utf-8"
    ) as f:
        mapping = json.load(f)

    print(f"Mapping entries : {len(mapping)}")
    print()

    master_records = []

    total_types = Counter()
    unknown_counter = Counter()

    successful = 0
    failed = 0
    skipped_missing_image = 0

    for idx, item in enumerate(mapping):

        image_path = resolve_path(
            item.get("image")
        )

        spice_path = resolve_path(
            item.get("spice")
        )

        caption_path = resolve_path(
            item.get("caption")
        )

        # We ultimately need paired image + SPICE data.
        # Skip records with missing image.
        if (
            image_path is None
            or not image_path.exists()
        ):
            skipped_missing_image += 1
            continue

        if (
            spice_path is None
            or not spice_path.exists()
        ):
            failed += 1
            continue

        try:
            parsed = parse_spice_file(
                spice_path
            )

            for comp in parsed["components"]:
                total_types[
                    comp["type"]
                ] += 1

            for line in parsed["unknown_lines"]:
                first = (
                    line.split()[0]
                    if line.split()
                    else "EMPTY"
                )

                prefix = (
                    first[0].upper()
                    if first
                    else "?"
                )

                unknown_counter[
                    prefix
                ] += 1

            sample_id = Path(
                item["spice"]
            ).stem.replace(
                "spice",
                ""
            )

            output_record = {
                "sample_id": sample_id,

                "image": str(
                    image_path
                ),

                "spice": str(
                    spice_path
                ),

                "caption": (
                    str(caption_path)
                    if caption_path
                    else None
                ),

                "circuit": parsed
            }

            output_path = (
                OUTPUT_DIR
                / f"sample_{sample_id}.json"
            )

            with open(
                output_path,
                "w",
                encoding="utf-8"
            ) as f:
                json.dump(
                    output_record,
                    f,
                    indent=2
                )

            master_records.append(
                output_record
            )

            successful += 1

        except Exception as e:

            failed += 1

            print(
                f"\nFAILED: {spice_path}"
            )

            print(
                f"Reason: {e}"
            )

        if (
            (idx + 1) % 500 == 0
            or idx + 1 == len(mapping)
        ):
            print(
                f"Processed "
                f"{idx + 1:>5} / "
                f"{len(mapping)}"
            )

    # --------------------------------------------------------
    # MASTER DATASET
    # --------------------------------------------------------

    with open(
        MASTER_FILE,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            master_records,
            f
        )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    summary = {
        "mapping_entries": len(mapping),
        "successful": successful,
        "failed": failed,
        "skipped_missing_image":
            skipped_missing_image,

        "component_types":
            dict(total_types),

        "unknown_prefixes":
            dict(unknown_counter),

        "output_directory":
            str(OUTPUT_DIR),

        "master_file":
            str(MASTER_FILE)
    }

    with open(
        SUMMARY_FILE,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            summary,
            f,
            indent=2
        )

    print()
    print("=" * 78)
    print("SPICE DATASET PARSING COMPLETE")
    print("=" * 78)

    print(
        f"Successful           : "
        f"{successful}"
    )

    print(
        f"Failed               : "
        f"{failed}"
    )

    print(
        f"Missing-image skipped: "
        f"{skipped_missing_image}"
    )

    print()

    print("COMPONENT TYPES")
    print("-" * 78)

    for ctype, count in (
        total_types.most_common()
    ):
        print(
            f"{ctype:<20} "
            f"{count:>8}"
        )

    print()

    print("UNKNOWN PREFIXES")
    print("-" * 78)

    if unknown_counter:
        for prefix, count in (
            unknown_counter.most_common()
        ):
            print(
                f"{prefix:<10} "
                f"{count:>8}"
            )
    else:
        print("None")

    print()

    print(
        f"Individual JSONs → "
        f"{OUTPUT_DIR}"
    )

    print(
        f"Master dataset   → "
        f"{MASTER_FILE}"
    )

    print(
        f"Summary          → "
        f"{SUMMARY_FILE}"
    )


if __name__ == "__main__":
    main()