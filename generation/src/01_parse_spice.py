import json
import re
import sys
from pathlib import Path

# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = ROOT / "input"
DATA_DIR = ROOT / "data"/"parsed"

DATA_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# UTILITY
# ---------------------------------------------------------

def merge_continuation_lines(text):
    """
    SPICE continuation:
        XM1 ...
        + W=10 L=0.18

    becomes one logical line.
    """

    result = []

    for raw in text.splitlines():

        line = raw.strip()

        if not line:
            continue

        if line.startswith("+") and result:

            result[-1] += " " + line[1:].strip()

        else:

            result.append(line)

    return result


def parse_params(tokens):

    params = {}

    for token in tokens:

        if "=" in token:

            key, value = token.split("=", 1)

            params[key.upper()] = value

    return params


def detect_mos_type(model):

    m = model.lower()

    if (
        "pfet" in m
        or "pmos" in m
        or "pch" in m
    ):
        return "PMOS"

    if (
        "nfet" in m
        or "nmos" in m
        or "nch" in m
    ):
        return "NMOS"

    return "MOS"


# ---------------------------------------------------------
# COMPONENT PARSERS
# ---------------------------------------------------------

def parse_native_mos(tokens, raw):

    # Mname D G S B model [params]

    if len(tokens) < 6:
        return None

    name = tokens[0]

    return {

        "name": name,

        "type": detect_mos_type(
            tokens[5]
        ),

        "pins": {

            "D": tokens[1],
            "G": tokens[2],
            "S": tokens[3],
            "B": tokens[4]

        },

        "model": tokens[5],

        "params": parse_params(
            tokens[6:]
        ),

        "raw": raw
    }


def parse_x_device(tokens, raw):

    """
    Handles SKY130 MOS devices written as:

    XM1 D G S B sky130_fd_pr__nfet_01v8 W=... L=...

    Also preserves generic X subcircuits.
    """

    if len(tokens) < 3:
        return None

    name = tokens[0]

    # Find SKY130 / MOS model token

    model_index = None

    for i in range(1, len(tokens)):

        low = tokens[i].lower()

        if (
            "nfet" in low
            or "pfet" in low
            or "nmos" in low
            or "pmos" in low
        ):

            model_index = i
            break

    # MOS subcircuit

    if model_index is not None and model_index >= 5:

        model = tokens[model_index]

        return {

            "name": name,

            "type": detect_mos_type(
                model
            ),

            "pins": {

                "D": tokens[1],
                "G": tokens[2],
                "S": tokens[3],
                "B": tokens[4]

            },

            "model": model,

            "params": parse_params(
                tokens[model_index + 1:]
            ),

            "raw": raw
        }

    # Generic X subcircuit:
    #
    # Xname node1 node2 ... subckt_name [params]

    non_param = [
        t for t in tokens[1:]
        if "=" not in t
    ]

    if not non_param:
        return None

    model = non_param[-1]

    nodes = non_param[:-1]

    return {

        "name": name,

        "type": "SUBCKT",

        "pins": {

            str(i + 1): node

            for i, node in enumerate(nodes)

        },

        "model": model,

        "params": parse_params(tokens),

        "raw": raw
    }


def parse_two_terminal(
    tokens,
    raw,
    component_type
):

    if len(tokens) < 4:
        return None

    return {

        "name": tokens[0],

        "type": component_type,

        "pins": {

            "1": tokens[1],
            "2": tokens[2]

        },

        "value": " ".join(
            tokens[3:]
        ),

        "raw": raw
    }


def parse_source(
    tokens,
    raw,
    source_type
):

    if len(tokens) < 4:
        return None

    return {

        "name": tokens[0],

        "type": source_type,

        "pins": {

            "+": tokens[1],
            "-": tokens[2]

        },

        "value": " ".join(
            tokens[3:]
        ),

        "raw": raw
    }


def parse_diode(tokens, raw):

    if len(tokens) < 4:
        return None

    return {

        "name": tokens[0],

        "type": "DIODE",

        "pins": {

            "A": tokens[1],
            "K": tokens[2]

        },

        "model": tokens[3],

        "params": parse_params(
            tokens[4:]
        ),

        "raw": raw
    }


def parse_bjt(tokens, raw):

    # Qname C B E [S] model

    if len(tokens) < 5:
        return None

    name = tokens[0]

    if len(tokens) >= 6:

        pins = {

            "C": tokens[1],
            "B": tokens[2],
            "E": tokens[3],
            "S": tokens[4]

        }

        model = tokens[5]

    else:

        pins = {

            "C": tokens[1],
            "B": tokens[2],
            "E": tokens[3]

        }

        model = tokens[4]

    return {

        "name": name,

        "type": "BJT",

        "pins": pins,

        "model": model,

        "raw": raw
    }


# ---------------------------------------------------------
# MAIN SPICE PARSER
# ---------------------------------------------------------

def parse_spice(path):

    text = path.read_text(
        errors="ignore"
    )

    lines = merge_continuation_lines(
        text
    )

    components = []

    directives = []

    comments = []

    ignored = []

    in_control = False

    for line in lines:

        stripped = line.strip()

        lower = stripped.lower()

        # -----------------------------------------
        # Comments
        # -----------------------------------------

        if stripped.startswith("*"):

            comments.append(stripped)

            continue

        # -----------------------------------------
        # ngspice control block
        # -----------------------------------------

        if lower == ".control":

            in_control = True

            directives.append(stripped)

            continue

        if lower == ".endc":

            in_control = False

            directives.append(stripped)

            continue

        if in_control:

            directives.append(stripped)

            continue

        # -----------------------------------------
        # SPICE directives
        # -----------------------------------------

        if stripped.startswith("."):

            directives.append(stripped)

            continue

        tokens = stripped.split()

        if not tokens:

            continue

        name = tokens[0]

        prefix = name[0].upper()

        component = None

        # -----------------------------------------
        # Device dispatch
        # -----------------------------------------

        if prefix == "M":

            component = parse_native_mos(
                tokens,
                stripped
            )

        elif prefix == "X":

            component = parse_x_device(
                tokens,
                stripped
            )

        elif prefix == "R":

            component = parse_two_terminal(
                tokens,
                stripped,
                "RESISTOR"
            )

        elif prefix == "C":

            component = parse_two_terminal(
                tokens,
                stripped,
                "CAPACITOR"
            )

        elif prefix == "L":

            component = parse_two_terminal(
                tokens,
                stripped,
                "INDUCTOR"
            )

        elif prefix == "V":

            component = parse_source(
                tokens,
                stripped,
                "VOLTAGE_SOURCE"
            )

        elif prefix == "I":

            component = parse_source(
                tokens,
                stripped,
                "CURRENT_SOURCE"
            )

        elif prefix == "D":

            component = parse_diode(
                tokens,
                stripped
            )

        elif prefix == "Q":

            component = parse_bjt(
                tokens,
                stripped
            )

        else:

            ignored.append(stripped)

        if component:

            components.append(
                component
            )

        elif stripped not in ignored:

            ignored.append(
                stripped
            )

    return {

        "source_file": path.name,

        "components": components,

        "directives": directives,

        "comments": comments,

        "ignored_lines": ignored
    }


# ---------------------------------------------------------
# BUILD NET-CENTRIC REPRESENTATION
# ---------------------------------------------------------

def build_nets(components):

    nets = {}

    for component in components:

        name = component["name"]

        ctype = component["type"]

        for pin, net in (
            component
            .get("pins", {})
            .items()
        ):

            if net not in nets:

                nets[net] = {

                    "name": net,

                    "connections": []

                }

            nets[net]["connections"].append({

                "component": name,

                "component_type": ctype,

                "pin": pin

            })

    # Add degree

    for net in nets.values():

        net["degree"] = len(
            net["connections"]
        )

    return nets


# ---------------------------------------------------------
# VALIDATION
# ---------------------------------------------------------

def validate_circuit(
    components,
    nets
):

    warnings = []

    names = [
        c["name"]
        for c in components
    ]

    # Duplicate components

    duplicates = {

        name

        for name in names

        if names.count(name) > 1

    }

    if duplicates:

        warnings.append(

            "Duplicate component names: "
            + ", ".join(
                sorted(duplicates)
            )

        )

    # Empty pins

    for component in components:

        if not component.get("pins"):

            warnings.append(

                f"{component['name']} "
                "has no parsed pins"

            )

    # Floating nets

    for name, net in nets.items():

        if net["degree"] == 1:

            warnings.append(

                f"Net '{name}' has only "
                "one connection"

            )

    return warnings


# ---------------------------------------------------------
# PROCESS ONE FILE
# ---------------------------------------------------------

def process_file(path):

    parsed = parse_spice(path)

    nets = build_nets(
        parsed["components"]
    )

    warnings = validate_circuit(
        parsed["components"],
        nets
    )

    result = {

        **parsed,

        "nets": nets,

        "statistics": {

            "component_count":
                len(parsed["components"]),

            "net_count":
                len(nets),

            "directive_count":
                len(parsed["directives"]),

            "ignored_line_count":
                len(parsed["ignored_lines"])

        },

        "validation_warnings":
            warnings
    }

    output_path = (

        DATA_DIR
        / f"{path.stem}_parsed.json"

    )

    output_path.write_text(

        json.dumps(
            result,
            indent=2
        )

    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        f"PARSING: {path.name}"
    )

    print(
        "=" * 70
    )

    print(
        f"Components : "
        f"{len(parsed['components'])}"
    )

    print(
        f"Nets       : "
        f"{len(nets)}"
    )

    print(
        f"Directives : "
        f"{len(parsed['directives'])}"
    )

    print(
        f"Ignored    : "
        f"{len(parsed['ignored_lines'])}"
    )

    print(
        "\nCOMPONENTS"
    )

    print(
        "-" * 70
    )

    for c in parsed["components"]:

        print(

            f"{c['name']:<15}"
            f"{c['type']:<20}"
            f"{c.get('pins', {})}"

        )

    print(
        "\nNETS"
    )

    print(
        "-" * 70
    )

    for name, net in nets.items():

        connections = [

            f"{x['component']}.{x['pin']}"

            for x in net["connections"]

        ]

        print(

            f"{name:<15}"
            f"degree={net['degree']:<3} "
            + ", ".join(connections)

        )

    if warnings:

        print(
            "\nVALIDATION WARNINGS"
        )

        print(
            "-" * 70
        )

        for warning in warnings:

            print(
                "WARNING:",
                warning
            )

    print(
        f"\nSaved → {output_path}"
    )

    return output_path


# ---------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------

def main():

    # Optional:
    #
    # python src/01_parse_spice.py file.spice

    if len(sys.argv) > 1:

        requested = Path(
            sys.argv[1]
        )

        if not requested.is_absolute():

            candidate = (
                INPUT_DIR
                / requested
            )

            if candidate.exists():

                requested = candidate

        if not requested.exists():

            raise FileNotFoundError(

                f"SPICE file not found: "
                f"{requested}"

            )

        files = [requested]

    else:

        files = sorted(

            list(
                INPUT_DIR.glob("*.spice")
            )

            +

            list(
                INPUT_DIR.glob("*.cir")
            )

            +

            list(
                INPUT_DIR.glob("*.sp")
            )

        )

    if not files:

        print(
            "No SPICE files found in:"
        )

        print(
            INPUT_DIR
        )

        return

    print(
        "=" * 70
    )

    print(
        "SPICE → STRUCTURED CIRCUIT PARSER"
    )

    print(
        "=" * 70
    )

    print(
        f"Input directory: {INPUT_DIR}"
    )

    print(
        f"Files found: {len(files)}"
    )

    for path in files:

        process_file(path)


if __name__ == "__main__":

    main()