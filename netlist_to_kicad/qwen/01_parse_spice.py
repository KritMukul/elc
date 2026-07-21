import json
import re
import sys
from pathlib import Path

DEFAULT_INPUT = "qwen_analogtobi_0535.spice"
OUTPUT = "parsed_circuit.json"


def clean_lines(path):
    raw = Path(path).read_text(errors="ignore").splitlines()
    merged = []

    for line in raw:
        line = line.strip()

        if not line:
            continue

        if line.startswith("+") and merged:
            merged[-1] += " " + line[1:].strip()
        else:
            merged.append(line)

    return merged


def parse_params(tokens):
    params = {}

    for token in tokens:
        if "=" in token:
            k, v = token.split("=", 1)
            params[k.upper()] = v

    return params


def parse_spice(path):
    components = []
    directives = []
    in_control = False

    for line in clean_lines(path):
        low = line.lower()

        if line.startswith("*"):
            continue

        if low == ".control":
            in_control = True
            continue

        if low == ".endc":
            in_control = False
            continue

        if in_control:
            continue

        if line.startswith("."):
            directives.append(line)
            continue

        tok = line.split()

        if len(tok) < 3:
            continue

        name = tok[0]
        prefix = name[0].upper()

        # -----------------------------------------
        # Native MOS:
        # Mname D G S B model [params]
        # -----------------------------------------
        if prefix == "M" and len(tok) >= 6:
            model = tok[5]
            ml = model.lower()

            if "pfet" in ml or "pmos" in ml:
                mos_type = "PMOS"
            elif "nfet" in ml or "nmos" in ml:
                mos_type = "NMOS"
            else:
                mos_type = "MOS"

            components.append({
                "name": name,
                "type": mos_type,
                "pins": {
                    "D": tok[1],
                    "G": tok[2],
                    "S": tok[3],
                    "B": tok[4]
                },
                "model": model,
                "params": parse_params(tok[6:]),
                "raw": line
            })

        # -----------------------------------------
        # SKY130 MOS subckt:
        # Xname D G S B model [params]
        # -----------------------------------------
        elif prefix == "X" and len(tok) >= 6:
            model = tok[5]
            ml = model.lower()

            if (
                "nfet" in ml or
                "pfet" in ml or
                "nmos" in ml or
                "pmos" in ml
            ):
                if "pfet" in ml or "pmos" in ml:
                    mos_type = "PMOS"
                else:
                    mos_type = "NMOS"

                components.append({
                    "name": name,
                    "type": mos_type,
                    "pins": {
                        "D": tok[1],
                        "G": tok[2],
                        "S": tok[3],
                        "B": tok[4]
                    },
                    "model": model,
                    "params": parse_params(tok[6:]),
                    "raw": line
                })

            else:
                # Generic subcircuit
                components.append({
                    "name": name,
                    "type": "SUBCKT",
                    "nodes": tok[1:-1],
                    "model": tok[-1],
                    "pins": {
                        str(i + 1): n
                        for i, n in enumerate(tok[1:-1])
                    },
                    "raw": line
                })

        elif prefix in {"R", "C", "L"} and len(tok) >= 4:
            type_map = {
                "R": "RESISTOR",
                "C": "CAPACITOR",
                "L": "INDUCTOR"
            }

            components.append({
                "name": name,
                "type": type_map[prefix],
                "pins": {
                    "1": tok[1],
                    "2": tok[2]
                },
                "value": " ".join(tok[3:]),
                "raw": line
            })

        elif prefix in {"V", "I"} and len(tok) >= 4:
            type_map = {
                "V": "VOLTAGE_SOURCE",
                "I": "CURRENT_SOURCE"
            }

            components.append({
                "name": name,
                "type": type_map[prefix],
                "pins": {
                    "+": tok[1],
                    "-": tok[2]
                },
                "value": " ".join(tok[3:]),
                "raw": line
            })

        elif prefix == "D" and len(tok) >= 4:
            components.append({
                "name": name,
                "type": "DIODE",
                "pins": {
                    "A": tok[1],
                    "K": tok[2]
                },
                "model": tok[3],
                "raw": line
            })

        else:
            components.append({
                "name": name,
                "type": "UNKNOWN",
                "raw": line,
                "pins": {}
            })

    return components, directives


def build_nets(components):
    nets = {}

    for comp in components:
        for pin, net in comp.get("pins", {}).items():
            nets.setdefault(net, []).append({
                "component": comp["name"],
                "pin": pin
            })

    return nets


def main():
    input_file = (
        sys.argv[1]
        if len(sys.argv) > 1
        else DEFAULT_INPUT
    )

    components, directives = parse_spice(input_file)

    result = {
        "source": input_file,
        "components": components,
        "nets": build_nets(components),
        "directives": directives
    }

    Path(OUTPUT).write_text(
        json.dumps(result, indent=2)
    )

    print("=" * 70)
    print("SPICE PARSING COMPLETE")
    print("=" * 70)

    print(f"Source     : {input_file}")
    print(f"Components : {len(components)}")
    print(f"Nets       : {len(result['nets'])}")

    for c in components:
        print(
            f"{c['name']:<15}"
            f"{c['type']:<18}"
            f"{c.get('pins', {})}"
        )

    print(f"\nSaved: {OUTPUT}")


if __name__ == "__main__":
    main()