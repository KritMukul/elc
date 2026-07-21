import json
import re
from pathlib import Path
from collections import Counter, defaultdict
from statistics import mean, median

from PIL import Image


# ============================================================
# PATHS
# ============================================================

WORKSPACE = Path("/workspace")

DATASET_DIR = WORKSPACE / "masala-chai-dataset-new"

IMAGE_DIR = DATASET_DIR / "images"
SPICE_DIR = DATASET_DIR / "spice"
CAPTION_DIR = DATASET_DIR / "captions"
MAPPING_FILE = DATASET_DIR / "data_mapping.json"

PROJECT_DIR = WORKSPACE / "layout_training"
REPORT_DIR = PROJECT_DIR / "reports"
DATA_DIR = PROJECT_DIR / "data"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# COMPONENT CLASSIFICATION
# ============================================================

PREFIX_TO_TYPE = {
    "R": "RESISTOR",
    "C": "CAPACITOR",
    "L": "INDUCTOR",
    "V": "VOLTAGE_SOURCE",
    "I": "CURRENT_SOURCE",
    "D": "DIODE",
    "Q": "BJT",
    "M": "MOSFET",
    "X": "SUBCIRCUIT",
    "E": "VCVS",
    "F": "CCCS",
    "G": "VCCS",
    "H": "CCVS",
}


# ============================================================
# UTILITY
# ============================================================

def safe_read_text(path):
    try:
        return path.read_text(
            encoding="utf-8",
            errors="ignore"
        )
    except Exception:
        return None


def natural_number(path_string):
    match = re.search(
        r"(\d+)",
        Path(path_string).stem
    )

    if match:
        return int(match.group(1))

    return -1


def resolve_dataset_path(relative_path):
    """
    data_mapping.json appears to contain paths like:

        images/img2.jpg
        captions/cap2.txt
        spice/spice2.txt
    """

    if not relative_path:
        return None

    path = Path(relative_path)

    if path.is_absolute():
        return path

    return DATASET_DIR / path


# ============================================================
# LOAD MAPPING
# ============================================================

def load_mapping():

    if not MAPPING_FILE.exists():
        raise FileNotFoundError(
            f"Mapping file not found: {MAPPING_FILE}"
        )

    with open(
        MAPPING_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(
            "Expected data_mapping.json root "
            "to be a JSON list."
        )

    return data


# ============================================================
# SPICE PARSING FOR AUDIT
# ============================================================

def clean_spice_lines(text):

    result = []

    for raw_line in text.splitlines():

        line = raw_line.strip()

        if not line:
            continue

        lower = line.lower()

        # Dataset sometimes includes labels such as:
        # plaintext
        # ```spice
        # ```

        if lower in {
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

        # SPICE comments
        if line.startswith("*"):
            continue

        # Ignore directives for component statistics
        if line.startswith("."):
            continue

        result.append(line)

    return result


def classify_component(line):

    tokens = line.split()

    if not tokens:
        return None

    name = tokens[0].strip()

    if not name:
        return None

    prefix = name[0].upper()

    return PREFIX_TO_TYPE.get(
        prefix,
        "UNKNOWN"
    )


def estimate_nodes_from_component(line):

    tokens = line.split()

    if len(tokens) < 2:
        return []

    name = tokens[0]

    prefix = name[0].upper()

    # Approximate node count by SPICE primitive.
    # This is only an audit, not our final parser.

    node_counts = {
        "R": 2,
        "C": 2,
        "L": 2,
        "V": 2,
        "I": 2,
        "D": 2,
        "Q": 3,
        "M": 4,
        "E": 4,
        "F": 2,
        "G": 4,
        "H": 2,
    }

    if prefix in node_counts:

        count = node_counts[prefix]

        return tokens[
            1:1 + count
        ]

    if prefix == "X":

        # For a subcircuit instance:
        # Xname n1 n2 ... subckt_name
        #
        # Last token is usually model/subckt name.

        if len(tokens) >= 3:
            return tokens[1:-1]

    return []


def analyze_spice(text):

    lines = clean_spice_lines(text)

    component_counts = Counter()

    nodes = set()

    unknown_lines = []

    component_names = []

    for line in lines:

        ctype = classify_component(line)

        if ctype is None:
            continue

        component_counts[
            ctype
        ] += 1

        tokens = line.split()

        if tokens:
            component_names.append(
                tokens[0]
            )

        for node in estimate_nodes_from_component(
            line
        ):
            nodes.add(node)

        if ctype == "UNKNOWN":
            unknown_lines.append(line)

    total_components = sum(
        component_counts.values()
    )

    return {
        "component_count": total_components,
        "component_types": dict(
            component_counts
        ),
        "node_count_estimate": len(nodes),
        "component_names": component_names,
        "unknown_lines": unknown_lines,
        "non_comment_lines": len(lines),
    }


# ============================================================
# IMAGE ANALYSIS
# ============================================================

def analyze_image(path):

    try:

        with Image.open(path) as img:

            width, height = img.size

            mode = img.mode
            fmt = img.format

        aspect_ratio = (
            width / height
            if height
            else None
        )

        return {
            "valid": True,
            "width": width,
            "height": height,
            "aspect_ratio": aspect_ratio,
            "mode": mode,
            "format": fmt,
            "error": None,
        }

    except Exception as e:

        return {
            "valid": False,
            "width": None,
            "height": None,
            "aspect_ratio": None,
            "mode": None,
            "format": None,
            "error": str(e),
        }


# ============================================================
# CAPTION ANALYSIS
# ============================================================

def analyze_caption(text):

    if text is None:

        return {
            "char_count": 0,
            "word_count": 0,
            "empty": True,
        }

    words = text.split()

    return {
        "char_count": len(text),
        "word_count": len(words),
        "empty": len(words) == 0,
    }


# ============================================================
# DETECT POSSIBLE CIRCUIT FAMILY
#
# This is intentionally coarse.
# It helps us understand what Masala-CHAI contains.
# ============================================================

def infer_circuit_family(
    spice_info,
    caption_text
):

    types = spice_info[
        "component_types"
    ]

    caption = (
        caption_text or ""
    ).lower()

    mos = types.get(
        "MOSFET",
        0
    )

    bjt = types.get(
        "BJT",
        0
    )

    opamp_words = [
        "operational amplifier",
        "op-amp",
        "op amp",
    ]

    if any(
        word in caption
        for word in opamp_words
    ):
        return "OPAMP_RELATED"

    if mos > 0 and bjt == 0:
        return "MOS"

    if bjt > 0 and mos == 0:
        return "BJT"

    if mos > 0 and bjt > 0:
        return "MIXED_TRANSISTOR"

    if (
        types.get(
            "DIODE",
            0
        ) > 0
    ):
        return "DIODE"

    if (
        types.get(
            "RESISTOR",
            0
        ) > 0
        or
        types.get(
            "CAPACITOR",
            0
        ) > 0
        or
        types.get(
            "INDUCTOR",
            0
        ) > 0
    ):
        return "PASSIVE_OR_BASIC"

    return "OTHER"


# ============================================================
# MAIN AUDIT
# ============================================================

def audit_dataset():

    mapping = load_mapping()

    print(
        "=" * 78
    )

    print(
        "MASALA-CHAI DATASET AUDIT"
    )

    print(
        "=" * 78
    )

    print(
        f"Dataset       : {DATASET_DIR}"
    )

    print(
        f"Mapping items : {len(mapping)}"
    )

    print()

    records = []

    missing_images = []
    missing_spice = []
    missing_captions = []

    corrupt_images = []
    empty_spice = []
    empty_captions = []

    total_component_types = Counter()
    circuit_families = Counter()
    image_formats = Counter()
    image_modes = Counter()

    component_counts = []
    node_counts = []
    image_widths = []
    image_heights = []
    caption_word_counts = []

    valid_complete_pairs = 0

    for idx, item in enumerate(
        mapping
    ):

        image_rel = item.get(
            "image"
        )

        spice_rel = item.get(
            "spice"
        )

        caption_rel = item.get(
            "caption"
        )

        image_path = resolve_dataset_path(
            image_rel
        )

        spice_path = resolve_dataset_path(
            spice_rel
        )

        caption_path = resolve_dataset_path(
            caption_rel
        )

        image_exists = (
            image_path is not None
            and image_path.exists()
        )

        spice_exists = (
            spice_path is not None
            and spice_path.exists()
        )

        caption_exists = (
            caption_path is not None
            and caption_path.exists()
        )

        if not image_exists:
            missing_images.append(
                image_rel
            )

        if not spice_exists:
            missing_spice.append(
                spice_rel
            )

        if not caption_exists:
            missing_captions.append(
                caption_rel
            )

        # ----------------------------------------------------
        # IMAGE
        # ----------------------------------------------------

        if image_exists:

            image_info = analyze_image(
                image_path
            )

            if image_info[
                "valid"
            ]:

                image_widths.append(
                    image_info[
                        "width"
                    ]
                )

                image_heights.append(
                    image_info[
                        "height"
                    ]
                )

                image_formats[
                    image_info[
                        "format"
                    ]
                ] += 1

                image_modes[
                    image_info[
                        "mode"
                    ]
                ] += 1

            else:

                corrupt_images.append(
                    {
                        "path": image_rel,
                        "error": image_info[
                            "error"
                        ],
                    }
                )

        else:

            image_info = {
                "valid": False,
                "width": None,
                "height": None,
                "aspect_ratio": None,
                "mode": None,
                "format": None,
                "error": "missing",
            }

        # ----------------------------------------------------
        # SPICE
        # ----------------------------------------------------

        if spice_exists:

            spice_text = safe_read_text(
                spice_path
            )

        else:

            spice_text = None

        if (
            spice_text is None
            or not spice_text.strip()
        ):

            spice_info = {
                "component_count": 0,
                "component_types": {},
                "node_count_estimate": 0,
                "component_names": [],
                "unknown_lines": [],
                "non_comment_lines": 0,
            }

            if spice_exists:
                empty_spice.append(
                    spice_rel
                )

        else:

            spice_info = analyze_spice(
                spice_text
            )

            component_counts.append(
                spice_info[
                    "component_count"
                ]
            )

            node_counts.append(
                spice_info[
                    "node_count_estimate"
                ]
            )

            total_component_types.update(
                spice_info[
                    "component_types"
                ]
            )

        # ----------------------------------------------------
        # CAPTION
        # ----------------------------------------------------

        if caption_exists:

            caption_text = safe_read_text(
                caption_path
            )

        else:

            caption_text = None

        caption_info = analyze_caption(
            caption_text
        )

        if caption_exists:

            caption_word_counts.append(
                caption_info[
                    "word_count"
                ]
            )

            if caption_info[
                "empty"
            ]:

                empty_captions.append(
                    caption_rel
                )

        # ----------------------------------------------------
        # CIRCUIT FAMILY
        # ----------------------------------------------------

        family = infer_circuit_family(
            spice_info,
            caption_text
        )

        circuit_families[
            family
        ] += 1

        # ----------------------------------------------------
        # COMPLETE PAIR?
        # ----------------------------------------------------

        complete = (
            image_exists
            and spice_exists
            and caption_exists
            and image_info[
                "valid"
            ]
            and spice_info[
                "component_count"
            ] > 0
        )

        if complete:
            valid_complete_pairs += 1

        # ----------------------------------------------------
        # RECORD
        # ----------------------------------------------------

        records.append(
            {
                "index": idx,

                "image": image_rel,
                "spice": spice_rel,
                "caption": caption_rel,

                "image_exists":
                    image_exists,

                "spice_exists":
                    spice_exists,

                "caption_exists":
                    caption_exists,

                "complete":
                    complete,

                "image_info":
                    image_info,

                "spice_info":
                    spice_info,

                "caption_info":
                    caption_info,

                "circuit_family":
                    family,
            }
        )

        # Progress

        if (
            (idx + 1) % 500 == 0
            or
            idx + 1 == len(mapping)
        ):

            print(
                f"Processed "
                f"{idx + 1:>5} / "
                f"{len(mapping)}"
            )

    # ========================================================
    # SUMMARY
    # ========================================================

    summary = {
        "dataset_directory":
            str(DATASET_DIR),

        "mapping_items":
            len(mapping),

        "valid_complete_pairs":
            valid_complete_pairs,

        "missing_images":
            len(missing_images),

        "missing_spice":
            len(missing_spice),

        "missing_captions":
            len(missing_captions),

        "corrupt_images":
            len(corrupt_images),

        "empty_spice":
            len(empty_spice),

        "empty_captions":
            len(empty_captions),

        "component_type_totals":
            dict(
                total_component_types
            ),

        "circuit_families":
            dict(
                circuit_families
            ),

        "image_formats":
            dict(
                image_formats
            ),

        "image_modes":
            dict(
                image_modes
            ),
    }

    if component_counts:

        summary[
            "components_per_circuit"
        ] = {
            "min":
                min(component_counts),

            "max":
                max(component_counts),

            "mean":
                round(
                    mean(
                        component_counts
                    ),
                    3
                ),

            "median":
                median(
                    component_counts
                ),
        }

    if node_counts:

        summary[
            "nodes_per_circuit_estimate"
        ] = {
            "min":
                min(node_counts),

            "max":
                max(node_counts),

            "mean":
                round(
                    mean(
                        node_counts
                    ),
                    3
                ),

            "median":
                median(
                    node_counts
                ),
        }

    if image_widths:

        summary[
            "image_width"
        ] = {
            "min":
                min(image_widths),

            "max":
                max(image_widths),

            "mean":
                round(
                    mean(
                        image_widths
                    ),
                    3
                ),

            "median":
                median(
                    image_widths
                ),
        }

    if image_heights:

        summary[
            "image_height"
        ] = {
            "min":
                min(image_heights),

            "max":
                max(image_heights),

            "mean":
                round(
                    mean(
                        image_heights
                    ),
                    3
                ),

            "median":
                median(
                    image_heights
                ),
        }

    if caption_word_counts:

        summary[
            "caption_words"
        ] = {
            "min":
                min(
                    caption_word_counts
                ),

            "max":
                max(
                    caption_word_counts
                ),

            "mean":
                round(
                    mean(
                        caption_word_counts
                    ),
                    3
                ),

            "median":
                median(
                    caption_word_counts
                ),
        }

    # ========================================================
    # SAVE REPORTS
    # ========================================================

    summary_path = (
        REPORT_DIR
        / "masala_chai_audit_summary.json"
    )

    records_path = (
        DATA_DIR
        / "masala_chai_audit_records.json"
    )

    problems_path = (
        REPORT_DIR
        / "masala_chai_problem_files.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2
        ),
        encoding="utf-8"
    )

    records_path.write_text(
        json.dumps(
            records,
            indent=2
        ),
        encoding="utf-8"
    )

    problems = {
        "missing_images":
            missing_images,

        "missing_spice":
            missing_spice,

        "missing_captions":
            missing_captions,

        "corrupt_images":
            corrupt_images,

        "empty_spice":
            empty_spice,

        "empty_captions":
            empty_captions,
    }

    problems_path.write_text(
        json.dumps(
            problems,
            indent=2
        ),
        encoding="utf-8"
    )

    # ========================================================
    # TERMINAL REPORT
    # ========================================================

    print()

    print(
        "=" * 78
    )

    print(
        "DATASET AUDIT COMPLETE"
    )

    print(
        "=" * 78
    )

    print(
        f"Mapping entries       : "
        f"{len(mapping)}"
    )

    print(
        f"Valid complete pairs  : "
        f"{valid_complete_pairs}"
    )

    print(
        f"Missing images        : "
        f"{len(missing_images)}"
    )

    print(
        f"Missing SPICE         : "
        f"{len(missing_spice)}"
    )

    print(
        f"Missing captions      : "
        f"{len(missing_captions)}"
    )

    print(
        f"Corrupt images        : "
        f"{len(corrupt_images)}"
    )

    print(
        f"Empty SPICE           : "
        f"{len(empty_spice)}"
    )

    print()

    print(
        "COMPONENT DISTRIBUTION"
    )

    print(
        "-" * 78
    )

    for (
        ctype,
        count
    ) in total_component_types.most_common():

        print(
            f"{ctype:<20} "
            f"{count:>8}"
        )

    print()

    print(
        "CIRCUIT FAMILY DISTRIBUTION"
    )

    print(
        "-" * 78
    )

    for (
        family,
        count
    ) in circuit_families.most_common():

        print(
            f"{family:<25} "
            f"{count:>8}"
        )

    if component_counts:

        print()

        print(
            "CIRCUIT COMPLEXITY"
        )

        print(
            "-" * 78
        )

        print(
            "Components/circuit:"
        )

        print(
            f"  Min    : "
            f"{min(component_counts)}"
        )

        print(
            f"  Median : "
            f"{median(component_counts)}"
        )

        print(
            f"  Mean   : "
            f"{mean(component_counts):.2f}"
        )

        print(
            f"  Max    : "
            f"{max(component_counts)}"
        )

    if image_widths:

        print()

        print(
            "IMAGE STATISTICS"
        )

        print(
            "-" * 78
        )

        print(
            f"Width  : "
            f"{min(image_widths)} "
            f"→ {max(image_widths)}"
        )

        print(
            f"Height : "
            f"{min(image_heights)} "
            f"→ {max(image_heights)}"
        )

    print()

    print(
        "REPORTS SAVED"
    )

    print(
        "-" * 78
    )

    print(
        summary_path
    )

    print(
        records_path
    )

    print(
        problems_path
    )


# ============================================================
# ENTRY
# ============================================================

if __name__ == "__main__":

    audit_dataset()
