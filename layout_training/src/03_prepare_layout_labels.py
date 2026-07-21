import json
import re
from pathlib import Path
from collections import Counter

from PIL import Image, ImageOps


# ============================================================
# PATHS
# ============================================================

WORKSPACE = Path("/workspace")

PROJECT_DIR = WORKSPACE / "layout_training"
DATA_DIR = PROJECT_DIR / "data"
REPORT_DIR = PROJECT_DIR / "reports"

PARSED_DIR = DATA_DIR / "parsed_spice"

PREPARED_DIR = DATA_DIR / "layout_prepared"
PREPARED_IMAGE_DIR = PREPARED_DIR / "images"
PREPARED_JSON_DIR = PREPARED_DIR / "requests"

MASTER_OUTPUT = (
    PREPARED_DIR
    / "layout_requests.json"
)

SUMMARY_OUTPUT = (
    REPORT_DIR
    / "layout_preparation_summary.json"
)

PREPARED_IMAGE_DIR.mkdir(
    parents=True,
    exist_ok=True
)

PREPARED_JSON_DIR.mkdir(
    parents=True,
    exist_ok=True
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SETTINGS
# ============================================================

# Keep enough resolution for later OCR / vision extraction.
TARGET_WIDTH = 1200

# White-border threshold.
WHITE_THRESHOLD = 245

# Padding after content crop.
CROP_PADDING = 20


# ============================================================
# HELPERS
# ============================================================

def load_json(path):

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


def save_json(
    path,
    data,
    indent=2
):

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            data,
            f,
            indent=indent
        )


def natural_number(path):

    m = re.search(
        r"(\d+)",
        path.stem
    )

    if m:
        return int(
            m.group(1)
        )

    return -1


# ============================================================
# IMAGE PREPARATION
# ============================================================

def find_content_bbox(image):

    """
    Detect approximate non-white schematic region.

    We only remove obvious outer white margins.
    We DO NOT segment components here.
    """

    gray = image.convert(
        "L"
    )

    # Anything darker than WHITE_THRESHOLD
    # is considered possible content.

    mask = gray.point(
        lambda p:
        255
        if p < WHITE_THRESHOLD
        else 0
    )

    bbox = mask.getbbox()

    if bbox is None:

        return (
            0,
            0,
            image.width,
            image.height
        )

    left, top, right, bottom = bbox

    left = max(
        0,
        left - CROP_PADDING
    )

    top = max(
        0,
        top - CROP_PADDING
    )

    right = min(
        image.width,
        right + CROP_PADDING
    )

    bottom = min(
        image.height,
        bottom + CROP_PADDING
    )

    return (
        left,
        top,
        right,
        bottom
    )


def prepare_image(
    source_path,
    output_path
):

    with Image.open(
        source_path
    ) as img:

        img = ImageOps.exif_transpose(
            img
        )

        img = img.convert(
            "RGB"
        )

        original_width = (
            img.width
        )

        original_height = (
            img.height
        )

        bbox = find_content_bbox(
            img
        )

        cropped = img.crop(
            bbox
        )

        crop_width = (
            cropped.width
        )

        crop_height = (
            cropped.height
        )

        # Upscale/downscale consistently.

        scale = (
            TARGET_WIDTH
            / crop_width
        )

        target_height = max(
            1,
            round(
                crop_height
                * scale
            )
        )

        resized = cropped.resize(
            (
                TARGET_WIDTH,
                target_height
            ),
            Image.Resampling.LANCZOS
        )

        resized.save(
            output_path,
            quality=95
        )

    return {
        "original_width":
            original_width,

        "original_height":
            original_height,

        "crop_bbox": {
            "left": bbox[0],
            "top": bbox[1],
            "right": bbox[2],
            "bottom": bbox[3]
        },

        "cropped_width":
            crop_width,

        "cropped_height":
            crop_height,

        "prepared_width":
            TARGET_WIDTH,

        "prepared_height":
            target_height,

        "resize_scale":
            scale
    }


# ============================================================
# COMPONENT NORMALIZATION
# ============================================================

def normalize_component_type(
    component
):

    ctype = component.get(
        "type",
        "UNKNOWN"
    )

    model = str(
        component.get(
            "model",
            ""
        )
        or ""
    ).lower()

    # --------------------------------------------------------
    # MOS polarity
    # --------------------------------------------------------

    if ctype == "MOSFET":

        if (
            "pmos" in model
            or "pfet" in model
            or "pch" in model
        ):

            return "PMOS"

        if (
            "nmos" in model
            or "nfet" in model
            or "nch" in model
        ):

            return "NMOS"

        return "MOSFET"

    # --------------------------------------------------------
    # BJT polarity
    # --------------------------------------------------------

    if ctype == "BJT":

        if "pnp" in model:

            return "PNP"

        if "npn" in model:

            return "NPN"

        return "BJT"

    return ctype


# ============================================================
# COMPONENT INVENTORY
# ============================================================

def build_component_inventory(
    circuit
):

    inventory = []

    for component in circuit.get(
        "components",
        []
    ):

        normalized_type = (
            normalize_component_type(
                component
            )
        )

        inventory.append(
            {
                "name":
                    component.get(
                        "name"
                    ),

                "type":
                    normalized_type,

                "base_type":
                    component.get(
                        "type"
                    ),

                "pins":
                    component.get(
                        "pins",
                        {}
                    ),

                "model":
                    component.get(
                        "model"
                    ),

                "value":
                    component.get(
                        "value"
                    ),

                # These are TARGET fields.
                # They remain unknown until
                # visual extraction stage.

                "layout_target": {
                    "visible":
                        None,

                    "center_normalized":
                        None,

                    "bbox_normalized":
                        None,

                    "orientation":
                        None,

                    "mirror":
                        None,

                    "confidence":
                        None
                }
            }
        )

    return inventory


# ============================================================
# NET INVENTORY
# ============================================================

def build_net_inventory(
    circuit
):

    result = []

    nets = circuit.get(
        "nets",
        {}
    )

    for (
        net_name,
        connections
    ) in nets.items():

        result.append(
            {
                "name":
                    net_name,

                "degree":
                    len(
                        connections
                    ),

                "connections":
                    connections
            }
        )

    result.sort(
        key=lambda x:
        (
            -x["degree"],
            x["name"]
        )
    )

    return result


# ============================================================
# EXPECTED VISUAL LABELS
# ============================================================

def build_expected_labels(
    inventory
):

    """
    Component names are useful hints for later OCR/VLM matching.

    Example:
        Q1
        R1
        M2
        C3

    We do NOT assume every SPICE component name must be
    explicitly visible in the schematic image.
    """

    labels = []

    for component in inventory:

        name = component.get(
            "name"
        )

        if name:

            labels.append(
                {
                    "component":
                        name,

                    "candidate_texts": [
                        name,
                        name.upper(),
                        name.lower()
                    ]
                }
            )

    return labels


# ============================================================
# BUILD EXTRACTION REQUEST
# ============================================================

def build_request(
    record,
    prepared_image_path,
    image_metadata
):

    circuit = record[
        "circuit"
    ]

    inventory = (
        build_component_inventory(
            circuit
        )
    )

    nets = (
        build_net_inventory(
            circuit
        )
    )

    request = {

        "sample_id":
            record[
                "sample_id"
            ],

        "source": {

            "original_image":
                record[
                    "image"
                ],

            "prepared_image":
                str(
                    prepared_image_path
                ),

            "spice":
                record[
                    "spice"
                ],

            "caption":
                record.get(
                    "caption"
                )
        },

        "image_metadata":
            image_metadata,

        # ----------------------------------------------------
        # Coordinate convention
        # ----------------------------------------------------

        "coordinate_system": {

            "type":
                "normalized",

            "x_range":
                [
                    0.0,
                    1.0
                ],

            "y_range":
                [
                    0.0,
                    1.0
                ],

            "origin":
                "top_left",

            "description":
                (
                    "x increases left-to-right; "
                    "y increases top-to-bottom. "
                    "All target component centers "
                    "and bounding boxes must be "
                    "normalized to prepared image size."
                )
        },

        # ----------------------------------------------------
        # Electrical truth
        # ----------------------------------------------------

        "components":
            inventory,

        "nets":
            nets,

        "topology":
            circuit.get(
                "topology",
                {}
            ),

        "expected_visual_labels":
            build_expected_labels(
                inventory
            ),

        # ----------------------------------------------------
        # Fields to be populated by next stage
        # ----------------------------------------------------

        "visual_extraction": {

            "status":
                "pending",

            "detected_components":
                [],

            "unmatched_spice_components":
                [],

            "unmatched_visual_components":
                [],

            "global_layout": {

                "power_direction":
                    None,

                "signal_flow":
                    None,

                "symmetry_axis":
                    None,

                "drawing_bbox_normalized":
                    None
            }
        }
    }

    return request


# ============================================================
# PROCESS ONE SAMPLE
# ============================================================

def process_sample(
    path
):

    record = load_json(
        path
    )

    sample_id = str(
        record[
            "sample_id"
        ]
    )

    source_image = Path(
        record[
            "image"
        ]
    )

    if not source_image.exists():

        raise FileNotFoundError(
            source_image
        )

    prepared_image_path = (
        PREPARED_IMAGE_DIR
        / f"sample_{sample_id}.jpg"
    )

    image_metadata = (
        prepare_image(
            source_image,
            prepared_image_path
        )
    )

    request = build_request(
        record,
        prepared_image_path,
        image_metadata
    )

    request_path = (
        PREPARED_JSON_DIR
        / f"sample_{sample_id}.json"
    )

    save_json(
        request_path,
        request
    )

    return {
        "sample_id":
            sample_id,

        "request_file":
            str(
                request_path
            ),

        "prepared_image":
            str(
                prepared_image_path
            ),

        "component_count":
            len(
                request[
                    "components"
                ]
            ),

        "component_types":
            [
                c[
                    "type"
                ]

                for c in request[
                    "components"
                ]
            ],

        "prepared_width":
            image_metadata[
                "prepared_width"
            ],

        "prepared_height":
            image_metadata[
                "prepared_height"
            ]
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 78
    )

    print(
        "MASALA-CHAI LAYOUT LABEL PREPARATION"
    )

    print(
        "=" * 78
    )

    files = sorted(
        PARSED_DIR.glob(
            "sample_*.json"
        ),
        key=natural_number
    )

    print(
        f"Parsed directory : "
        f"{PARSED_DIR}"
    )

    print(
        f"Files found      : "
        f"{len(files)}"
    )

    if not files:

        print()

        print(
            "No parsed samples found."
        )

        print(
            "Run:"
        )

        print(
            "python "
            "src/02_parse_spice_dataset.py"
        )

        return

    master = []

    successful = 0
    failed = 0

    type_counter = Counter()

    component_counts = []

    errors = []

    for i, path in enumerate(
        files,
        start=1
    ):

        try:

            result = process_sample(
                path
            )

            master.append(
                result
            )

            successful += 1

            component_counts.append(
                result[
                    "component_count"
                ]
            )

            type_counter.update(
                result[
                    "component_types"
                ]
            )

        except Exception as e:

            failed += 1

            errors.append(
                {
                    "file":
                        str(
                            path
                        ),

                    "error":
                        str(
                            e
                        )
                }
            )

        if (
            i % 500 == 0
            or i == len(files)
        ):

            print(
                f"Processed "
                f"{i:>5} / "
                f"{len(files)}"
            )

    # --------------------------------------------------------
    # Save master request index
    # --------------------------------------------------------

    save_json(
        MASTER_OUTPUT,
        master
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary = {

        "input_samples":
            len(files),

        "successful":
            successful,

        "failed":
            failed,

        "prepared_image_directory":
            str(
                PREPARED_IMAGE_DIR
            ),

        "request_directory":
            str(
                PREPARED_JSON_DIR
            ),

        "master_request_file":
            str(
                MASTER_OUTPUT
            ),

        "component_type_distribution":
            dict(
                type_counter
            ),

        "errors":
            errors
    }

    if component_counts:

        summary[
            "component_count"
        ] = {

            "min":
                min(
                    component_counts
                ),

            "max":
                max(
                    component_counts
                ),

            "mean":
                (
                    sum(
                        component_counts
                    )
                    / len(
                        component_counts
                    )
                )
        }

    save_json(
        SUMMARY_OUTPUT,
        summary
    )

    # --------------------------------------------------------
    # Terminal output
    # --------------------------------------------------------

    print()

    print(
        "=" * 78
    )

    print(
        "LAYOUT PREPARATION COMPLETE"
    )

    print(
        "=" * 78
    )

    print(
        f"Successful : "
        f"{successful}"
    )

    print(
        f"Failed     : "
        f"{failed}"
    )

    print()

    print(
        "NORMALIZED COMPONENT TYPES"
    )

    print(
        "-" * 78
    )

    for (
        ctype,
        count
    ) in type_counter.most_common():

        print(
            f"{ctype:<20} "
            f"{count:>8}"
        )

    print()

    print(
        "OUTPUT"
    )

    print(
        "-" * 78
    )

    print(
        f"Prepared images → "
        f"{PREPARED_IMAGE_DIR}"
    )

    print(
        f"Layout requests → "
        f"{PREPARED_JSON_DIR}"
    )

    print(
        f"Master index    → "
        f"{MASTER_OUTPUT}"
    )

    print(
        f"Summary         → "
        f"{SUMMARY_OUTPUT}"
    )

    print()

    print(
        "IMPORTANT:"
    )

    print(
        "No component coordinates were guessed."
    )

    print(
        "All layout_target fields are intentionally "
        "empty until visual extraction."
    )


if __name__ == "__main__":

    main()