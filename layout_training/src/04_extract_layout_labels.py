import argparse
import json
import re
import traceback
from pathlib import Path

import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
)

# ============================================================
# PATHS
# ============================================================

WORKSPACE = Path("/workspace")
PROJECT_DIR = WORKSPACE / "layout_training"

DATA_DIR = PROJECT_DIR / "data"
REPORT_DIR = PROJECT_DIR / "reports"

REQUEST_DIR = (
    DATA_DIR
    / "layout_prepared"
    / "requests"
)

OUTPUT_DIR = (
    DATA_DIR
    / "layout_extracted"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

SUMMARY_FILE = (
    REPORT_DIR
    / "layout_extraction_summary.json"
)


# ============================================================
# MODEL
# ============================================================

MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"


# ============================================================
# JSON HELPERS
# ============================================================

def load_json(path):

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


def save_json(path, data):

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            data,
            f,
            indent=2
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
# COMPONENT INVENTORY
# ============================================================

def make_component_description(
    request
):

    lines = []

    for comp in request[
        "components"
    ]:

        name = comp[
            "name"
        ]

        ctype = comp[
            "type"
        ]

        value = comp.get(
            "value"
        )

        model = comp.get(
            "model"
        )

        pins = comp.get(
            "pins",
            {}
        )

        text = (
            f"- {name}: "
            f"type={ctype}, "
            f"pins={pins}"
        )

        if value:

            text += (
                f", value={value}"
            )

        if model:

            text += (
                f", model={model}"
            )

        lines.append(
            text
        )

    return "\n".join(
        lines
    )


# ============================================================
# PROMPT
# ============================================================

def build_prompt(
    request
):

    components = (
        make_component_description(
            request
        )
    )

    prompt = f"""
You are analyzing an analog electronic circuit schematic image.

You are given the exact component inventory extracted from the corresponding
SPICE netlist.

Your task is NOT to redesign the circuit and NOT to infer a new circuit.

Your task is only to locate the visible schematic symbols corresponding to
the supplied SPICE components.

COMPONENT INVENTORY:

{components}

Coordinate system:

- x = 0.0 is the LEFT edge of the image
- x = 1.0 is the RIGHT edge
- y = 0.0 is the TOP edge
- y = 1.0 is the BOTTOM edge

For each SPICE component:

1. Determine whether its physical schematic symbol is visible.
2. If visible, estimate the CENTER of the symbol.
3. Estimate a bounding box:
   [x_min, y_min, x_max, y_max]
4. Determine orientation using ONLY:
   vertical
   horizontal
   rotated_90
   rotated_180
   rotated_270
   unknown
5. Determine mirror:
   true
   false
   null
6. Give confidence from 0.0 to 1.0.

Important rules:

- Use the exact SPICE component names provided.
- Do not invent component names.
- Do not omit a component just because its text label is difficult to read.
- Match using symbol type, nearby labels, values, and circuit topology.
- Voltage/current sources may be represented by supply symbols or source symbols.
- Ground symbols are nets, not components, unless explicitly listed as a source.
- Wires are not components.
- Text labels are not components.
- Bounding boxes should cover the physical schematic SYMBOL, not long wires.
- If uncertain, use lower confidence rather than inventing a precise answer.

Also estimate:

- power_direction:
  top_to_bottom, bottom_to_top, left_to_right, right_to_left, mixed, unknown

- signal_flow:
  left_to_right, right_to_left, top_to_bottom, bottom_to_top, mixed, unknown

- symmetry_axis:
  vertical, horizontal, none, unknown

Return ONLY valid JSON.

Required schema:

{{
  "components": [
    {{
      "name": "exact SPICE component name",
      "visible": true,
      "center_normalized": [0.0, 0.0],
      "bbox_normalized": [0.0, 0.0, 0.0, 0.0],
      "orientation": "vertical",
      "mirror": false,
      "confidence": 0.0
    }}
  ],

  "global_layout": {{
    "power_direction": "unknown",
    "signal_flow": "unknown",
    "symmetry_axis": "unknown",
    "drawing_bbox_normalized": [
      0.0,
      0.0,
      1.0,
      1.0
    ]
  }}
}}
"""

    return prompt.strip()


# ============================================================
# MODEL LOADING
# ============================================================

def load_model():

    print(
        "=" * 78
    )

    print(
        "LOADING VISION MODEL"
    )

    print(
        "=" * 78
    )

    print(
        f"Model : {MODEL_NAME}"
    )

    if torch.cuda.is_available():

        print(
            f"GPU   : "
            f"{torch.cuda.get_device_name(0)}"
        )

        print(
            f"CUDA  : available"
        )

    else:

        print(
            "WARNING: CUDA not available."
        )

    processor = (
        AutoProcessor.from_pretrained(
            MODEL_NAME
        )
    )

    model = (
        Qwen2_5_VLForConditionalGeneration
        .from_pretrained(
            MODEL_NAME,
            torch_dtype=(
                torch.bfloat16
                if torch.cuda.is_available()
                else torch.float32
            ),
            device_map="auto"
        )
    )

    model.eval()

    return (
        model,
        processor
    )


# ============================================================
# EXTRACT JSON FROM MODEL TEXT
# ============================================================

def extract_json(
    text
):

    text = text.strip()

    # Remove markdown fences if model returns them.

    text = re.sub(
        r"^```(?:json)?",
        "",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"```$",
        "",
        text
    )

    text = text.strip()

    # First try direct parse.

    try:

        return json.loads(
            text
        )

    except Exception:

        pass

    # Try largest {...} region.

    start = text.find(
        "{"
    )

    end = text.rfind(
        "}"
    )

    if (
        start != -1
        and end != -1
        and end > start
    ):

        candidate = text[
            start:end + 1
        ]

        return json.loads(
            candidate
        )

    raise ValueError(
        "No valid JSON object "
        "found in model output."
    )


# ============================================================
# NORMALIZATION / VALIDATION
# ============================================================

VALID_ORIENTATIONS = {
    "vertical",
    "horizontal",
    "rotated_90",
    "rotated_180",
    "rotated_270",
    "unknown"
}


def clamp01(x):

    try:

        x = float(x)

    except Exception:

        return None

    return max(
        0.0,
        min(
            1.0,
            x
        )
    )


def normalize_center(
    center
):

    if (
        not isinstance(
            center,
            list
        )
        or len(center) != 2
    ):

        return None

    x = clamp01(
        center[0]
    )

    y = clamp01(
        center[1]
    )

    if (
        x is None
        or y is None
    ):

        return None

    return [
        x,
        y
    ]


def normalize_bbox(
    bbox
):

    if (
        not isinstance(
            bbox,
            list
        )
        or len(bbox) != 4
    ):

        return None

    vals = [
        clamp01(v)
        for v in bbox
    ]

    if any(
        v is None
        for v in vals
    ):

        return None

    x1, y1, x2, y2 = vals

    x_min = min(
        x1,
        x2
    )

    x_max = max(
        x1,
        x2
    )

    y_min = min(
        y1,
        y2
    )

    y_max = max(
        y1,
        y2
    )

    return [
        x_min,
        y_min,
        x_max,
        y_max
    ]


def normalize_prediction(
    request,
    prediction
):

    expected = {

        comp["name"]:
            comp

        for comp in request[
            "components"
        ]
    }

    predicted_by_name = {}

    for item in prediction.get(
        "components",
        []
    ):

        name = item.get(
            "name"
        )

        if name in expected:

            predicted_by_name[
                name
            ] = item

    normalized_components = []

    for (
        name,
        source_comp
    ) in expected.items():

        pred = predicted_by_name.get(
            name,
            {}
        )

        visible = pred.get(
            "visible"
        )

        if not isinstance(
            visible,
            bool
        ):

            visible = False

        center = normalize_center(
            pred.get(
                "center_normalized"
            )
        )

        bbox = normalize_bbox(
            pred.get(
                "bbox_normalized"
            )
        )

        orientation = pred.get(
            "orientation",
            "unknown"
        )

        if orientation not in (
            VALID_ORIENTATIONS
        ):

            orientation = "unknown"

        mirror = pred.get(
            "mirror"
        )

        if mirror not in {
            True,
            False,
            None
        }:

            mirror = None

        confidence = clamp01(
            pred.get(
                "confidence",
                0.0
            )
        )

        if confidence is None:

            confidence = 0.0

        # Visible component must have usable geometry.

        if visible and (
            center is None
            or bbox is None
        ):

            confidence = min(
                confidence,
                0.25
            )

        normalized_components.append(
            {
                "name":
                    name,

                "type":
                    source_comp[
                        "type"
                    ],

                "visible":
                    visible,

                "center_normalized":
                    center,

                "bbox_normalized":
                    bbox,

                "orientation":
                    orientation,

                "mirror":
                    mirror,

                "confidence":
                    confidence
            }
        )

    global_pred = prediction.get(
        "global_layout",
        {}
    )

    global_layout = {

        "power_direction":
            global_pred.get(
                "power_direction",
                "unknown"
            ),

        "signal_flow":
            global_pred.get(
                "signal_flow",
                "unknown"
            ),

        "symmetry_axis":
            global_pred.get(
                "symmetry_axis",
                "unknown"
            ),

        "drawing_bbox_normalized":
            normalize_bbox(
                global_pred.get(
                    "drawing_bbox_normalized"
                )
            )
    }

    if (
        global_layout[
            "drawing_bbox_normalized"
        ]
        is None
    ):

        global_layout[
            "drawing_bbox_normalized"
        ] = [
            0.0,
            0.0,
            1.0,
            1.0
        ]

    return {
        "components":
            normalized_components,

        "global_layout":
            global_layout
    }


# ============================================================
# RUN MODEL ON ONE IMAGE
# ============================================================

def infer_one(
    model,
    processor,
    request
):

    image_path = Path(
        request[
            "source"
        ][
            "prepared_image"
        ]
    )

    if not image_path.exists():

        raise FileNotFoundError(
            image_path
        )

    image = Image.open(
        image_path
    ).convert(
        "RGB"
    )

    prompt = build_prompt(
        request
    )

    messages = [
        {
            "role":
                "user",

            "content": [
                {
                    "type":
                        "image",

                    "image":
                        image
                },

                {
                    "type":
                        "text",

                    "text":
                        prompt
                }
            ]
        }
    ]

    text = (
        processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    )

    inputs = processor(
        text=[
            text
        ],
        images=[
            image
        ],
        padding=True,
        return_tensors="pt"
    )

    # Move tensors to model device.

    device = next(
        model.parameters()
    ).device

    inputs = {
        k:
            (
                v.to(
                    device
                )
                if torch.is_tensor(
                    v
                )
                else v
            )

        for k, v
        in inputs.items()
    }

    with torch.inference_mode():

        generated_ids = (
            model.generate(
                **inputs,

                max_new_tokens=1800,

                do_sample=False,

                repetition_penalty=1.02
            )
        )

    input_length = (
        inputs[
            "input_ids"
        ].shape[1]
    )

    generated_only = (
        generated_ids[
            :,
            input_length:
        ]
    )

    output_text = (
        processor.batch_decode(
            generated_only,

            skip_special_tokens=True,

            clean_up_tokenization_spaces=False
        )[0]
    )

    prediction = extract_json(
        output_text
    )

    normalized = (
        normalize_prediction(
            request,
            prediction
        )
    )

    return (
        normalized,
        output_text
    )


# ============================================================
# QUALITY STATISTICS
# ============================================================

def sample_statistics(
    result
):

    components = result[
        "visual_extraction"
    ][
        "components"
    ]

    total = len(
        components
    )

    visible = [
        c
        for c in components
        if c[
            "visible"
        ]
    ]

    high_conf = [
        c
        for c in visible
        if c[
            "confidence"
        ] >= 0.70
    ]

    mean_conf = 0.0

    if visible:

        mean_conf = (
            sum(
                c[
                    "confidence"
                ]
                for c in visible
            )
            / len(
                visible
            )
        )

    return {

        "total_components":
            total,

        "visible_components":
            len(
                visible
            ),

        "high_confidence_visible":
            len(
                high_conf
            ),

        "mean_visible_confidence":
            mean_conf
    }


# ============================================================
# PROCESS ONE SAMPLE
# ============================================================

def process_sample(
    model,
    processor,
    request_path
):

    request = load_json(
        request_path
    )

    sample_id = str(
        request[
            "sample_id"
        ]
    )

    prediction, raw_output = (
        infer_one(
            model,
            processor,
            request
        )
    )

    result = dict(
        request
    )

    result[
        "visual_extraction"
    ] = {

        "status":
            "completed",

        "method":
            MODEL_NAME,

        "components":
            prediction[
                "components"
            ],

        "global_layout":
            prediction[
                "global_layout"
            ]
    }

    result[
        "extraction_statistics"
    ] = sample_statistics(
        result
    )

    output_path = (
        OUTPUT_DIR
        / f"sample_{sample_id}.json"
    )

    save_json(
        output_path,
        result
    )

    # Save raw model answer separately.
    # Useful for debugging hallucinations.

    raw_path = (
        OUTPUT_DIR
        / f"sample_{sample_id}_raw.txt"
    )

    raw_path.write_text(
        raw_output,
        encoding="utf-8"
    )

    return (
        output_path,
        result[
            "extraction_statistics"
        ]
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--limit",
        type=int,
        default=0,

        help=(
            "Maximum number of samples "
            "to process. Default=10. "
            "Use 0 for all samples."
        )
    )

    parser.add_argument(
        "--start",
        type=int,
        default=0,

        help=(
            "Start index in sorted "
            "request list."
        )
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",

        help=(
            "Re-run samples whose output "
            "already exists."
        )
    )

    args = parser.parse_args()

    files = sorted(
        REQUEST_DIR.glob(
            "sample_*.json"
        ),
        key=natural_number
    )

    print(
        "=" * 78
    )

    print(
        "SCHEMATIC VISUAL LAYOUT EXTRACTION"
    )

    print(
        "=" * 78
    )

    print(
        f"Request directory : "
        f"{REQUEST_DIR}"
    )

    print(
        f"Requests found    : "
        f"{len(files)}"
    )

    if not files:

        print(
            "\nNo layout requests found."
        )

        print(
            "Run first:"
        )

        print(
            "python "
            "src/03_prepare_layout_labels.py"
        )

        return

    files = files[
        args.start:
    ]

    if args.limit > 0:

        files = files[
            :args.limit
        ]

    print(
        f"Selected samples  : "
        f"{len(files)}"
    )

    print()

    # Load model only after confirming inputs exist.

    model, processor = (
        load_model()
    )

    successful = 0
    failed = 0
    skipped = 0

    summary_records = []

    for index, request_path in enumerate(
        files,
        start=1
    ):

        sample_id = str(
            natural_number(
                request_path
            )
        )

        output_path = (
            OUTPUT_DIR
            / f"sample_{sample_id}.json"
        )

        print()

        print(
            "-" * 78
        )

        print(
            f"[{index}/{len(files)}] "
            f"Sample {sample_id}"
        )

        if (
            output_path.exists()
            and not args.overwrite
        ):

            print(
                "SKIP: output already exists"
            )

            skipped += 1

            continue

        try:

            saved_path, stats = (
                process_sample(
                    model,
                    processor,
                    request_path
                )
            )

            successful += 1

            print(
                f"Saved   : "
                f"{saved_path}"
            )

            print(
                f"Visible : "
                f"{stats['visible_components']}"
                f"/"
                f"{stats['total_components']}"
            )

            print(
                f"HighConf: "
                f"{stats['high_confidence_visible']}"
            )

            print(
                f"MeanConf: "
                f"{stats['mean_visible_confidence']:.3f}"
            )

            summary_records.append(
                {
                    "sample_id":
                        sample_id,

                    "status":
                        "success",

                    **stats
                }
            )

        except Exception as e:

            failed += 1

            print(
                "FAILED:"
            )

            print(
                type(e).__name__,
                ":",
                e
            )

            traceback.print_exc()

            summary_records.append(
                {
                    "sample_id":
                        sample_id,

                    "status":
                        "failed",

                    "error":
                        str(
                            e
                        )
                }
            )

        # Release temporary CUDA cache.
        # This does not unload the model.

        if torch.cuda.is_available():

            torch.cuda.empty_cache()

    summary = {

        "model":
            MODEL_NAME,

        "selected_samples":
            len(files),

        "successful":
            successful,

        "failed":
            failed,

        "skipped":
            skipped,

        "records":
            summary_records
    }

    save_json(
        SUMMARY_FILE,
        summary
    )

    print()

    print(
        "=" * 78
    )

    print(
        "LAYOUT EXTRACTION COMPLETE"
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

    print(
        f"Skipped    : "
        f"{skipped}"
    )

    print(
        f"Output     : "
        f"{OUTPUT_DIR}"
    )

    print(
        f"Summary    : "
        f"{SUMMARY_FILE}"
    )


if __name__ == "__main__":

    main()