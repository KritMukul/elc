import json
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


# ============================================================
# PATHS
# ============================================================

ROOT = Path("/workspace/layout_training")

EXTRACTED_DIR = (
    ROOT / "data" / "layout_extracted"
)

OUTPUT_DIR = (
    ROOT / "data" / "verification_overlays"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# HELPERS
# ============================================================

def natural_number(path):

    m = re.search(
        r"(\d+)",
        path.stem
    )

    if m:
        return int(m.group(1))

    return -1


def load_json(path):

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


def get_font(size=22):

    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
    ]

    for path in candidates:

        if Path(path).exists():

            try:
                return ImageFont.truetype(
                    path,
                    size=size
                )

            except Exception:
                pass

    return ImageFont.load_default()


# ============================================================
# DRAW COMPONENT PREDICTION
# ============================================================

def draw_component(
    draw,
    component,
    width,
    height,
    font
):

    name = component.get(
        "name",
        "?"
    )

    visible = component.get(
        "visible",
        False
    )

    confidence = float(
        component.get(
            "confidence",
            0.0
        )
        or 0.0
    )

    bbox = component.get(
        "bbox_normalized"
    )

    center = component.get(
        "center_normalized"
    )

    # If model says component isn't visible,
    # there is no geometry to draw.

    if not visible:
        return False

    if (
        not isinstance(bbox, list)
        or len(bbox) != 4
    ):
        return False

    x1 = int(
        bbox[0] * width
    )

    y1 = int(
        bbox[1] * height
    )

    x2 = int(
        bbox[2] * width
    )

    y2 = int(
        bbox[3] * height
    )

    # Clamp coordinates.

    x1 = max(
        0,
        min(width - 1, x1)
    )

    x2 = max(
        0,
        min(width - 1, x2)
    )

    y1 = max(
        0,
        min(height - 1, y1)
    )

    y2 = max(
        0,
        min(height - 1, y2)
    )

    # Use different visual strength
    # depending on confidence.

    if confidence >= 0.80:

        box_width = 5

    elif confidence >= 0.60:

        box_width = 4

    else:

        box_width = 3

    draw.rectangle(
        [
            x1,
            y1,
            x2,
            y2
        ],
        outline="red",
        width=box_width
    )

    # --------------------------------------------------------
    # Center point
    # --------------------------------------------------------

    if (
        isinstance(center, list)
        and len(center) == 2
    ):

        cx = int(
            center[0] * width
        )

        cy = int(
            center[1] * height
        )

        r = 7

        draw.ellipse(
            [
                cx - r,
                cy - r,
                cx + r,
                cy + r
            ],
            fill="blue",
            outline="white",
            width=2
        )

    # --------------------------------------------------------
    # Label
    # --------------------------------------------------------

    label = (
        f"{name}  "
        f"{confidence:.2f}"
    )

    try:

        text_box = draw.textbbox(
            (0, 0),
            label,
            font=font
        )

        tw = (
            text_box[2]
            - text_box[0]
        )

        th = (
            text_box[3]
            - text_box[1]
        )

    except Exception:

        tw = len(label) * 12
        th = 24

    label_x = x1

    label_y = max(
        0,
        y1 - th - 10
    )

    draw.rectangle(
        [
            label_x,
            label_y,
            label_x + tw + 12,
            label_y + th + 8
        ],
        fill="white",
        outline="red",
        width=2
    )

    draw.text(
        (
            label_x + 6,
            label_y + 2
        ),
        label,
        fill="red",
        font=font
    )

    return True


# ============================================================
# DRAW MISSING COMPONENT PANEL
# ============================================================

def add_missing_panel(
    image,
    components,
    font
):

    missing = [

        c["name"]

        for c in components

        if not c.get(
            "visible",
            False
        )
    ]

    if not missing:

        return image

    panel_height = max(
        70,
        35 + 30 * len(missing)
    )

    output = Image.new(
        "RGB",
        (
            image.width,
            image.height
            + panel_height
        ),
        "white"
    )

    output.paste(
        image,
        (0, 0)
    )

    draw = ImageDraw.Draw(
        output
    )

    y = image.height + 10

    draw.text(
        (
            15,
            y
        ),
        "MODEL MARKED NOT VISIBLE:",
        fill="black",
        font=font
    )

    y += 30

    draw.text(
        (
            15,
            y
        ),
        ", ".join(missing),
        fill="red",
        font=font
    )

    return output


# ============================================================
# PROCESS ONE SAMPLE
# ============================================================

def process_file(path):

    data = load_json(
        path
    )

    sample_id = str(
        data.get(
            "sample_id",
            natural_number(path)
        )
    )

    image_path = Path(
        data[
            "source"
        ][
            "prepared_image"
        ]
    )

    if not image_path.exists():

        raise FileNotFoundError(
            f"Prepared image not found: "
            f"{image_path}"
        )

    image = Image.open(
        image_path
    ).convert(
        "RGB"
    )

    draw = ImageDraw.Draw(
        image
    )

    font = get_font(
        22
    )

    components = (
        data[
            "visual_extraction"
        ].get(
            "components",
            []
        )
    )

    drawn = 0

    for component in components:

        success = draw_component(
            draw,
            component,
            image.width,
            image.height,
            font
        )

        if success:
            drawn += 1

    image = add_missing_panel(
        image,
        components,
        font
    )

    output_path = (
        OUTPUT_DIR
        / f"sample_{sample_id}_overlay.png"
    )

    image.save(
        output_path
    )

    missing = [

        c["name"]

        for c in components

        if not c.get(
            "visible",
            False
        )
    ]

    return {
        "sample_id":
            sample_id,

        "output":
            output_path,

        "total":
            len(components),

        "drawn":
            drawn,

        "missing":
            missing
    }


# ============================================================
# MAIN
# ============================================================

def main():

    files = sorted(
        [
            p

            for p in EXTRACTED_DIR.glob(
                "sample_*.json"
            )

            if not p.name.endswith(
                "_raw.json"
            )
        ],
        key=natural_number
    )

    print(
        "=" * 78
    )

    print(
        "LAYOUT EXTRACTION VISUAL VERIFICATION"
    )

    print(
        "=" * 78
    )

    print(
        f"Extracted directory : "
        f"{EXTRACTED_DIR}"
    )

    print(
        f"JSON files found    : "
        f"{len(files)}"
    )

    if not files:

        print(
            "\nNo extracted JSON files found."
        )

        print(
            "Run 04_extract_layout_labels.py first."
        )

        return

    success = 0
    failed = 0

    for path in files:

        try:

            result = process_file(
                path
            )

            success += 1

            print()

            print(
                f"Sample {result['sample_id']}"
            )

            print(
                f"  Boxes   : "
                f"{result['drawn']}/"
                f"{result['total']}"
            )

            if result[
                "missing"
            ]:

                print(
                    "  Missing : "
                    + ", ".join(
                        result[
                            "missing"
                        ]
                    )
                )

            print(
                f"  Saved   : "
                f"{result['output']}"
            )

        except Exception as e:

            failed += 1

            print()

            print(
                f"FAILED: {path.name}"
            )

            print(
                type(e).__name__,
                ":",
                e
            )

    print()

    print(
        "=" * 78
    )

    print(
        "VERIFICATION OVERLAYS COMPLETE"
    )

    print(
        "=" * 78
    )

    print(
        f"Successful : {success}"
    )

    print(
        f"Failed     : {failed}"
    )

    print(
        f"Output     : {OUTPUT_DIR}"
    )


if __name__ == "__main__":

    main()