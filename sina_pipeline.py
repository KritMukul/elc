import os
import sys
import cv2
import json
import argparse
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    print("Warning: ultralytics not installed. YOLO component detection will fail.")

try:
    import easyocr
except ImportError:
    print("Warning: easyocr not installed. Text extraction will fail.")

try:
    from skimage.morphology import skeletonize
except ImportError:
    print("Warning: scikit-image not installed. Skeletonization will fail.")

def parse_arguments():
    parser = argparse.ArgumentParser(description="SINA Image-to-Netlist & Layout Pipeline")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing schematic images")
    parser.add_argument("--output_dir", type=str, default="sina_output", help="Directory to save JSON graphs")
    parser.add_argument("--yolo_weights", type=str, default="yolo_weights.pt", help="Path to trained YOLOv11/v8 model")
    parser.add_argument("--use_vlm", action="store_true", help="Enable VLM for reference designator verification (Placeholder)")
    return parser.parse_args()

def extract_line_segments_from_skeleton(skeleton, min_length=10):
    """
    Given a boolean skeleton image, extracts line segments.
    Uses Probabilistic Hough Transform as a simple approximation.
    """
    skel_img = (skeleton * 255).astype(np.uint8)
    # Adjust parameters depending on image resolution
    lines = cv2.HoughLinesP(skel_img, 1, np.pi/180, threshold=15, minLineLength=min_length, maxLineGap=10)
    segments = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            segments.append([int(x1), int(y1), int(x2), int(y2)])
    return segments

def get_pin_role(bbox, px, py):
    """
    Heuristically determine pin role based on which side of the bbox the wire intersects.
    bbox: [x1, y1, x2, y2]
    px, py: intersection point
    """
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    
    # Very simple heuristic: left=P, right=N, top=P, bottom=N (depends on component type normally)
    # For a more robust system, this needs component-specific pin mappings.
    if abs(px - x1) < abs(px - x2) and abs(px - cx) > abs(py - cy):
        return "P" # Left
    elif abs(px - x2) < abs(px - x1) and abs(px - cx) > abs(py - cy):
        return "N" # Right
    elif abs(py - y1) < abs(py - y2):
        return "P" # Top
    else:
        return "N" # Bottom

def process_image(img_path, yolo_model, reader, output_dir):
    print(f"Processing {img_path}...")
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error loading {img_path}")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. Component Detection (YOLO)
    components = []
    if yolo_model:
        results = yolo_model(img, verbose=False)[0]
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            cls_name = results.names[cls_id]
            components.append({
                "name": f"U{len(components)}", # Placeholder name, will be updated by OCR
                "type": cls_name,
                "bbox": [x1, y1, x2, y2],
                "conf": conf,
                "pins": [] # To be populated in connectivity stage
            })

    # 2. Text Extraction (OCR)
    texts = []
    if reader:
        ocr_results = reader.readtext(gray)
        for (bbox, text, prob) in ocr_results:
            # bbox is a list of 4 points: [tl, tr, br, bl]
            x_coords = [p[0] for p in bbox]
            y_coords = [p[1] for p in bbox]
            x1, y1, x2, y2 = int(min(x_coords)), int(min(y_coords)), int(max(x_coords)), int(max(y_coords))
            texts.append({"text": text, "bbox": [x1, y1, x2, y2]})

    # Map text to components (Basic spatial heuristic: nearest component)
    for comp in components:
        cx = (comp["bbox"][0] + comp["bbox"][2]) / 2
        cy = (comp["bbox"][1] + comp["bbox"][3]) / 2
        best_text = None
        min_dist = float('inf')
        for txt in texts:
            tcx = (txt["bbox"][0] + txt["bbox"][2]) / 2
            tcy = (txt["bbox"][1] + txt["bbox"][3]) / 2
            dist = np.hypot(cx - tcx, cy - tcy)
            if dist < min_dist and dist < 100: # Distance threshold
                min_dist = dist
                best_text = txt["text"]
        if best_text:
            # Very naive assumption: text might be refdes (e.g. R1) or value (e.g. 10k)
            # A real VLM would parse this contextually.
            if any(char.isdigit() for char in best_text):
                if best_text[0].isalpha():
                    comp["name"] = best_text
                else:
                    comp["params"] = {"value": best_text}

    # 3. Connectivity Inference
    # Binarize to get wires (assuming white background, black wires)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    
    # Mask out components and texts
    wire_mask = binary.copy()
    for comp in components:
        x1, y1, x2, y2 = comp["bbox"]
        # Expand slightly to break connections inside component body
        cv2.rectangle(wire_mask, (max(0, x1-5), max(0, y1-5)), (min(wire_mask.shape[1], x2+5), min(wire_mask.shape[0], y2+5)), 0, -1)
    
    for txt in texts:
        x1, y1, x2, y2 = txt["bbox"]
        cv2.rectangle(wire_mask, (max(0, x1-2), max(0, y1-2)), (min(wire_mask.shape[1], x2+2), min(wire_mask.shape[0], y2+2)), 0, -1)

    # Connected Component Labeling
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(wire_mask, connectivity=8)
    
    nets = []
    # Identify which net intersects with which component
    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] < 20: # Filter noise
            continue
            
        net_name = f"N{label}"
        nets.append({"name": net_name})
        
        # Extract wiring arrangement for this net
        net_mask = (labels == label).astype(np.uint8)
        
        # Skeletonize
        skeleton = skeletonize(net_mask > 0)
        wire_segments = extract_line_segments_from_skeleton(skeleton)
        
        # Add wires to the net definition for KiCad export
        nets[-1]["wires"] = wire_segments
        
        # Check intersections with components
        # We dilate the net mask slightly to overlap with the component bounding boxes
        dilated_net = cv2.dilate(net_mask, np.ones((11,11), np.uint8))
        
        for comp in components:
            x1, y1, x2, y2 = comp["bbox"]
            comp_mask = np.zeros_like(net_mask)
            cv2.rectangle(comp_mask, (x1, y1), (x2, y2), 1, -1)
            
            intersection = cv2.bitwise_and(dilated_net, comp_mask)
            if np.any(intersection):
                # Find the center of the intersection to determine pin role
                ys, xs = np.where(intersection > 0)
                px, py = np.mean(xs), np.mean(ys)
                
                role = get_pin_role(comp["bbox"], px, py)
                comp["pins"].append({"role": role, "net": net_name})

    # 4. Schematic Recreation Export
    output_graph = {
        "source_file": os.path.basename(img_path),
        "nets": nets,
        "devices": components
    }
    
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    out_file = os.path.join(output_dir, f"{base_name}_graph.json")
    with open(out_file, "w") as f:
        json.dump(output_graph, f, indent=4)
        
    print(f"Saved {out_file}")

def main():
    args = parse_arguments()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize models
    yolo_model = None
    if os.path.exists(args.yolo_weights):
        try:
            yolo_model = YOLO(args.yolo_weights)
        except Exception as e:
            print(f"Failed to load YOLO model: {e}")
    else:
        print(f"YOLO weights not found at {args.yolo_weights}. Proceeding without component detection.")

    reader = None
    try:
        reader = easyocr.Reader(['en'])
    except Exception as e:
        print(f"Failed to initialize EasyOCR: {e}")
        
    # Process images
    image_extensions = {".png", ".jpg", ".jpeg"}
    for filename in os.listdir(args.image_dir):
        ext = os.path.splitext(filename)[1].lower()
        if ext in image_extensions:
            img_path = os.path.join(args.image_dir, filename)
            process_image(img_path, yolo_model, reader, args.output_dir)

if __name__ == "__main__":
    main()
