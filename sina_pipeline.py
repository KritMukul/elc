import os
import sys
import cv2
import json
import argparse
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

try:
    import easyocr
except ImportError:
    easyocr = None

try:
    from skimage.morphology import skeletonize
except ImportError:
    print("Warning: scikit-image not installed. Skeletonization will fail.")

def parse_arguments():
    parser = argparse.ArgumentParser(description="SINA Image-to-Netlist & Layout Pipeline")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing schematic images")
    parser.add_argument("--output_dir", type=str, default="sina_output", help="Directory to save JSON graphs")
    parser.add_argument("--yolo_weights", type=str, default=None, help="Path to trained YOLOv11/v8 model (optional)")
    parser.add_argument("--use_vlm", action="store_true", help="Enable VLM verification (Placeholder)")
    parser.add_argument("--no_ocr", action="store_true", help="Skip OCR text extraction")
    return parser.parse_args()

def extract_line_segments_from_skeleton(skeleton, min_length=10):
    """Extract line segments from a skeleton using Hough Transform."""
    skel_img = (skeleton * 255).astype(np.uint8)
    lines = cv2.HoughLinesP(skel_img, 1, np.pi/180, threshold=15, minLineLength=min_length, maxLineGap=10)
    segments = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            segments.append([int(x1), int(y1), int(x2), int(y2)])
    return segments

def get_pin_role(bbox, px, py):
    """Determine pin role based on which side of the bbox the wire intersects."""
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    if abs(px - x1) < abs(px - x2) and abs(px - cx) > abs(py - cy):
        return "P"
    elif abs(px - x2) < abs(px - x1) and abs(px - cx) > abs(py - cy):
        return "N"
    elif abs(py - y1) < abs(py - y2):
        return "P"
    else:
        return "N"


def classify_component_by_shape(contour, bbox, aspect_ratio, area):
    """
    Heuristically classify a detected component based on its shape properties.
    This is a rough approximation — accuracy improves with YOLO.
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    
    # Compute solidity (filled area / convex hull area)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0
    
    # Compute circularity
    perimeter = cv2.arcLength(contour, True)
    circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
    
    # Classification heuristics for common schematic symbols
    if aspect_ratio > 2.5 or aspect_ratio < 0.4:
        # Very elongated: likely resistor or inductor
        return "resistor"
    elif 0.7 < circularity < 1.0 and solidity > 0.8:
        # Round and solid: likely a node/junction or voltage source
        return "vsource"
    elif solidity < 0.5:
        # Low solidity (lots of holes/gaps): could be a complex symbol like transistor
        return "unknown"
    elif 0.8 < aspect_ratio < 1.2:
        # Roughly square: could be capacitor or generic component
        return "capacitor"
    else:
        return "unknown"


def detect_components_contour(gray, binary):
    """
    Detect components using contour analysis on the binary image.
    Components are compact blobs; wires are long thin lines.
    
    Strategy:
    1. Find all contours in the binary image
    2. For each contour, compute bounding box, area, aspect ratio
    3. Filter: components are compact (low aspect ratio deviation from 1)
       and have moderate area. Wires are very thin or very long.
    4. Use morphological operations to separate touching components
    """
    h, w = binary.shape
    img_area = h * w
    
    # Morphological closing to connect broken component parts
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close, iterations=2)
    
    # Find contours
    contours, hierarchy = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    components = []
    
    # Adaptive thresholds based on image size
    min_area = max(50, img_area * 0.001)   # At least 0.1% of image
    max_area = img_area * 0.15              # At most 15% of image
    min_dim = max(8, min(h, w) * 0.03)     # At least 3% of smallest dimension
    max_aspect = 6.0                        # Max aspect ratio (higher = more wire-like)
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area or area > max_area:
            continue
        
        x, y, bw, bh = cv2.boundingRect(contour)
        
        # Skip very small detections
        if bw < min_dim and bh < min_dim:
            continue
        
        aspect_ratio = max(bw, bh) / (min(bw, bh) + 1e-6)
        
        # Extent: ratio of contour area to bounding box area
        extent = area / (bw * bh + 1e-6)
        
        # Filter out wire-like shapes (very elongated AND thin)
        if aspect_ratio > max_aspect and min(bw, bh) < min_dim * 1.5:
            continue  # This is likely a wire segment, not a component
        
        # Filter out tiny noise
        if bw * bh < min_area * 0.5:
            continue
        
        # Classify the component type
        bbox = [x, y, x + bw, y + bh]
        comp_type = classify_component_by_shape(contour, bbox, aspect_ratio, area)
        
        components.append({
            "name": f"U{len(components)}",
            "type": comp_type,
            "bbox": bbox,
            "conf": round(extent, 3),  # Use extent as a proxy confidence
            "pins": [],
            "_area": area,
            "_aspect": round(aspect_ratio, 2),
        })
    
    # Remove overlapping detections (non-maximum suppression)
    components = nms_components(components, iou_threshold=0.5)
    
    # Re-index names
    for i, comp in enumerate(components):
        comp["name"] = f"U{i}"
        # Remove internal fields
        comp.pop("_area", None)
        comp.pop("_aspect", None)
    
    return components


def nms_components(components, iou_threshold=0.5):
    """Simple non-maximum suppression for overlapping component detections."""
    if not components:
        return []
    
    # Sort by area (larger first — prefer larger detections)
    components.sort(key=lambda c: c["_area"], reverse=True)
    
    keep = []
    for comp in components:
        overlap = False
        for kept in keep:
            iou = compute_iou(comp["bbox"], kept["bbox"])
            if iou > iou_threshold:
                overlap = True
                break
        if not overlap:
            keep.append(comp)
    
    return keep


def compute_iou(box1, box2):
    """Compute Intersection over Union between two bboxes [x1,y1,x2,y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    
    return inter / union if union > 0 else 0


def process_image(img_path, yolo_model, reader, output_dir):
    print(f"Processing {img_path}...")
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error loading {img_path}")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Binarize (white background, black components/wires)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    
    # 1. Component Detection
    components = []
    
    # Try YOLO first
    if yolo_model:
        results = yolo_model(img, verbose=False)[0]
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            cls_name = results.names[cls_id]
            components.append({
                "name": f"U{len(components)}",
                "type": cls_name,
                "bbox": [x1, y1, x2, y2],
                "conf": conf,
                "pins": []
            })
    
    # Fallback: contour-based detection if YOLO found nothing
    if not components:
        components = detect_components_contour(gray, binary)
        if components:
            print(f"  Contour detector found {len(components)} components")
    
    # 2. Text Extraction (OCR)
    texts = []
    if reader:
        ocr_results = reader.readtext(gray)
        for (bbox, text, prob) in ocr_results:
            x_coords = [p[0] for p in bbox]
            y_coords = [p[1] for p in bbox]
            x1, y1, x2, y2 = int(min(x_coords)), int(min(y_coords)), int(max(x_coords)), int(max(y_coords))
            texts.append({"text": text, "bbox": [x1, y1, x2, y2]})

    # Map text to nearest component
    for comp in components:
        cx = (comp["bbox"][0] + comp["bbox"][2]) / 2
        cy = (comp["bbox"][1] + comp["bbox"][3]) / 2
        best_text = None
        min_dist = float('inf')
        for txt in texts:
            tcx = (txt["bbox"][0] + txt["bbox"][2]) / 2
            tcy = (txt["bbox"][1] + txt["bbox"][3]) / 2
            dist = np.hypot(cx - tcx, cy - tcy)
            if dist < min_dist and dist < 100:
                min_dist = dist
                best_text = txt["text"]
        if best_text:
            if any(char.isdigit() for char in best_text):
                if best_text[0].isalpha():
                    comp["name"] = best_text
                else:
                    comp["params"] = {"value": best_text}

    # 3. Connectivity Inference
    # Mask out detected components and text from the wire mask
    wire_mask = binary.copy()
    for comp in components:
        x1, y1, x2, y2 = comp["bbox"]
        cv2.rectangle(wire_mask, (max(0, x1-5), max(0, y1-5)),
                      (min(wire_mask.shape[1], x2+5), min(wire_mask.shape[0], y2+5)), 0, -1)
    
    for txt in texts:
        x1, y1, x2, y2 = txt["bbox"]
        cv2.rectangle(wire_mask, (max(0, x1-2), max(0, y1-2)),
                      (min(wire_mask.shape[1], x2+2), min(wire_mask.shape[0], y2+2)), 0, -1)

    # Connected Component Labeling on remaining pixels (wires)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(wire_mask, connectivity=8)
    
    nets = []
    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] < 20:
            continue
            
        net_name = f"N{label}"
        nets.append({"name": net_name})
        
        net_mask = (labels == label).astype(np.uint8)
        
        # Skeletonize and extract wire segments
        skeleton = skeletonize(net_mask > 0)
        wire_segments = extract_line_segments_from_skeleton(skeleton)
        nets[-1]["wires"] = wire_segments
        
        # Check which components this net touches
        dilated_net = cv2.dilate(net_mask, np.ones((11,11), np.uint8))
        
        for comp in components:
            x1, y1, x2, y2 = comp["bbox"]
            comp_mask = np.zeros_like(net_mask)
            cv2.rectangle(comp_mask, (x1, y1), (x2, y2), 1, -1)
            
            intersection = cv2.bitwise_and(dilated_net, comp_mask)
            if np.any(intersection):
                ys, xs = np.where(intersection > 0)
                px, py = np.mean(xs), np.mean(ys)
                role = get_pin_role(comp["bbox"], px, py)
                comp["pins"].append({"role": role, "net": net_name})

import math

import math

def point_to_segment_dist(px, py, x1, y1, x2, y2):
    """Distance from point (px,py) to line segment (x1,y1)-(x2,y2)."""
    l2 = (x2 - x1)**2 + (y2 - y1)**2
    if l2 == 0:
        return math.hypot(px - x1, py - y1), (x1, y1)
    t = max(0, min(1, ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / l2))
    proj_x = x1 + t * (x2 - x1)
    proj_y = y1 + t * (y2 - y1)
    return math.hypot(px - proj_x, py - proj_y), (proj_x, proj_y)

def fix_graph_connectivity(graph):
    components = graph["devices"]
    nets = graph["nets"]
    
    # 1. Merge collinear or close nets
    MERGE_DIST = 35.0
    COLLINEAR_MERGE_DIST = 200.0
    ALIGN_THRESH = 15.0 # Max pixel difference in X or Y to be considered collinear
    
    while True:
        merged = False
        for i in range(len(nets)):
            for j in range(i + 1, len(nets)):
                net_a = nets[i]
                net_b = nets[j]
                
                should_merge = False
                for wa in net_a.get("wires", []):
                    for wb in net_b.get("wires", []):
                        if len(wa) < 4 or len(wb) < 4: continue
                        pts_a = [(wa[0], wa[1]), (wa[2], wa[3])]
                        pts_b = [(wb[0], wb[1]), (wb[2], wb[3])]
                        for pa in pts_a:
                            for pb in pts_b:
                                dist = math.hypot(pa[0]-pb[0], pa[1]-pb[1])
                                
                                # Close distance merge
                                if dist < MERGE_DIST:
                                    should_merge = True
                                    
                                # Collinear horizontal merge
                                elif dist < COLLINEAR_MERGE_DIST and abs(pa[1] - pb[1]) < ALIGN_THRESH:
                                    # Ensure no component is between them horizontally
                                    min_x, max_x = min(pa[0], pb[0]), max(pa[0], pb[0])
                                    blocked = False
                                    for comp in components:
                                        bx1, by1, bx2, by2 = comp["bbox"]
                                        if by1 < pa[1] < by2 and min_x < bx1 and bx2 < max_x:
                                            blocked = True
                                            break
                                    if not blocked:
                                        should_merge = True
                                        
                                # Collinear vertical merge
                                elif dist < COLLINEAR_MERGE_DIST and abs(pa[0] - pb[0]) < ALIGN_THRESH:
                                    min_y, max_y = min(pa[1], pb[1]), max(pa[1], pb[1])
                                    blocked = False
                                    for comp in components:
                                        bx1, by1, bx2, by2 = comp["bbox"]
                                        if bx1 < pa[0] < bx2 and min_y < by1 and by2 < max_y:
                                            blocked = True
                                            break
                                    if not blocked:
                                        should_merge = True

                                if should_merge:
                                    # Add a bridging wire
                                    net_a["wires"].append([pa[0], pa[1], pb[0], pb[1]])
                                    break
                            if should_merge: break
                        if should_merge: break
                    if should_merge: break
                
                if should_merge:
                    # Merge B into A
                    net_a.setdefault("wires", []).extend(net_b.get("wires", []))
                    for comp in components:
                        for pin in comp.get("pins", []):
                            if pin["net"] == net_b["name"]:
                                pin["net"] = net_a["name"]
                    nets.pop(j)
                    merged = True
                    break
            if merged: break
        if not merged: break
            
    # 2. Draw explicit connections from components to their assigned nets
    for comp in components:
        x1, y1, x2, y2 = comp["bbox"]
        # Component center
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        
        for pin in comp.get("pins", []):
            net_name = pin["net"]
            net = next((n for n in nets if n["name"] == net_name), None)
            if not net: continue
            
            # Find the closest point on any wire in this net to the component center
            min_d = float('inf')
            best_proj = None
            
            for wire in net.get("wires", []):
                if len(wire) < 4: continue
                d, proj = point_to_segment_dist(cx, cy, wire[0], wire[1], wire[2], wire[3])
                if d < min_d:
                    min_d = d
                    best_proj = proj
                    
            # If we found a point and it's somewhat nearby, draw a line from the edge of the bbox to that point
            if best_proj and min_d < 200:
                px, py = best_proj
                
                # Intersect line (cx,cy)->(px,py) with the bounding box so the wire starts exactly at the edge
                if abs(px - cx) > abs(py - cy):
                    # Intersects vertical edge
                    edge_x = x1 if px < cx else x2
                    edge_y = cy + (py - cy) * (edge_x - cx) / (px - cx + 1e-6)
                else:
                    # Intersects horizontal edge
                    edge_y = y1 if py < cy else y2
                    edge_x = cx + (px - cx) * (edge_y - cy) / (py - cy + 1e-6)
                    
                edge_x = max(x1, min(x2, edge_x))
                edge_y = max(y1, min(y2, edge_y))
                
                if math.hypot(px - edge_x, py - edge_y) > 2:
                    net["wires"].append([int(edge_x), int(edge_y), int(px), int(py)])
                    
    return graph


    # 4. Export
    output_graph = {
        "source_file": os.path.basename(img_path),
        "nets": nets,
        "devices": components
    }
    
    # Fix connectivity issues (split nets, disconnected bounding boxes)
    output_graph = fix_graph_connectivity(output_graph)
    
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    out_file = os.path.join(output_dir, f"{base_name}_graph.json")
    with open(out_file, "w") as f:
        json.dump(output_graph, f, indent=4)
        
    print(f"  Saved {out_file} ({len(components)} devices, {len(nets)} nets)")

def main():
    args = parse_arguments()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize YOLO (optional)
    yolo_model = None
    if args.yolo_weights and os.path.exists(args.yolo_weights) and YOLO:
        try:
            yolo_model = YOLO(args.yolo_weights)
            # Quick test: if the model produces zero detections, warn the user
            print(f"YOLO model loaded from {args.yolo_weights}")
        except Exception as e:
            print(f"Failed to load YOLO model: {e}")
    
    if not yolo_model:
        print("No YOLO model — using contour-based component detection fallback.")

    # Initialize OCR (optional)
    reader = None
    if not args.no_ocr and easyocr:
        try:
            reader = easyocr.Reader(['en'], gpu=True)
        except Exception as e:
            print(f"Failed to initialize EasyOCR: {e}")
    
    # Process images
    image_extensions = {".png", ".jpg", ".jpeg"}
    files = sorted([f for f in os.listdir(args.image_dir)
                    if os.path.splitext(f)[1].lower() in image_extensions])
    
    print(f"Found {len(files)} images to process.\n")
    
    for i, filename in enumerate(files):
        img_path = os.path.join(args.image_dir, filename)
        process_image(img_path, yolo_model, reader, args.output_dir)
        
        if (i + 1) % 100 == 0:
            print(f"  Progress: {i+1}/{len(files)}")

    print(f"\nDone! Processed {len(files)} images. Output in {args.output_dir}/")

if __name__ == "__main__":
    main()
