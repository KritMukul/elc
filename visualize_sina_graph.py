#!/usr/bin/env python3
"""
visualize_sina_graph.py - Draw SINA graph components and wires using Matplotlib.

Usage:
    python visualize_sina_graph.py --input graph.json [--image original.jpg] --output plot.png
"""

import os
import sys
import json
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg

def plot_graph(graph_json, image_path, output_path):
    with open(graph_json, "r") as f:
        graph = json.load(f)
        
    devices = graph.get("devices", [])
    nets = graph.get("nets", [])
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    if image_path and os.path.exists(image_path):
        img = mpimg.imread(image_path)
        ax.imshow(img)
    else:
        # If no image, set inverted Y axis to match image coordinates
        ax.invert_yaxis()
        
    # Plot Devices (Bounding Boxes)
    for dev in devices:
        if "bbox" in dev:
            x1, y1, x2, y2 = dev["bbox"]
            width = x2 - x1
            height = y2 - y1
            
            # Draw box
            rect = patches.Rectangle((x1, y1), width, height, linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            
            # Label
            label = f"{dev.get('name', 'U')} ({dev.get('type', 'unk')})"
            ax.text(x1, y1 - 5, label, color='r', fontsize=10, weight='bold', 
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
            
    # Plot Wires (Lines)
    # Assign a unique color to each net
    cmap = plt.get_cmap("tab20")
    for i, net in enumerate(nets):
        color = cmap(i % 20)
        net_name = net.get("name", f"N{i}")
        
        for wire in net.get("wires", []):
            if len(wire) >= 4:
                x1, y1, x2, y2 = wire[:4]
                ax.plot([x1, x2], [y1, y2], color=color, linewidth=2)
                
        # Optional: Label the net on the first wire segment
        wires = net.get("wires", [])
        if wires and len(wires[0]) >= 4:
            wx, wy = wires[0][0], wires[0][1]
            ax.text(wx, wy, net_name, color=color, fontsize=9, weight='bold',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
            
    ax.set_title(f"SINA Graph: {os.path.basename(graph_json)}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved visualization to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize SINA graph output")
    parser.add_argument("--input", required=True, help="Path to SINA graph JSON")
    parser.add_argument("--image", help="Path to original image (optional)")
    parser.add_argument("--output", required=True, help="Output PNG path")
    args = parser.parse_args()
    
    plot_graph(args.input, args.image, args.output)
