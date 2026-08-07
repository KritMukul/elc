#!/usr/bin/env python3
"""
run_pipeline_demo.py - End-to-End Pipeline Demo

This script automates the full pipeline:
1. Extract a single graph from masala_chai_graphs.json
2. Generate synthetic schematic image (using generate_yolo_dataset.py logic)
3. Run SINA image-to-netlist on the generated image
4. Convert detected graph to KiCad schematic using Microsoft SchGen
"""

import os
import json
import argparse
import subprocess
import sys
import shutil

def parse_args():
    parser = argparse.ArgumentParser(description="Run full image-to-kicad pipeline on a sample graph.")
    parser.add_argument("--graphs", type=str, default="masala_chai_graphs.json", help="Path to input graphs JSON.")
    parser.add_argument("--sample_index", type=int, default=0, help="Index of the circuit to use.")
    parser.add_argument("--yolo_weights", type=str, default=None, help="Path to YOLO weights for SINA.")
    parser.add_argument("--output_dir", type=str, default="demo_output", help="Directory to save all intermediate and final outputs.")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create isolated output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    temp_graphs_file = os.path.join(args.output_dir, "demo_input_graphs.json")
    yolo_dataset_dir = os.path.join(args.output_dir, "yolo_dataset")
    sina_out_dir = os.path.join(args.output_dir, "sina_out")
    kicad_proj_dir = os.path.join(args.output_dir, "kicad_project")
    
    print("=" * 60)
    print(f"Step 1: Extracting graph index {args.sample_index} from {args.graphs}")
    print("=" * 60)
    
    if not os.path.exists(args.graphs):
        print(f"Error: Could not find {args.graphs}")
        sys.exit(1)
        
    with open(args.graphs, "r") as f:
        graphs = json.load(f)
        
    if args.sample_index >= len(graphs) or args.sample_index < 0:
        print(f"Error: Invalid sample_index {args.sample_index}. Dataset has {len(graphs)} circuits.")
        sys.exit(1)
        
    sample_graph = graphs[args.sample_index]
    
    # Save it as a single-element list for generate_yolo_dataset.py
    with open(temp_graphs_file, "w") as f:
        json.dump([sample_graph], f, indent=4)
        
    print(f"Saved sample graph with {len(sample_graph.get('devices', []))} devices to {temp_graphs_file}")
    
    print("\n" + "=" * 60)
    print("Step 2: Generating synthetic image using generate_yolo_dataset.py")
    print("=" * 60)
    
    # Note: On a machine with kicad-cli, this will produce a real schematic image.
    # Otherwise, it produces a placeholder using OpenCV fallback.
    cmd2 = [
        sys.executable, "generate_yolo_dataset.py",
        "--graphs", temp_graphs_file,
        "--output_dir", yolo_dataset_dir,
        "--max_circuits", "1",
        "--train_split", "1.0" # force into train folder
    ]
    
    result = subprocess.run(cmd2)
    if result.returncode != 0:
        print("Error during image generation.")
        sys.exit(1)
        
    # The generated image should be at yolo_dataset/images/train/train_00000.png
    image_dir = os.path.join(yolo_dataset_dir, "images", "train")
    if not os.path.exists(image_dir) or not os.listdir(image_dir):
        print(f"Error: Expected images in {image_dir} but found none.")
        sys.exit(1)
        
    generated_image = os.path.join(image_dir, os.listdir(image_dir)[0])
    print(f"Image successfully generated: {generated_image}")
    
    print("\n" + "=" * 60)
    print("Step 3: Running SINA Image-to-Netlist pipeline")
    print("=" * 60)
    
    os.makedirs(sina_out_dir, exist_ok=True)
    cmd3 = [
        sys.executable, "sina_pipeline.py",
        "--image_dir", image_dir,
        "--output_dir", sina_out_dir
    ]
    if args.yolo_weights:
        cmd3.extend(["--yolo_weights", args.yolo_weights])
        
    result = subprocess.run(cmd3)
    if result.returncode != 0:
        print("Error during SINA pipeline.")
        sys.exit(1)
        
    # SINA outputs files like train_00000_graph.json
    sina_json_files = [f for f in os.listdir(sina_out_dir) if f.endswith(".json")]
    if not sina_json_files:
        print(f"Error: SINA did not output any JSON files in {sina_out_dir}.")
        sys.exit(1)
        
    detected_graph_file = os.path.join(sina_out_dir, sina_json_files[0])
    print(f"SINA successfully extracted graph: {detected_graph_file}")
    
    print("\n" + "=" * 60)
    print("Step 4: Converting detected graph to KiCad schematic")
    print("=" * 60)
    
    graph_to_kicad_script = os.path.join("netlist_to_kicad", "Microsoft_Schgen", "graph_to_kicad.py")
    
    # We pass the --sina flag to graph_to_kicad to indicate explicit wires and coordinate systems
    
    # We must run this from the Microsoft_Schgen directory because it relies on local imports and init_project.py
    schgen_dir = os.path.join("netlist_to_kicad", "Microsoft_Schgen")
    detected_graph_abs = os.path.abspath(detected_graph_file)
    kicad_proj_abs = os.path.abspath(kicad_proj_dir)
    
    cmd4_adjusted = [
        sys.executable, "graph_to_kicad.py",
        "--graph", detected_graph_abs,
        "--project_name", kicad_proj_abs,
        "--sina"
    ]
    
    result = subprocess.run(cmd4_adjusted, cwd=schgen_dir)
    if result.returncode != 0:
        print("Error during KiCad generation.")
        sys.exit(1)
        
    print("\n" + "=" * 60)
    print(f"SUCCESS! Pipeline completed.")
    print(f"The final KiCad project is located at: {kicad_proj_abs}")
    print("=" * 60)

if __name__ == "__main__":
    main()
