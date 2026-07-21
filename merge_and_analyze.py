import json
import os
from collections import Counter
from datasets import load_dataset

def load_local_json(filepath):
    if not os.path.exists(filepath):
        print(f"Error: Could not find {filepath}")
        return []
    with open(filepath, 'r') as f:
        return json.load(f)

def fetch_sky130_from_hf():
    print("Fetching 'pphilip/analog-circuits-sky130' from Hugging Face...")
    try:
        # Load the dataset (it will cache locally so subsequent runs are instant)
        dataset = load_dataset("pphilip/analog-circuits-sky130", "default")
        
        sky130_graphs = []
        # Combine all splits (train, validation, test)
        for split in dataset.keys():
            print(f"Processing SKY130 {split} split...")
            for row in dataset[split]:
                # The graph structure is stored in the netlist_json column
                netlist_str = row.get("netlist_json", "{}")
                if netlist_str:
                    try:
                        graph_data = json.loads(netlist_str)
                        # Tag the source for traceability during training
                        graph_data["source_dataset"] = "SKY130"
                        sky130_graphs.append(graph_data)
                    except json.JSONDecodeError:
                        continue
                        
        print(f"Successfully loaded {len(sky130_graphs)} graphs from SKY130.")
        return sky130_graphs
    
    except Exception as e:
        print(f"Failed to fetch from Hugging Face: {e}")
        return []

def compute_statistics(dataset, dataset_name):
    print(f"\n--- Statistics for {dataset_name} ---")
    total_graphs = len(dataset)
    print(f"Total Circuits: {total_graphs}")
    
    if total_graphs == 0:
        return

    device_types = Counter()
    total_devices = 0
    total_nets = 0

    for graph in dataset:
        nets = graph.get("nets", [])
        devices = graph.get("devices", [])
        
        total_nets += len(nets)
        total_devices += len(devices)
        
        for dev in devices:
            device_types[dev.get("type", "unknown")] += 1

    print(f"Average Nets per Circuit: {total_nets / total_graphs:.1f}")
    print(f"Average Devices per Circuit: {total_devices / total_graphs:.1f}")
    
    print("Component Distribution:")
    for comp_type, count in device_types.most_common(10):
        print(f"  - {comp_type}: {count}")

def main():
    # 1. Define paths
    masala_chai_path = "masala_chai_graphs.json"
    output_path = "master_dataset.json"

    # 2. Load Masala-CHAI (Local)
    print("Loading Masala-CHAI dataset...")
    masala_chai_data = load_local_json(masala_chai_path)
    # Tag local dataset
    for graph in masala_chai_data:
        graph["source_dataset"] = "Masala-CHAI"

    # 3. Load SKY130 (Hugging Face API)
    sky130_data = fetch_sky130_from_hf()

    # 4. Compute individual statistics
    if masala_chai_data:
        compute_statistics(masala_chai_data, "Masala-CHAI")
    if sky130_data:
        compute_statistics(sky130_data, "Analog-Circuits-SKY130")

    # 5. Merge
    master_dataset = sky130_data + masala_chai_data
    
    # 6. Compute master statistics
    if master_dataset:
        compute_statistics(master_dataset, "Master Dataset (Merged)")

        # 7. Save the final dataset
        print(f"\nSaving merged dataset to {output_path}...")
        with open(output_path, 'w') as f:
            json.dump(master_dataset, f, indent=4)
        print("Dataset fusion complete! Ready for LoRA training.")

if __name__ == "__main__":
    main()