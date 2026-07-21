import json
from datasets import load_dataset

def build_parallel_dataset():
    master_training_set = []

    # 1. Pull both Graph and Raw SPICE from Hugging Face (SKY130)
    print("Fetching parallel data from Hugging Face SKY130...")
    
    # UPDATE 1: Load the 'with_testbench' config to get the SPICE text
    hf_dataset = load_dataset("pphilip/analog-circuits-sky130", "with_testbench")
    
    for split in hf_dataset.keys():
        print(f"Processing SKY130 {split} split...")
        for row in hf_dataset[split]:
            graph_json_str = row.get("netlist_json")
            
            # UPDATE 2: Use the correct column name for the SPICE target
            spice_target_str = row.get("testbench_spice") 
            
            if graph_json_str and spice_target_str:
                try:
                    graph_dict = json.loads(graph_json_str)
                    master_training_set.append({
                        "graph_input": graph_dict,
                        "spice_output": spice_target_str.strip(),
                        "source": "SKY130"
                    })
                except json.JSONDecodeError:
                    continue

    print(f"Loaded {len(master_training_set)} samples from SKY130.")

    # 2. Placeholder for merging the Masala-CHAI pairs
    # (We will add the Masala-CHAI text-to-graph mapping here next)
    
    print(f"Total parallel training pairs ready: {len(master_training_set)}")
    
    with open("master_parallel_dataset.json", "w") as f:
        json.dump(master_training_set, f, indent=2)
    print("Saved to master_parallel_dataset.json")

if __name__ == "__main__":
    build_parallel_dataset()