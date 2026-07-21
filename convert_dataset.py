# convert_dataset.py
import os
import json
from spice_to_graph.parser import SpiceParser

def batch_convert(input_dir: str, output_file: str):
    parser = SpiceParser()
    unified_dataset = []
    
    if not os.path.exists(input_dir):
        print(f"Error: Directory '{input_dir}' not found.")
        return

    # Grab all .txt or .spice files
    spice_files = [f for f in os.listdir(input_dir) if f.endswith('.txt') or f.endswith('.spice')]
    total_files = len(spice_files)
    print(f"Found {total_files} files in {input_dir}. Starting conversion...\n")

    success_count = 0
    error_count = 0

    for filename in spice_files:
        filepath = os.path.join(input_dir, filename)
        try:
            # Parse the file
            graph_data = parser.parse(filepath)
            
            # Attach the source filename so we can track it back to the original image/SPICE
            graph_data["source_file"] = filename 
            
            unified_dataset.append(graph_data)
            success_count += 1
            
        except Exception as e:
            print(f"Failed to parse {filename}: {e}")
            error_count += 1

    # Save the master dataset
    print(f"\n--- Conversion Complete ---")
    print(f"Successful: {success_count}/{total_files}")
    print(f"Failed:     {error_count}/{total_files}")
    
    with open(output_file, 'w') as f:
        json.dump(unified_dataset, f, indent=4)
        
    print(f"Unified dataset saved to {output_file}")

if __name__ == "__main__":
    # Adjust this path if your Masala-CHAI folder is located elsewhere
    INPUT_DIRECTORY = "masala-chai-dataset-new/spice/"
    OUTPUT_JSON = "masala_chai_graphs.json"
    
    batch_convert(INPUT_DIRECTORY, OUTPUT_JSON)