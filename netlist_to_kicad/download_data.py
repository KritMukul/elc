import json
from datasets import load_dataset
from tqdm import tqdm

print("Connecting to Hugging Face Stream (Bypassing local disk cache)...")

# CRITICAL: streaming=True forces it to download bytes on the fly. 
# It will NOT cache the 317 GB parquet files to your hard drive.
dataset = load_dataset(
    "bshada/open-schematics", 
    split="train", 
    streaming=True
).select_columns(['schematic_json'])

local_data = []

print("Streaming and extracting valid schematics (This will take a few minutes)...")
# Since it's streaming, we don't know the exact total length, so tqdm will just count up
for row in tqdm(dataset, desc="Rows streamed"):
    if row['schematic_json'] is not None:
        local_data.append(row['schematic_json'])

output_file = "local_schematics.json"
print(f"\nStream complete! Saving {len(local_data)} schematics to {output_file}...")

# Write the final array to disk (Expect this file to be ~1-2 GB max)
with open(output_file, "w") as f:
    json.dump(local_data, f)
    
print("Done! You can now run the 14-core train_pipeline.py.")