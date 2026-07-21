from datasets import load_dataset

dataset = load_dataset("json", data_files="master_parallel_dataset.json", split="train")
dataset_splits = dataset.train_test_split(test_size=0.05, seed=42)

# This is exactly what the trainer used as eval — same seed guarantees same split
test_split = dataset_splits["test"]
print(f"Test split size: {len(test_split)}")  # should print 6521

# Save it for reuse
test_split.to_json("test_split.json")