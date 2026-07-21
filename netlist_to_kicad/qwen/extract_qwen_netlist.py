import json
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

print("Loading model and tokenizer...")
# Updated paths to go up 2 directories
adapter_path = "../../final_circuit_adapter"
dataset_path = "../../master_parallel_dataset.json"

tokenizer = AutoTokenizer.from_pretrained(adapter_path)
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-Coder-7B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="cuda"
)
model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()

print("Loading dataset...")
# Using the updated dataset path
dataset = load_dataset("json", data_files=dataset_path, split="train")
test_split = dataset.train_test_split(test_size=0.05, seed=42)["test"]

# Index 224 is analogtobi_0535 based on your previous notes
target_idx = 224
example = test_split[target_idx]

print(f"Generating SPICE for index {target_idx}...")
graph_str = json.dumps(example["graph_input"])
prompt = (
    f"<|im_start|>system\nYou are an expert analog circuit designer. "
    f"Convert the provided netlist JSON graph representation into a valid, "
    f"simulatable SPICE netlist.<|im_end|>\n"
    f"<|im_start|>user\n{graph_str}<|im_end|>\n"
    f"<|im_start|>assistant\n"
)

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=2048,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )

generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
predicted_spice = tokenizer.decode(generated_ids, skip_special_tokens=True)

output_filename = "qwen_analogtobi_0535.spice"
with open(output_filename, "w") as f:
    f.write(predicted_spice)

print(f"\nSuccess! Saved to {output_filename}")