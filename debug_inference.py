import json, torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

ADAPTER_PATH  = "/workspace/final_circuit_adapter"
DATASET_PATH  = "/workspace/master_parallel_dataset.json"
SKY130_LIB    = "/usr/local/lib/python3.10/dist-packages/sky130/src/sky130_fd_pr/models/sky130.lib.spice"

tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-Coder-7B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="cuda"
)
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.eval()

dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
dataset_splits = dataset.train_test_split(test_size=0.05, seed=42)
test_split = dataset_splits["test"]

example = test_split[0]
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
        max_new_tokens=512,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )

generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
predicted = tokenizer.decode(generated_ids, skip_special_tokens=True)

print("=== PREDICTED SPICE ===")
print(predicted[:1000])
print()
print("=== GROUND TRUTH SPICE ===")
print(example["spice_output"][:1000])
