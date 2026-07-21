import json
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL_ID = "deepseek-ai/deepseek-coder-6.7b-instruct"
ADAPTER_PATH  = "./deepseek_final_circuit_adapter"
DATASET_PATH  = "master_parallel_dataset.json"

tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH, local_files_only=True)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    dtype=torch.bfloat16,
    device_map="cuda",
    trust_remote_code=True,
    local_files_only=True
)
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.eval()

dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
dataset_splits = dataset.train_test_split(test_size=0.05, seed=42)
test_split = dataset_splits["test"]

example = test_split[0]
graph_str = json.dumps(example["graph_input"])

messages = [
    {
        "role": "user",
        "content": (
            "You are an expert analog circuit designer. Convert the "
            "provided netlist JSON graph representation into a valid, "
            f"simulatable SPICE netlist.\n\n{graph_str}"
        )
    }
]

# First, print the RAW PROMPT TEXT so we can see exactly what template is applied
prompt_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
print("=== RAW PROMPT (what the model actually sees) ===")
print(prompt_text[:1000])
print("...")
print()

inputs = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
).to("cuda")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=1024,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id
    )

generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
predicted = tokenizer.decode(generated_ids, skip_special_tokens=True)
predicted_raw = tokenizer.decode(generated_ids, skip_special_tokens=False)

print("=== PREDICTED (skip_special_tokens=True) ===")
print(predicted[:1500])
print()
print("=== PREDICTED (skip_special_tokens=False, showing all tokens) ===")
print(predicted_raw[:1500])
print()
print("=== GROUND TRUTH ===")
print(example["spice_output"][:800])