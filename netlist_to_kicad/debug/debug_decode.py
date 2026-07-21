import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import json

base_model_path = "deepseek-ai/deepseek-coder-6.7b-instruct"
adapter_path = "../deepseek_final_circuit_adapter" # Update if your DeepSeek adapter is named differently

print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(
    base_model_path, 
    local_files_only=True,
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    local_files_only=True,
    trust_remote_code=True
)
model = PeftModel.from_pretrained(model, adapter_path)

# Use a tiny placeholder graph just to trigger a response
dummy_graph = {"nodes": [{"id": "GND", "type": "Ground"}], "edges": []}
messages = [
    {"role": "user", "content": f"Convert this circuit graph to a SPICE netlist:\n{json.dumps(dummy_graph)}"}
]

inputs = tokenizer.apply_chat_template(
    messages, 
    return_tensors="pt", 
    return_dict=True, 
    add_generation_prompt=True
).to(model.device)

print("Generating...")
# Generate just 50 tokens to see the initial formatting
outputs = model.generate(**inputs, max_new_tokens=50)

# Extract only the newly generated tokens
generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]

print("\n=== RAW TOKEN IDs ===")
print(generated_tokens.tolist())

print("\n=== DECODED (Default) ===")
print(tokenizer.decode(generated_tokens, skip_special_tokens=True))

print("\n=== DECODED (clean_up_tokenization_spaces=False) ===")
print(tokenizer.decode(generated_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False))