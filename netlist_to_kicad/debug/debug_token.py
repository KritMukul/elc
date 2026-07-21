from transformers import AutoTokenizer

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    "deepseek-ai/deepseek-coder-6.7b-instruct", 
    local_files_only=True,
    trust_remote_code=True,
    use_fast=False  # <--- THIS IS THE FIX
)

# A fake SPICE string with very clear spaces and newlines
dummy_spice = "* Testbench for analogtobi\n.lib \"{{SKY130_LIB}}\" tt\nVDD N1 0 1.8V"

messages = [
    {"role": "user", "content": "Convert graph"},
    {"role": "assistant", "content": dummy_spice}
]

# Step 1: Apply the template (as text)
templated_text = tokenizer.apply_chat_template(messages, tokenize=False)
print("\n=== 1. TEMPLATED TEXT (Raw String) ===")
print(repr(templated_text))

# Step 2: Tokenize it (how the model sees it during training)
tokens = tokenizer(templated_text, return_tensors="pt")['input_ids'][0]

# Step 3: Decode it back
decoded_text = tokenizer.decode(tokens)
print("\n=== 2. DECODED TEXT (After Tokenization) ===")
print(repr(decoded_text))