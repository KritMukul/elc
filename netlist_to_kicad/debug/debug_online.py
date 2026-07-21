from transformers import AutoTokenizer

print("Loading tokenizer online...")
tokenizer = AutoTokenizer.from_pretrained(
    "deepseek-ai/deepseek-coder-6.7b-instruct", 
    trust_remote_code=True
    # local_files_only=True is REMOVED
)

text = "VDD N1 0 1.8V"
token_ids = tokenizer(text, add_special_tokens=False)['input_ids']
pieces = tokenizer.convert_ids_to_tokens(token_ids)

print(f"Token Pieces: {pieces}")