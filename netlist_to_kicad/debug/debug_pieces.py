from transformers import AutoTokenizer

for fast_flag in [True, False]:
    print(f"\n=== Testing use_fast={fast_flag} ===")
    
    tokenizer = AutoTokenizer.from_pretrained(
        "deepseek-ai/deepseek-coder-6.7b-instruct", 
        local_files_only=True,
        trust_remote_code=True,
        use_fast=fast_flag
    )
    
    text = "VDD N1 0 1.8V"
    
    # Get the raw token IDs
    token_ids = tokenizer(text, add_special_tokens=False)['input_ids']
    
    # Convert IDs directly to their string representations (bypassing decode)
    pieces = tokenizer.convert_ids_to_tokens(token_ids)
    
    print(f"Token IDs: {token_ids}")
    print(f"Token Pieces: {pieces}")