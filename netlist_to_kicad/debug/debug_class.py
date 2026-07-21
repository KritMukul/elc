from transformers import AutoTokenizer, LlamaTokenizer, PreTrainedTokenizerFast

model_path = "deepseek-ai/deepseek-coder-6.7b-instruct"

print("=== 1. AutoTokenizer (trust_remote_code=False) ===")
try:
    t1 = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    print("Class:", type(t1).__name__)
    print("Tokens:", t1.convert_ids_to_tokens(t1("VDD N1 0 1.8V", add_special_tokens=False)['input_ids']))
except Exception as e:
    print("Failed:", e)

print("\n=== 2. PreTrainedTokenizerFast (Direct Load) ===")
try:
    # DeepSeek tokenizers are usually just standard Fast BPE tokenizers
    t2 = PreTrainedTokenizerFast.from_pretrained(model_path, local_files_only=True)
    print("Class:", type(t2).__name__)
    print("Tokens:", t2.convert_ids_to_tokens(t2("VDD N1 0 1.8V", add_special_tokens=False)['input_ids']))
except Exception as e:
    print("Failed:", e)
    
print("\n=== 3. LlamaTokenizer (Direct Load) ===")
try:
    t3 = LlamaTokenizer.from_pretrained(model_path, local_files_only=True)
    print("Class:", type(t3).__name__)
    print("Tokens:", t3.convert_ids_to_tokens(t3("VDD N1 0 1.8V", add_special_tokens=False)['input_ids']))
except Exception as e:
    print("Failed:", e)
    