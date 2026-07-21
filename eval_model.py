import json
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm

def main():
    base_model_id = "Qwen/Qwen2.5-Coder-7B-Instruct"
    adapter_path = "./final_circuit_adapter" # The output from your training script

    print("1. Loading Tokenizer and Base Model...")
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="cuda",
        trust_remote_code=True
    )
    
    print("2. Merging Fine-Tuned LoRA Adapter...")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval() # Set to evaluation mode

    print("3. Fetching the STRICT 'test' split from Hugging Face...")
    # This split contains topologies the model has never seen
    dataset = load_dataset("pphilip/analog-circuits-sky130", "with_testbench", split="test")
    
    # We will test on a random sample of 500 unseen circuits to save time
    test_sample = dataset.shuffle(seed=42).select(range(500))

    exact_matches = 0
    valid_syntax = 0

    print(f"4. Starting Inference Evaluation on {len(test_sample)} circuits...")
    
    for row in tqdm(test_sample):
        graph_str = row.get("netlist_json")
        target_spice = row.get("testbench_spice", "").strip()
        
        if not graph_str or not target_spice:
            continue

        # Format prompt exactly like training
        prompt = (
            f"<|im_start|>system\nYou are an expert analog circuit designer. "
            f"Convert the provided netlist JSON graph representation into a valid, "
            f"simulatable SPICE netlist.<|im_end|>\n"
            f"<|im_start|>user\n{graph_str}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        
        # Generate the netlist
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.1, # Low temp for deterministic code generation
                eos_token_id=tokenizer.eos_token_id
            )
        
        # Decode and extract just the assistant's response
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        try:
            generated_spice = generated_text.split("assistant\n")[1].strip()
        except IndexError:
            generated_spice = generated_text.strip()

        # Tier 1: Syntactic Check (Does it look like SPICE?)
        if "M" in generated_spice or "R" in generated_spice or ".subckt" in generated_spice:
            valid_syntax += 1

        # Tier 2: Structural/Exact Match
        # In real EDA evaluation, you would use a graph-isomorphism checker here.
        # For this script, we check if the LLM perfectly replicated the target string structure.
        if generated_spice == target_spice:
            exact_matches += 1

    # Print Report
    print("\n" + "="*50)
    print(" 📊 FINAL LLM TESTING ACCURACY REPORT")
    print("="*50)
    print(f"Total Test Topologies: {len(test_sample)}")
    print(f"Syntactically Valid:   {(valid_syntax / len(test_sample)) * 100:.2f}%")
    print(f"Exact Structural Match:{(exact_matches / len(test_sample)) * 100:.2f}%")
    print("="*50)
    print("Note: An exact string match is a very strict metric. A lower exact match")
    print("does not mean the circuit fails simulation. The next step is piping")
    print("these outputs into ngspice to verify functional accuracy.")

if __name__ == "__main__":
    main()