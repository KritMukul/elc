import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import subprocess
import sys
import argparse
import os

def generate_spice_from_graph(graph_json_str, base_model_id, adapter_dir):
    print("=====================================")
    print("--- Phase 1: Graph to SPICE ---")
    print(f"Loading Graph-to-SPICE model from '{adapter_dir}'...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id, 
        torch_dtype=torch.bfloat16, 
        device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, adapter_dir).to(device)
    model.eval()
    
    messages = [
        {
            "role": "user",
            "content": (
                "You are an expert analog circuit designer. Convert the "
                "provided netlist JSON graph representation into a valid, "
                f"simulatable SPICE netlist.\n\n{graph_json_str}"
            )
        }
    ]
    
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(device)
    
    print("Generating SPICE netlist...")
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs, 
            max_new_tokens=1024,
            do_sample=False
        )
    
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)]
    spice_output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    # Cleanup memory before Phase 2
    del model
    del base_model
    torch.cuda.empty_cache()
    
    return spice_output

def main():
    parser = argparse.ArgumentParser(description="End-to-End: Graph JSON -> SPICE -> KiCad Schematic")
    parser.add_argument("--graph", type=str, required=True, help="Path to input JSON graph file")
    parser.add_argument("--project_name", type=str, default="auto_generated_schematic", help="Output KiCad project name")
    parser.add_argument("--spice_qwen_base", type=str, default="Qwen/Qwen2.5-Coder-7B-Instruct", help="Base model for Graph->SPICE")
    parser.add_argument("--spice_adapter", type=str, required=True, help="Path to your fine-tuned LoRA adapter for Graph->SPICE")
    args = parser.parse_args()
    
    if not os.path.exists(args.graph):
        print(f"Error: Graph file '{args.graph}' not found.")
        return

    with open(args.graph, "r") as f:
        graph_json_str = f.read()
        
    # --- Step 1: Graph to SPICE ---
    spice_netlist = generate_spice_from_graph(graph_json_str, args.spice_qwen_base, args.spice_adapter)
    
    # Clean up the output in case it's in a markdown block
    if "```spice" in spice_netlist:
        spice_netlist = spice_netlist.split("```spice")[1].split("```")[0].strip()
    elif "```" in spice_netlist:
        spice_netlist = spice_netlist.split("```")[1].strip()
        
    print("\n--- Generated SPICE Netlist ---")
    print(spice_netlist)
    print("-------------------------------\n")
    
    # --- Step 2: SPICE to KiCad Schematic ---
    print("=====================================")
    print("--- Phase 2: SPICE to KiCad Schematic ---")
    
    # Explicitly enforce power rails mapping
    prompt = f"Generate a KiCad schematic for the following SPICE netlist. Ensure you strictly map power pins to 'VDD' and 'GND' and layout the components cleanly.\n\n{spice_netlist}"
    
    cmd = [
        sys.executable, "generate_schematici.py", 
        "--project_name", args.project_name, 
        "--prompt", prompt
    ]
    
    print(f"Running SchGen layout pipeline for project '{args.project_name}'...")
    subprocess.run(cmd)
    print("Pipeline Complete!")

if __name__ == "__main__":
    main()
