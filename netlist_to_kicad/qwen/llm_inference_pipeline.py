import json
import torch
import argparse
import subprocess
import tempfile
import os
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# --- Configuration ---
ADAPTER_PATH = "../../final_circuit_adapter"
BASE_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"
DATASET_PATH = "../../master_parallel_dataset.json"
SKY130_LIB_PATH = "/usr/local/lib/python3.10/dist-packages/sky130/src/sky130_fd_pr/models/sky130.lib.spice"

class SPICEGenerator:
    def __init__(self):
        print("Loading tokenizer and model...")
        self.tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.bfloat16,
            device_map="cuda"
        )
        self.model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
        self.model.eval()

    def generate(self, graph_dict: dict) -> str:
        graph_str = json.dumps(graph_dict)
        prompt = (
            f"<|im_start|>system\nYou are an expert analog circuit designer. "
            f"Convert the provided netlist JSON graph representation into a valid, "
            f"simulatable SPICE netlist.<|im_end|>\n"
            f"<|im_start|>user\n{graph_str}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=2048,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True)

def validate_with_ngspice(spice_text: str) -> bool:
    spice_text = spice_text.replace("{{SKY130_LIB}}", SKY130_LIB_PATH)
    
    if not spice_text.strip().lower().endswith(".end"):
        spice_text = spice_text.strip() + "\n.end\n"

    with tempfile.NamedTemporaryFile(mode="w", suffix=".cir", delete=False) as f:
        f.write(spice_text)
        tmp_path = f.name

    try:
        result = subprocess.run(
            ["ngspice", "-b", tmp_path],
            capture_output=True,
            text=True,
            timeout=30
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    finally:
        os.unlink(tmp_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate SPICE netlists from JSON graphs using trained Qwen model.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--json_file", type=str, help="Path to a JSON file containing the circuit graph")
    group.add_argument("--dataset_index", type=int, help="Index from the test split to generate (e.g., 224 for analogtobi_0535)")
    parser.add_argument("--output", type=str, default="generated_circuit.spice", help="Output SPICE filename")
    
    args = parser.parse_args()
    generator = SPICEGenerator()
    
    # 1. Load Input Graph
    if args.json_file:
        print(f"Loading graph from {args.json_file}...")
        with open(args.json_file, 'r') as f:
            graph_input = json.load(f)
    else:
        print(f"Loading test split to fetch index {args.dataset_index}...")
        dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
        test_split = dataset.train_test_split(test_size=0.05, seed=42)["test"]
        graph_input = test_split[args.dataset_index]["graph_input"]

    # 2. Generate SPICE
    print("Generating SPICE netlist...")
    spice_output = generator.generate(graph_input)
    
    # 3. Validate
    print("Validating with ngspice...")
    is_valid = validate_with_ngspice(spice_output)
    if is_valid:
        print("✅ ngspice simulation passed.")
    else:
        print("❌ ngspice simulation failed (but saving anyway for inspection).")

    # 4. Save for KiCad Pipeline
    with open(args.output, "w") as f:
        f.write(spice_output)
    print(f"\nSaved successfully to {args.output}")