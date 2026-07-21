import json
import torch
import subprocess
import tempfile
import os
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ── 1. Load model ─────────────────────────────────────────────────────────────
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained("./final_circuit_adapter")
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-Coder-7B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="cuda"
)
model = PeftModel.from_pretrained(base_model, "./final_circuit_adapter")
model.eval()

# ── 2. Load the same test split ───────────────────────────────────────────────
print("Loading test split...")
dataset = load_dataset("json", data_files="master_parallel_dataset.json", split="train")
dataset_splits = dataset.train_test_split(test_size=0.05, seed=42)
test_split = dataset_splits["test"]
print(f"Test split size: {len(test_split)}")


SKY130_LIB_PATH = "/usr/local/lib/python3.10/dist-packages/sky130/src/sky130_fd_pr/models/sky130.lib.spice"



# ── 3. Simulation helper ──────────────────────────────────────────────────────
def run_ngspice(spice_text: str) -> dict:
    spice_text = spice_text.replace("{{SKY130_LIB}}", SKY130_LIB_PATH)  # <-- add this line
    
    if not spice_text.strip().lower().endswith(".end"):
        spice_text = spice_text.strip() + "\n.end\n"

    with tempfile.NamedTemporaryFile(mode="w", suffix=".cir", delete=False) as f:
        f.write(spice_text)
        tmp_path = f.name

    try:
        result = subprocess.run(
            ["ngspice", "-b", tmp_path],      # -b = batch mode, no GUI
            capture_output=True,
            text=True,
            timeout=30                         # kill if hangs
        )
        passed = result.returncode == 0
        return {
            "passed": passed,
            "returncode": result.returncode,
            "stderr": result.stderr[:300]      # trim long errors
        }
    except subprocess.TimeoutExpired:
        return {"passed": False, "returncode": -1, "stderr": "TIMEOUT"}
    finally:
        os.unlink(tmp_path)

# ── 4. Generate SPICE from model ──────────────────────────────────────────────
def generate_spice(graph_input: dict) -> str:
    graph_str = json.dumps(graph_input)
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
            max_new_tokens=2048,
            do_sample=False,          # greedy — deterministic output
            pad_token_id=tokenizer.eos_token_id
        )
    # Slice off the prompt tokens, decode only the generated part
    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True)

# ── 5. Eval loop ──────────────────────────────────────────────────────────────
NUM_SAMPLES = 200   # increase to 500/1000 once you confirm it works

results = {
    "predicted_pass": 0,
    "predicted_fail": 0,
    "groundtruth_pass": 0,
    "groundtruth_fail": 0,
    "both_pass": 0,
    "both_fail": 0,
    "pred_pass_gt_fail": 0,
    "pred_fail_gt_pass": 0,
    "failures": []           # store failed cases for inspection
}

print(f"\nRunning simulation eval on {NUM_SAMPLES} examples...\n")

for idx, example in enumerate(test_split.select(range(NUM_SAMPLES))):
    predicted_spice = generate_spice(example["graph_input"])
    ground_truth_spice = example["spice_output"]

    pred_result = run_ngspice(predicted_spice)
    gt_result   = run_ngspice(ground_truth_spice)

    pred_passed = pred_result["passed"]
    gt_passed   = gt_result["passed"]

    # Tally
    results["predicted_pass" if pred_passed else "predicted_fail"] += 1
    results["groundtruth_pass" if gt_passed else "groundtruth_fail"] += 1

    if pred_passed and gt_passed:
        results["both_pass"] += 1
    elif not pred_passed and not gt_passed:
        results["both_fail"] += 1
    elif pred_passed and not gt_passed:
        results["pred_pass_gt_fail"] += 1
    else:
        results["pred_fail_gt_pass"] += 1
        # Log these — model failed where ground truth passes
        if len(results["failures"]) < 20:
            results["failures"].append({
                "idx": idx,
                "predicted": predicted_spice[:500],
                "ground_truth": ground_truth_spice[:500],
                "stderr": pred_result["stderr"]
            })

    # Progress every 10 examples
    if (idx + 1) % 10 == 0:
        so_far = idx + 1
        print(
            f"[{so_far}/{NUM_SAMPLES}] "
            f"Pred pass rate: {results['predicted_pass']/so_far:.1%} | "
            f"GT pass rate: {results['groundtruth_pass']/so_far:.1%}"
        )

# ── 6. Final report ───────────────────────────────────────────────────────────
print("\n" + "="*60)
print("SIMULATION EVALUATION REPORT")
print("="*60)
print(f"Total evaluated       : {NUM_SAMPLES}")
print(f"Predicted PASS        : {results['predicted_pass']} ({results['predicted_pass']/NUM_SAMPLES:.1%})")
print(f"Predicted FAIL        : {results['predicted_fail']} ({results['predicted_fail']/NUM_SAMPLES:.1%})")
print(f"Ground Truth PASS     : {results['groundtruth_pass']} ({results['groundtruth_pass']/NUM_SAMPLES:.1%})")
print(f"Ground Truth FAIL     : {results['groundtruth_fail']} ({results['groundtruth_fail']/NUM_SAMPLES:.1%})")
print("-"*60)
print(f"Both PASS             : {results['both_pass']}  ← model correct")
print(f"Both FAIL             : {results['both_fail']}  ← dataset issue (not model's fault)")
print(f"Pred PASS, GT FAIL    : {results['pred_pass_gt_fail']}  ← model generated valid SPICE (bonus!)")
print(f"Pred FAIL, GT PASS    : {results['pred_fail_gt_pass']}  ← model errors to investigate")
print("="*60)

# Save failures to inspect
with open("simulation_failures.json", "w") as f:
    json.dump(results["failures"], f, indent=2)
print("\nFailed cases saved to simulation_failures.json")