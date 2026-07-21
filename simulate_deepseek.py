import json
import torch
import subprocess
import tempfile
import os
from datasets import load_dataset
from transformers import PreTrainedTokenizerFast
from peft import PeftModel

# ── Config ─────────────────────────────────────────────────────────────────
BASE_MODEL_ID   = "deepseek-ai/deepseek-coder-6.7b-instruct"
ADAPTER_PATH    = "./deepseek_final_circuit_adapter"
DATASET_PATH    = "master_parallel_dataset.json"
SKY130_LIB_PATH = "/usr/local/lib/python3.10/dist-packages/sky130/src/sky130_fd_pr/models/sky130.lib.spice"
NUM_SAMPLES     = 200   # match the Qwen eval for direct comparison

# ── 1. Load model ─────────────────────────────────────────────────────────
print("Loading tokenizer and model...")
tokenizer = PreTrainedTokenizerFast.from_pretrained(
    "deepseek-ai/deepseek-coder-6.7b-instruct"
    # No trust_remote_code, no local_files_only, no legacy flags
)

# Crucial for DeepSeek when using the Fast class directly
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    dtype=torch.bfloat16,
    device_map="cuda",
    trust_remote_code=True,
    local_files_only=True
)
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.eval()

# ── 2. Load the SAME test split (seed=42) used for the Qwen eval ─────────
print("Loading test split...")
dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
dataset_splits = dataset.train_test_split(test_size=0.05, seed=42)
test_split = dataset_splits["test"]
print(f"Test split size: {len(test_split)}")

# ── 3. Simulation helper (identical to the Qwen eval script) ─────────────
def run_ngspice(spice_text: str) -> dict:
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
        passed = result.returncode == 0
        return {"passed": passed, "returncode": result.returncode, "stderr": result.stderr[:300]}
    except subprocess.TimeoutExpired:
        return {"passed": False, "returncode": -1, "stderr": "TIMEOUT"}
    finally:
        os.unlink(tmp_path)

# ── 4. Generate SPICE using DeepSeek's own chat template ─────────────────
def generate_spice(graph_input: dict) -> str:
    graph_str = json.dumps(graph_input)
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
    # add_generation_prompt=True -> matches the official inference usage,
    # leaves the assistant turn open for the model to fill in.
    # return_dict=True + **inputs unpacking is the robust pattern -- avoids
    # ambiguity over whether apply_chat_template returns a raw tensor or a
    # BatchEncoding dict (this varies by tokenizer/transformers version).
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=2048,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )

    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True)

# ── 5. Eval loop (identical structure to Qwen eval, for direct comparison) ─
results = {
    "predicted_pass": 0,
    "predicted_fail": 0,
    "groundtruth_pass": 0,
    "groundtruth_fail": 0,
    "both_pass": 0,
    "both_fail": 0,
    "pred_pass_gt_fail": 0,
    "pred_fail_gt_pass": 0,
    "failures": []
}

print(f"\nRunning simulation eval on {NUM_SAMPLES} examples (DeepSeek-Coder-6.7B)...\n")

for idx, example in enumerate(test_split.select(range(NUM_SAMPLES))):
    predicted_spice = generate_spice(example["graph_input"])
    ground_truth_spice = example["spice_output"]

    pred_result = run_ngspice(predicted_spice)
    gt_result   = run_ngspice(ground_truth_spice)

    pred_passed = pred_result["passed"]
    gt_passed   = gt_result["passed"]

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
        if len(results["failures"]) < 20:
            results["failures"].append({
                "idx": idx,
                "predicted": predicted_spice[:500],
                "ground_truth": ground_truth_spice[:500],
                "stderr": pred_result["stderr"]
            })

    if (idx + 1) % 10 == 0:
        so_far = idx + 1
        print(
            f"[{so_far}/{NUM_SAMPLES}] "
            f"Pred pass rate: {results['predicted_pass']/so_far:.1%} | "
            f"GT pass rate: {results['groundtruth_pass']/so_far:.1%}"
        )

# ── 6. Final report ───────────────────────────────────────────────────────
print("\n" + "="*60)
print("DEEPSEEK-CODER-6.7B SIMULATION EVALUATION REPORT")
print("="*60)
print(f"Total evaluated       : {NUM_SAMPLES}")
print(f"Predicted PASS        : {results['predicted_pass']} ({results['predicted_pass']/NUM_SAMPLES:.1%})")
print(f"Predicted FAIL        : {results['predicted_fail']} ({results['predicted_fail']/NUM_SAMPLES:.1%})")
print(f"Ground Truth PASS     : {results['groundtruth_pass']} ({results['groundtruth_pass']/NUM_SAMPLES:.1%})")
print(f"Ground Truth FAIL     : {results['groundtruth_fail']} ({results['groundtruth_fail']/NUM_SAMPLES:.1%})")
print("-"*60)
print(f"Both PASS             : {results['both_pass']}")
print(f"Both FAIL             : {results['both_fail']}  (dataset issue, not model's fault)")
print(f"Pred PASS, GT FAIL    : {results['pred_pass_gt_fail']}")
print(f"Pred FAIL, GT PASS    : {results['pred_fail_gt_pass']}  (model errors to investigate)")
print("="*60)
print("\nCompare directly against your Qwen2.5-Coder-7B results:")
print("  Qwen:     Predicted pass rate ~88%,  Ground truth pass rate ~99%")
print(f"  DeepSeek: Predicted pass rate {results['predicted_pass']/NUM_SAMPLES:.1%},  "
      f"Ground truth pass rate {results['groundtruth_pass']/NUM_SAMPLES:.1%}")

with open("deepseek_simulation_failures.json", "w") as f:
    json.dump(results["failures"], f, indent=2)
print("\nFailed cases saved to deepseek_simulation_failures.json")