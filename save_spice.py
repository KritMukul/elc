import json

# Assuming your Qwen eval script saved the predictions to a JSON file
results_file = "qwen_eval_results.json" 

with open(results_file, 'r') as f:
    data = json.load(f)

# Find a specific circuit (or just grab the first one that passed ngspice)
target_circuit = "analogtobi_0535"
for item in data:
    if target_circuit in item.get("prompt", "") or target_circuit in item.get("prediction", ""):
        with open(f"{target_circuit}_qwen.spice", "w") as out:
            out.write(item["prediction"])
        print(f"Saved {target_circuit}_qwen.spice")
        break