import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import os
import subprocess
import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description="Generate KiCad Schematic via LLM")
    parser.add_argument("--prompt", type=str, required=True, help="Description or SPICE netlist for the circuit")
    parser.add_argument("--project_name", type=str, default="generated_project", help="Name of the KiCad project")
    args = parser.parse_args()

    base_model_id = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    adapter_dir = "./schgen-qwen-lora-final"
    
    print("Loading base model and fine-tuned adapters...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id, 
        torch_dtype=torch.bfloat16, 
        device_map="auto"
    )
    # Merge the trained LoRA layers back into the base LLM structure
    model = PeftModel.from_pretrained(base_model, adapter_dir).to(device)
    model.eval()

    user_prompt = args.prompt

    # Construct the exact same message structure used during training
    # Construct the exact same message structure used during training
    system_instruction = """
You need to complete a user request by outputting executable Python code that generates a KiCad schematic file corresponding to the request. Generate the Python code to edit the schematic file using the KiCad Python API. YOU MUST MAKE SURE THE FINAL CODE ALIGN WITH THE THINKING PROCESS.
###
You have the following functions available to you and can create new functions based on them:
- def add_schematic_symbol(symbol_lib="RF_Module", symbol_name="ESP-WROOM-02", pos_x=150, pos_y=100, reference="U1", value="", rotation=0, mirror:str =None): Add any component symbol from a KiCad library into your schematic. The symbol_lib and symbol_name specify the library and symbol name of the component to add. The pos_x and pos_y specify the position of the center of the symbol in mm. The reference is the unique identifier for the component, e.g., "U1", "R1", "C1". The value is the value of the component, e.g., "10K", "100nF", "ESP32". The rotation is the angle in degrees to rotate the symbol, e.g., 0, 90, 180, 270. The mirror can be "X" or "Y" to flip the symbol according to X or Y axis. If mirror is None, no mirroring is applied.
NOTE:If there is no related information for value, you need to set a value based on your knowledge about what the schematic design. For example, for a pull up resistor, you can set a value of "10K", for a decoupling capacitor, you can set a value of "100nF". value string should NOT include space or `()` or use `TBD`, for example, `12pF (NC)` should be set as "12pF", and `10K (pull up)` should be set as "10K". 

- def get_pin_location(symbol_ref: str, pin_name: str): Get the location of a pin in the schematic. Args: symbol_ref (str): The reference of the symbol or label. pin_name (str): The name or id of the pin, for power symbol or label, this should be "1".

- def add_label(label_pos: list, label_text: str, label_ref: str, label_type: str = "input", text_orient: str ="left"): Add a label to the schematic. The label_pos is a list of two floats [x, y] specifying the position of the label's pin in mm. The label_text is the text of the label, e.g., "IO1", "SDA", "RXD". The label_ref is the unique identifier for the label, e.g., "IO1_0", "SDA_0", "SDA_1". The label_type can be "input", "output", "bidirectional" to specify the type of the label. The text_orient can be "left", "right", "top", or "bottom" to specify the orientation of the text relative to the label pin position.

- def connect_pins(sym_a: str, pin_a: str, sym_b: str, pin_b: str): Create a connection between pin_a of sym_a and pin_b of sym_b. The sym_a and sym_b are the references of the symbols or labels to connect. The pin_a and pin_b are the names or ids of the pins to connect. Note: We treat labels as a kind of symbols, identified by their unique reference. For power symbols and labels, they only have one pin, so pin_a or pin_b should be "1".

- def write_out_all_wires(): Write out all wires of connections in the schematic.

###
NOTE:
1. You should mind the spatial placement of the components. Make sure they are at reasonable positions and ample spacing so that they do not overlap with each other!
2. The size of the schematic is 210 by 297 mm, size of a A4 paper. It uses a X-Y axes based coordinate system. The origin is [0,0] at bottom left corner of the sheet. X axis is horizontal, and Y axis is vertical. To keep the circuit in the center region.
3. You should check the symbol context to see the spatial information, including the size, orientation, pin locations. The center of the symbol is at (0, 0) and the pin locations are relative to the center of the symbol. X axis is horizontal, and Y axis is vertical. For symbol definition, the Y axis points upward, that means higher Y position means higher position, same direction as the schematic coordinate system.
4. The code should be valid Python code with correct indentation and syntax. For example, comment should start with #.
5. You MUST include `from modules.kicad_sch_interface import *` at the very beginning of the script.
6. Explicit Spacing: "Ensure all components are spaced at least 50 units (mm) apart on both the X and Y axes. DO NOT place any two components at the same X or Y coordinate unless they are separated by at least 50 units."
7. Label Placement: "DO NOT place labels (like VDD, GND, +BATT) at the exact same coordinate as a component's center. Labels must be offset by at least 15 units from any component."
8. Power Rails: "You MUST use standard power net names like 'VDD', 'VCC', or 'GND' instead of generic names like '#PWR1' or '#PWR2'."
9. Grid Layout Instruction: "Do not stack components tightly. Calculate positions using a strict grid, e.g., pos_x = 50 * i + 100, pos_y = 50 * j + 100."
10. Allowed Symbols: "You MUST strictly use standard KiCad symbol names from the 'Device' library (e.g., 'R' for resistor, 'C' for capacitor, 'L' for inductor, 'D' for diode, 'Battery' or 'VDC' for voltage source). NEVER use 'analogtobi' or hallucinated names."
"""

    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": user_prompt}
    ]
    
    # Format for the chat template
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(device)
    
    print("Generating code structure from Netlist...")
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs, 
            max_new_tokens=1500,
            temperature=0.1, # Keep temperature low for exact syntax generation
            do_sample=False
        )
    
    # Extract only the newly generated tokens
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    # Extract only the python block if the model outputs markdown
    if "```python" in response:
        response = response.split("```python")[1].split("```")[0].strip()
    elif "```" in response:
        response = response.split("```")[1].strip()

    # Ensure the required import is present for init_project.py
    required_import = "from modules.kicad_sch_interface import *"
    if required_import not in response:
        response = f"{required_import}\n\n" + response

    print("\n--- Generated Python Code Output ---")
    print(response)
    print("------------------------------------\n")
    
    # Save the output to a Python file
    output_script = "output_layout_script.py"
    with open(output_script, "w") as f:
        f.write(response)
    print(f"Saved generated script to {output_script}")
    
    # Automatically run init_project.py to compile the schematic
    print(f"\nInitializing KiCad project '{args.project_name}'...")
    subprocess.run([sys.executable, "init_project.py", args.project_name, output_script, "--overwrite"])

if __name__ == "__main__":
    main()