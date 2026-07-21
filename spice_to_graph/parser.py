# spice_to_graph/parser.py
import re
from typing import Dict, Any, List
from .devices import get_device_def
from .utils import parse_parameters

class SpiceParser:
    def __init__(self):
        self.devices = []
        self.nets = set() # Using a set to ensure unique nets

    def _preprocess_lines(self, raw_lines: List[str]) -> List[str]:
        """Handles comments, empty lines, and SPICE line continuations (+)."""
        processed = []
        current_line = ""

        for line in raw_lines:
            # Strip inline comments (often ; or $) and whitespace
            line = re.split(r'[;$]', line)[0].strip()
            
            # Ignore full line comments or empty lines
            if not line or line.startswith('*'):
                continue
            
            # Handle continuation lines
            if line.startswith('+'):
                current_line += " " + line[1:].strip()
            else:
                if current_line:
                    processed.append(current_line)
                current_line = line
                
        if current_line:
            processed.append(current_line)
            
        return processed

    def parse(self, filepath: str) -> Dict[str, Any]:
        """Parses a SPICE file and returns a SKY130-compatible dictionary."""
        self.devices = []
        self.nets = set()
        
        with open(filepath, 'r') as f:
            lines = self._preprocess_lines(f.readlines())

        for line in lines:
            # Skip SPICE directives for now (e.g., .model, .subckt, .end)
            if line.startswith('.') or line.startswith('//') or line.startswith('```'):
                continue
                
            tokens = line.split()
            if not tokens:
                continue
                
            comp_name = tokens[0]
            
            if comp_name.lower() in ['spice', 'plaintext', 'tran', 'plot', 'run', 'endc', 'dc', 'ac']:
                continue
                
            try:
                device_def = get_device_def(comp_name)
            except ValueError:
                # We can now safely suppress the print statement completely 
                # to keep the batch output clean for unfixable junk lines.
                continue

            # --- DYNAMIC SUBCIRCUIT HANDLER ---
            if comp_name.upper().startswith('X'):
                # SPICE subcircuit format: Xname node1 node2 ... subckt_name
                # We will treat everything between the name and the subcircuit model as a pin.
                if len(tokens) < 3:
                    continue # Malformed line
                    
                node_tokens = tokens[1:-1]
                model_name = tokens[-1]
                
                pins = []
                for i, node in enumerate(node_tokens):
                    pins.append({"role": f"pin_{i+1}", "net": node})
                    self.nets.add(node)
                
                self.devices.append({
                    "name": comp_name,
                    "type": "subcircuit",
                    "pins": pins,
                    "params": {"model": model_name}
                })
                continue # Move to the next line

            # --- STANDARD COMPONENT HANDLER ---
            pin_count = len(device_def.pin_roles)
            if len(tokens) < 1 + pin_count:
                # Malformed SPICE line in dataset
                continue
                
            node_tokens = tokens[1:1+pin_count]
            param_tokens = tokens[1+pin_count:]

            pins = []
            for role, node in zip(device_def.pin_roles, node_tokens):
                pins.append({"role": role, "net": node})
                self.nets.add(node)

            params_str = " ".join(param_tokens)
            clean_params = parse_parameters(params_str)

            self.devices.append({
                "name": comp_name,
                "type": device_def.type_name,
                "pins": pins,
                "params": clean_params
            })

        return {
            "nets": list(self.nets),
            "devices": self.devices
        }