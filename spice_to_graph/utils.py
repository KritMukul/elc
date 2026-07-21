# spice_to_graph/utils.py
import re
from typing import Dict, Any

def parse_parameters(param_str: str) -> Dict[str, Any]:
    """
    Parses a raw SPICE parameter string into a clean dictionary.
    Extracts explicit key=value pairs and bundles positional arguments.
    """
    params = {}
    positional = []
    
    # Regex breakdown:
    # Group 1 & 2: Matches Key = Value (handles optional spaces)
    # Group 3: Matches any standalone non-whitespace string (positional args)
    pattern = re.compile(r'([a-zA-Z0-9_]+)\s*=\s*([^\s]+)|([^\s]+)')
    
    for match in pattern.finditer(param_str):
        if match.group(1) and match.group(2):
            # It's a Key=Value pair
            params[match.group(1)] = match.group(2)
        elif match.group(3):
            # It's a standalone positional value
            positional.append(match.group(3))
            
    # Bundle positional arguments cleanly for the LLM
    if positional:
        # If there's only one positional arg (common for R, C, L), map it as "value"
        if len(positional) == 1:
            params["value"] = positional[0]
        else:
            params["positional"] = positional
            
    return params