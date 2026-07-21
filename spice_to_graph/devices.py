# spice_to_graph/devices.py
from dataclasses import dataclass
from typing import List, Dict

@dataclass
class DeviceDefinition:
    prefix: str
    type_name: str
    pin_roles: List[str]

# Standard SPICE primitives mapped to SKY130 terminology
PRIMITIVES: Dict[str, DeviceDefinition] = {
    "R": DeviceDefinition("R", "resistor", ["pos", "neg"]),
    "C": DeviceDefinition("C", "capacitor", ["pos", "neg"]),
    "L": DeviceDefinition("L", "inductor", ["pos", "neg"]),
    "D": DeviceDefinition("D", "diode", ["anode", "cathode"]),
    "Q": DeviceDefinition("Q", "bjt", ["c", "b", "e"]), 
    "M": DeviceDefinition("M", "mosfet", ["d", "g", "s", "b"]),
    "V": DeviceDefinition("V", "vsource", ["pos", "neg"]),
    "I": DeviceDefinition("I", "isource", ["pos", "neg"]),

    # Phase 2: Dependent Sources
    "E": DeviceDefinition("E", "vcvs", ["pos", "neg", "cp", "cn"]),
    "G": DeviceDefinition("G", "vccs", ["pos", "neg", "cp", "cn"]),
    "F": DeviceDefinition("F", "ccvs", ["pos", "neg", "cp", "cn"]),
    "H": DeviceDefinition("H", "cccs", ["pos", "neg", "cp", "cn"]),
    
    # Subcircuits have variable pins, so we leave the roles empty and handle it dynamically
    "X": DeviceDefinition("X", "subcircuit", []),
}

def get_device_def(component_name: str) -> DeviceDefinition:
    """Returns the device definition based on the first letter of the component name."""
    prefix = component_name[0].upper()
    if prefix in PRIMITIVES:
        return PRIMITIVES[prefix]
    raise ValueError(f"Unsupported device prefix '{prefix}' for component '{component_name}'")