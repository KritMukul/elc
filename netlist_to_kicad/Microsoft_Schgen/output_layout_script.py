# Auto-generated schematic symbols
import sys
import os

# Get project path and import kicad schematic interface
PROJECT_PATH = os.environ['PROJECT_PATH']
sys.path.append(PROJECT_PATH)
from modules.kicad_sch_interface import *

### Placing center symbol 1 : Device:Q_NMOS_DGS###

center_x_1, center_y_1 = 150.0, 110.0

add_schematic_symbol(symbol_lib="Device", symbol_name="Q_NMOS_DGS", pos_x=center_x_1, pos_y=center_y_1, reference="XM1", value="Q_NMOS_DGS", rotation=0, mirror="None")

### Placing other symbols in the Schematic with respect to the center symbol 1###

add_schematic_symbol(symbol_lib="power", symbol_name="+BATT", pos_x=center_x_1 + (-20), pos_y=center_y_1 + (10), reference="#PWR1", value="+BATT", rotation=0, mirror="None")
add_schematic_symbol(symbol_lib="Device", symbol_name="Q_PMOS_DGS", pos_x=center_x_1 + (0), pos_y=center_y_1 + (10), reference="XM2", value="Q_PMOS_DGS", rotation=0, mirror="None")
add_schematic_symbol(symbol_lib="Device", symbol_name="Q_PMOS_DGS", pos_x=center_x_1 + (0), pos_y=center_y_1 + (-10), reference="XM3", value="Q_PMOS_DGS", rotation=0, mirror="None")
add_schematic_symbol(symbol_lib="Device", symbol_name="Q_PMOS_DGS", pos_x=center_x_1 + (0), pos_y=center_y_1 + (-20), reference="XM4", value="Q_PMOS_DGS", rotation=0, mirror="None")
add_schematic_symbol(symbol_lib="Device", symbol_name="R", pos_x=center_x_1 + (15), pos_y=center_y_1 + (0), reference="R1", value="10k", rotation=0, mirror="None")
add_schematic_symbol(symbol_lib="Device", symbol_name="R", pos_x=center_x_1 + (15), pos_y=center_y_1 + (-10), reference="R2", value="10k", rotation=0, mirror="None")
add_schematic_symbol(symbol_lib="Device", symbol_name="R", pos_x=center_x_1 + (15), pos_y=center_y_1 + (-20), reference="R3", value="10k", rotation=0, mirror="None")
add_schematic_symbol(symbol_lib="Device", symbol_name="R", pos_x=center_x_1 + (15), pos_y=center_y_1 + (-30), reference="R4", value="10k", rotation=0, mirror="None")
add_schematic_symbol(symbol_lib="Device", symbol_name="C", pos_x=center_x_1 + (25), pos_y=center_y_1 + (-30), reference="CL", value="1uF", rotation=0, mirror="None")
add_schematic_symbol(symbol_lib="power", symbol_name="GND", pos_x=center_x_1 + (25), pos_y=center_y_1 + (-40), reference="#PWR2", value="GND", rotation=0, mirror="None")

### Placing all global labels in the Schematic and connect them to the neighbor pin ###

# Add label VCONT1 next to XM2 pin G 
x_XM2_2, y_XM2_2 = get_pin_location(symbol_ref="XM2", pin_name="G")
add_label(label_pos=[x_XM2_2+(-10), y_XM2_2+(0)], label_text="VCONT1", label_ref="VCONT1_0", label_type="input", text_orient="left")
# Connecting Label VCONT1 label_id:0 to XM2 pin G (Pin ID 2 -- Name G)
connect_pins("VCONT1_0", "1", "XM2", "G")


### Connecting all wires in the Schematic ###


# Connecting R4 pin 1 (Pin ID 1 -- Name None) to CL pin 1 (Pin ID 1 -- Name None)
connect_pins("R4", "1", "CL", "1")

# Connecting #PWR1 pin +BATT (Pin ID 1 -- Name +BATT) to XM1 pin S (Pin ID 2 -- Name S)
connect_pins("#PWR1", "+BATT", "XM1", "S")

# Connecting XM1 pin D (Pin ID 1 -- Name D) to R1 pin 1 (Pin ID 1 -- Name None)
connect_pins("XM1", "D", "R1", "1")

# Connecting XM2 pin S (Pin ID 1 -- Name S) to R2 pin 1 (Pin ID 1 -- Name None)
connect_pins("XM2", "S", "R2", "1")

# Connecting XM3 pin S (Pin ID 1 -- Name S) to R3 pin 1 (Pin ID 1 -- Name None)
connect_pins("XM3", "S", "R3", "1")

# Connecting XM4 pin S (Pin ID 1 -- Name S) to R4 pin 1 (Pin ID 1 -- Name None)
connect_pins("XM4", "S", "R4", "1")

# Connecting XM1 pin D (Pin ID 1 -- Name D) to XM2 pin D (Pin ID 2 -- Name D)
connect_pins("XM1", "D", "XM2", "D")

# Connecting XM2 pin D (Pin ID 2 -- Name D) to XM3 pin D (Pin ID 2 -- Name D)
connect_pins("XM2", "D", "XM3", "D")

# Connecting XM3 pin D (Pin ID 2 -- Name D) to XM4 pin D (Pin ID 2 -- Name D)
connect_pins("XM3", "D", "XM4", "D")

# Connecting XM4 pin D (Pin ID 2 -- Name D) to CL pin 2 (Pin ID 2 -- Name None)
connect_pins("XM4", "D", "CL", "2")

# Connecting XM1 pin S (Pin ID 2 -- Name S) to R1 pin 2 (Pin ID 2 -- Name None)
connect_pins("XM1", "S", "R1", "2")

# Connecting XM2 pin S (Pin ID 1 -- Name S) to R2 pin 2 (Pin ID 2 -- Name None)
connect_pins("XM2", "S", "R2", "2")

# Connecting XM3 pin S (Pin ID 1 -- Name S) to R3 pin 2 (Pin ID 2 -- Name None)
connect_pins("XM3", "S", "R3", "2")

# Connecting XM4 pin S (Pin ID 1