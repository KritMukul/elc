import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
import uuid
import argparse
import numpy as np
import math

# ==========================================
# 1. YOUR MODEL DEFINITION 
# ==========================================
class SchematicGVAE(nn.Module):
    def __init__(self, in_channels, hidden_channels, latent_dim):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=4, concat=False)
        self.conv_mu = GATConv(hidden_channels, latent_dim, heads=1, concat=False)
        self.conv_logvar = GATConv(hidden_channels, latent_dim, heads=1, concat=False)
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 2)
        )

    def reparameterize(self, mu, logvar):
        return mu  # Deterministic inference

    def forward(self, x, edge_index):
        h = F.relu(self.conv1(x, edge_index))
        mu = self.conv_mu(h, edge_index)
        logvar = self.conv_logvar(h, edge_index)
        pred_pos = self.decoder(self.reparameterize(mu, logvar))
        return pred_pos, mu, logvar

# ==========================================
# 2. DATA STRUCTURES (This was missing!)
# ==========================================
COMP_TYPES = {
    'R': 0, 'C': 1, 'L': 2, 'D': 3, 'Q': 4, 'U': 5, 'J': 6, 
    'GND': 7, 'VCC': 8, 'LED': 9, 'SW': 10, 'BAT': 11, 'SPK': 12
}

class Component:
    def __init__(self, ref, kicad_id, pins):
        self.ref = ref
        self.kicad_id = kicad_id
        self.pins = pins # List of net names
        self.x = 0.0
        self.y = 0.0
        self.type_idx = COMP_TYPES['U'] # Default

# ==========================================
# 3. SPICE TO GRAPH TRANSLATOR (PDK SYMBOLS)
# ==========================================
def parse_spice_to_graph(filepath):
    components = []
    nets = {}  
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        line = line.strip()
        if not line or line.startswith('*') or line.startswith('.') or line.upper().startswith('MEAS'):
            continue
            
        parts = line.split()
        if not parts:
            continue
            
        ref = parts[0].upper()
        comp = None
        
        if (ref.startswith('M') or ref.startswith('X')) and ("nfet" in line.lower() or "pfet" in line.lower()): 
            is_nfet = "nfet" in line.lower()
            kicad_id = "sky130_fd_pr:nfet_01v8" if is_nfet else "sky130_fd_pr:pfet_01v8"
            if len(parts) >= 5:
                comp = Component(ref, kicad_id, parts[1:5])
                comp.type_idx = COMP_TYPES['Q']
                
        elif ref.startswith('R'):
            if len(parts) >= 3:
                comp = Component(ref, "Device:R", parts[1:3])
                comp.type_idx = COMP_TYPES['R']
                
        elif ref.startswith('C'):
            if len(parts) >= 3:
                comp = Component(ref, "Device:C", parts[1:3])
                comp.type_idx = COMP_TYPES['C']
                
        elif ref.startswith('V'):
            if len(parts) >= 3:
                comp = Component(ref, "Simulation_SPICE:VDC", parts[1:3])
                comp.type_idx = COMP_TYPES['VCC'] if 'VDD' in line.upper() else COMP_TYPES['U']
            
        if comp:
            comp_idx = len(components)
            components.append(comp)
            for net in comp.pins:
                if net == "0": 
                    net = "GND"
                if net not in nets:
                    nets[net] = []
                nets[net].append(comp_idx)
                
    x_tensor = torch.zeros((len(components), len(COMP_TYPES)))
    for i, comp in enumerate(components):
        x_tensor[i][comp.type_idx] = 1.0
        
    edge_sources, edge_targets = [], []
    for net, comp_indices in nets.items():
        unique_comps = list(set(comp_indices))
        for i in range(len(unique_comps)):
            for j in range(len(unique_comps)):
                if i != j:
                    edge_sources.append(unique_comps[i])
                    edge_targets.append(unique_comps[j])
                    
    edge_index = torch.tensor([edge_sources, edge_targets], dtype=torch.long)
    return components, x_tensor, edge_index

# ==========================================
# 4. KICAD GENERATOR (PDK SYMBOLS & ROUTING)
# ==========================================
def write_kicad_sch(components, output_path):
    header = """(kicad_sch (version 20230121) (generator gvae_placer)
  (paper "A4")
  (lib_symbols
    (symbol "Device:R" (pin_names (offset 0) hide) (exclude_from_sim no) (in_bom yes) (on_board yes))
    (symbol "Device:C" (pin_names (offset 0) hide) (exclude_from_sim no) (in_bom yes) (on_board yes))
    (symbol "Simulation_SPICE:VDC" (pin_names (offset 0) hide) (exclude_from_sim no) (in_bom yes) (on_board yes))
    (symbol "sky130_fd_pr:nfet_01v8" (pin_names (offset 0) hide) (exclude_from_sim no) (in_bom yes) (on_board yes))
    (symbol "sky130_fd_pr:pfet_01v8" (pin_names (offset 0) hide) (exclude_from_sim no) (in_bom yes) (on_board yes))
  )\n"""
  
    net_pins = {} 

    with open(output_path, 'w') as f:
        f.write(header)
        for comp in components:
            sym_uuid = str(uuid.uuid4())
            f.write(f"""
  (symbol (lib_id "{comp.kicad_id}") (at {comp.x:.2f} {comp.y:.2f} 0) (unit 1)
    (in_bom yes) (on_board yes) (dnp no)
    (uuid "{sym_uuid}")
    (property "Reference" "{comp.ref}" (at {comp.x + 5:.2f} {comp.y - 7:.2f} 0)
      (effects (font (size 1.27 1.27)))
    )
  )""")
            
            for idx, net_name in enumerate(comp.pins):
                pin_num = idx + 1
                offset_x = comp.x - 7.5 if pin_num % 2 != 0 else comp.x + 7.5
                offset_y = comp.y - 3.81 + (2.54 * pin_num)
                
                if net_name not in net_pins:
                    net_pins[net_name] = []
                net_pins[net_name].append((offset_x, offset_y))
                
                f.write(f"""
  (wire (pts (xy {comp.x:.2f} {comp.y:.2f}) (xy {offset_x:.2f} {offset_y:.2f}))
    (stroke (width 0) (type default))
    (uuid "{str(uuid.uuid4())}")
  )""")

        POWER_NETS = ["VDD", "VSS", "GND", "0"]
        
        for net_name, pins in net_pins.items():
            if net_name.upper() in POWER_NETS or len(pins) == 1:
                for px, py in pins:
                    f.write(f"""
  (global_label "{net_name}" (shape input) (at {px:.2f} {py:.2f} 0)
    (effects (font (size 1.27 1.27)) (justify left))
    (uuid "{str(uuid.uuid4())}")
  )""")
            else:
                for i in range(len(pins) - 1):
                    x1, y1 = pins[i]
                    x2, y2 = pins[i+1]
                    
                    f.write(f"""
  (wire (pts (xy {x1:.2f} {y1:.2f}) (xy {x1:.2f} {y2:.2f}))
    (stroke (width 0) (type default))
    (uuid "{str(uuid.uuid4())}")
  )
  (wire (pts (xy {x1:.2f} {y2:.2f}) (xy {x2:.2f} {y2:.2f}))
    (stroke (width 0) (type default))
    (uuid "{str(uuid.uuid4())}")
  )""")
                    
                    f.write(f"""
  (label "{net_name}" (at {(x1+x2)/2:.2f} {y2:.2f} 0)
    (effects (font (size 1.27 1.27)) (justify left))
    (uuid "{str(uuid.uuid4())}")
  )""")

        f.write("\n)\n")

# ==========================================
# 5. INFERENCE EXECUTION
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Use GVAE to place a SPICE netlist into KiCad")
    parser.add_argument("--spice_file", type=str, default="qwen_vco.spice", help="Input SPICE file")
    parser.add_argument("--weights", type=str, default="gvae_weights.pth", help="Trained GVAE weights")
    parser.add_argument("--output", type=str, default="qwen_vco_ai_placed.kicad_sch", help="Output KiCad schematic")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SchematicGVAE(in_channels=13, hidden_channels=64, latent_dim=32).to(device)
    
    print(f"Loading GVAE weights from {args.weights}...")
    try:
        model.load_state_dict(torch.load(args.weights, map_location=device))
    except FileNotFoundError:
        print(f"Error: {args.weights} not found.")
        exit(1)
        
    model.eval()
    
    print(f"Parsing SPICE topology from {args.spice_file}...")
    components, x_tensor, edge_index = parse_spice_to_graph(args.spice_file)
    x_tensor, edge_index = x_tensor.to(device), edge_index.to(device)
    
    print("Predicting component placements...")
    with torch.no_grad():
        pred_pos, _, _ = model(x_tensor, edge_index)
        
    pred_pos = pred_pos.cpu().numpy()
    
    # 1. Noise injection for tie-breaking
    np.random.seed(42)
    pred_pos += np.random.normal(0, 1e-4, pred_pos.shape)
    
    # 2. Min-Max Normalization
    min_val = pred_pos.min(axis=0)
    max_val = pred_pos.max(axis=0)
    range_val = max_val - min_val
    range_val[range_val < 1e-6] = 1.0 
    
    pred_pos = (pred_pos - min_val) / range_val
    
    SHEET_W, SHEET_H = 200.0, 130.0
    OFFSET_X, OFFSET_Y = 50.0, 50.0
    
    for i, comp in enumerate(components):
        comp.x = (pred_pos[i][0] * SHEET_W) + OFFSET_X
        comp.y = (pred_pos[i][1] * SHEET_H) + OFFSET_Y

    # 3. Collision Resolution Loop
    print("Resolving component collisions...")
    MIN_SPACING = 50.0 
    
    for _ in range(100):
        moved = False
        for i in range(len(components)):
            for j in range(i + 1, len(components)):
                c1, c2 = components[i], components[j]
                dx = c1.x - c2.x
                dy = c1.y - c2.y
                dist = math.hypot(dx, dy)
                
                if dist < MIN_SPACING:
                    if dist < 0.001: 
                        dx, dy = np.random.rand(), np.random.rand()
                        dist = math.hypot(dx, dy)
                        
                    push = (MIN_SPACING - dist) / 2.0
                    push_x = (dx / dist) * push
                    push_y = (dy / dist) * push
                    
                    c1.x += push_x
                    c1.y += push_y
                    c2.x -= push_x
                    c2.y -= push_y
                    moved = True
                    
        if not moved:
            break
            
    write_kicad_sch(components, args.output)
    print(f"✅ Generated properly spaced AI-placed schematic: {args.output}")