import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
from tqdm import tqdm

# ==========================================
# 1. SPATIAL TOPOLOGY EXTRACTION
# ==========================================
class UnionFind:
    def __init__(self):
        self.parent = {}
    def find(self, item):
        if item not in self.parent:
            self.parent[item] = item
        elif self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]
    def union(self, set1, set2):
        root1, root2 = self.find(set1), self.find(set2)
        if root1 != root2:
            self.parent[root2] = root1

def extract_topology(sch_data, coords, tolerance=12.0):
    """
    Extracts the graph edge_index by matching wire coordinates to component 
    centers using a distance tolerance heuristic.
    """
    uf = UnionFind()
    def hash_pt(x, y): return (round(x, 1), round(y, 1))
    
    wire_pts = set()
    
    # 1. Group connected wires into Nets
    for item in sch_data.get('graphicalItems', []):
        if item.get('type') == 'wire':
            pts = item.get('points', [])
            for i in range(len(pts) - 1):
                p1 = hash_pt(pts[i]['x'], pts[i]['y'])
                p2 = hash_pt(pts[i+1]['x'], pts[i+1]['y'])
                uf.union(p1, p2)
                wire_pts.add(p1)
                wire_pts.add(p2)
                
    # 2. Map components to Nets
    net_to_components = defaultdict(list)
    for comp_idx, (cx, cy) in enumerate(coords):
        connected_nets = set()
        
        # Check all wire endpoints. If it falls inside the component's bounding box, connect it.
        for wx, wy in wire_pts:
            if math.hypot(cx - wx, cy - wy) < tolerance:
                connected_nets.add(uf.find((wx, wy)))
                
        for net in connected_nets:
            net_to_components[net].append(comp_idx)
            
    # 3. Build graph cliques
    edge_sources, edge_targets = [], []
    for comp_indices in net_to_components.values():
        unique_comps = list(set(comp_indices))
        for i in range(len(unique_comps)):
            for j in range(len(unique_comps)):
                if i != j:
                    edge_sources.append(unique_comps[i])
                    edge_targets.append(unique_comps[j])
                    
    if not edge_sources:
        return torch.empty((2, 0), dtype=torch.long)
        
    return torch.tensor([edge_sources, edge_targets], dtype=torch.long)

# ==========================================
# 2. JSONL DATASET CLASS
# ==========================================
class KiCadJSONLDataset(Dataset):
    def __init__(self, filepath):
        super().__init__()
        self.data_strings = []
        
        print(f"Loading JSONL from {filepath}...")
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Parsing JSON Lines"):
                if not line.strip(): continue
                try:
                    row_dict = json.loads(line)
                    # Extract the nested schematic JSON string
                    if "schematic_json" in row_dict and row_dict["schematic_json"]:
                        self.data_strings.append(row_dict["schematic_json"])
                except json.JSONDecodeError:
                    pass
        
        print(f"Loaded {len(self.data_strings)} valid schematics.")
        
        # Extended Component Taxonomy based on your JSON dump
        self.comp_types = {
            'R': 0, 'C': 1, 'L': 2, 'D': 3, 'Q': 4, 'U': 5, 'J': 6, 
            'GND': 7, 'VCC': 8, 'LED': 9, 'SW': 10, 'BAT': 11, 'SPK': 12
        }

    def __len__(self):
        return len(self.data_strings)

    def _get_node_feature(self, nickname, entry_name):
        vec = torch.zeros(len(self.comp_types))
        entry_upper = entry_name.upper()
        
        if 'GND' in entry_upper: vec[self.comp_types['GND']] = 1.0
        elif 'VCC' in entry_upper or '+5V' in entry_upper or '+3V3' in entry_upper or '+' in entry_upper: vec[self.comp_types['VCC']] = 1.0
        elif 'LED' in entry_upper or 'WS2812' in entry_upper: vec[self.comp_types['LED']] = 1.0
        elif 'SW_' in entry_upper or 'BUTTON' in entry_upper: vec[self.comp_types['SW']] = 1.0
        elif 'BATTERY' in entry_upper: vec[self.comp_types['BAT']] = 1.0
        elif 'SPEAKER' in entry_upper or 'BUZZER' in entry_upper: vec[self.comp_types['SPK']] = 1.0
        elif entry_upper.startswith('R'): vec[self.comp_types['R']] = 1.0
        elif entry_upper.startswith('C'): vec[self.comp_types['C']] = 1.0
        elif entry_upper.startswith('D'): vec[self.comp_types['D']] = 1.0
        elif entry_upper.startswith('Q'): vec[self.comp_types['Q']] = 1.0
        elif 'CONN' in entry_upper or entry_upper.startswith('J'): vec[self.comp_types['J']] = 1.0
        else: vec[self.comp_types['U']] = 1.0 # Default to IC
        
        return vec

    def __getitem__(self, idx):
        raw_json = self.data_strings[idx]
        sch = json.loads(raw_json)
        
        symbols = sch.get('schematicSymbols', [])
        num_nodes = len(symbols)
        
        node_features = []
        coords = []
        power_mask = torch.zeros(num_nodes, dtype=torch.bool)
        ground_mask = torch.zeros(num_nodes, dtype=torch.bool)
        
        for i, symbol in enumerate(symbols):
            nickname = symbol.get('libraryNickname', '')
            entry = symbol.get('entryName', '')
            
            node_features.append(self._get_node_feature(nickname, entry))
            
            pos = symbol.get('position', {'x': 0.0, 'y': 0.0})
            coords.append([pos['x'], pos['y']])
            
            # Masks for custom loss
            if 'GND' in entry.upper(): ground_mask[i] = True
            elif 'VCC' in entry.upper() or '+' in entry: power_mask[i] = True
                
        edge_index = extract_topology(sch, coords)
        
        x_tensor = torch.stack(node_features) if node_features else torch.empty((0, len(self.comp_types)))
        
        # --- ADD THIS NORMALIZATION BLOCK ---
        raw_coords = torch.tensor(coords, dtype=torch.float)
        
        if raw_coords.size(0) > 0:
            min_vals, _ = torch.min(raw_coords, dim=0)
            max_vals, _ = torch.max(raw_coords, dim=0)
            range_vals = max_vals - min_vals
            # Prevent division by zero if all components share an axis
            range_vals[range_vals == 0] = 1.0 
            
            # Scales all X and Y coordinates to fall exactly between 0.0 and 1.0
            norm_coords = (raw_coords - min_vals) / range_vals
        else:
            norm_coords = raw_coords
            
        y_tensor = norm_coords
        # ------------------------------------

        data = Data(x=x_tensor, edge_index=edge_index, y=y_tensor)
        data.power_mask = power_mask
        data.ground_mask = ground_mask
        return data

# ==========================================
# 3. GRAPH VAE ARCHITECTURE
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
        if self.training:
            return mu + torch.randn_like(torch.exp(0.5 * logvar)) * torch.exp(0.5 * logvar)
        return mu

    def forward(self, x, edge_index):
        h = F.relu(self.conv1(x, edge_index))
        mu = self.conv_mu(h, edge_index)
        logvar = self.conv_logvar(h, edge_index)
        pred_pos = self.decoder(self.reparameterize(mu, logvar))
        return pred_pos, mu, logvar

# ==========================================
# 4. CUSTOM SCHEMATIC LOSS
# ==========================================
def compute_loss(pred_pos, true_pos, mu, logvar, power_mask, ground_mask):
    recon_loss = F.mse_loss(pred_pos, true_pos)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    voltage_penalty = torch.tensor(0.0, device=pred_pos.device)
    power_y = pred_pos[power_mask, 1]
    ground_y = pred_pos[ground_mask, 1]
    
    if len(power_y) > 0 and len(ground_y) > 0:
        # KiCad Y-axis increases downwards, so Power Y should be < Ground Y
        voltage_penalty = F.relu(power_y.mean() - ground_y.mean())
        
    return recon_loss + (0.01 * kl_loss) + (2.0 * voltage_penalty)

# ==========================================
# 5. MAIN TRAINING LOOP
# ==========================================
def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Initializing on: {device}")

    # Pass your .jsonl file path here
    dataset = KiCadJSONLDataset("local_schematics.jsonl") 
    
    # DataLoader utilizing CPU cores
    loader = DataLoader(dataset, batch_size=32, num_workers=14, prefetch_factor=2, shuffle=True)
    
    # in_channels=13 to match the expanded self.comp_types
    model = SchematicGVAE(in_channels=13, hidden_channels=64, latent_dim=32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    model.train()
    epochs = 50
    
    for epoch in range(epochs):
        total_loss = 0
        valid_batches = 0
        
        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{epochs}")
        for batch in pbar:
            batch = batch.to(device)
            if batch.num_nodes == 0: continue
                
            optimizer.zero_grad()
            pred_pos, mu, logvar = model(batch.x, batch.edge_index)
            
            loss = compute_loss(pred_pos, batch.y, mu, logvar, batch.power_mask, batch.ground_mask)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            valid_batches += 1
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        avg_loss = total_loss / valid_batches if valid_batches > 0 else 0
        print(f"=== Epoch {epoch + 1} Complete | Avg Loss: {avg_loss:.4f} ===")

    # ADD THESE TWO LINES HERE (Outside the epoch loop, aligned with 'for')
    torch.save(model.state_dict(), "gvae_weights.pth")
    print("\n✅ Training complete. Weights saved to gvae_weights.pth")

if __name__ == "__main__":
    import torch.multiprocessing as mp
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    
    train()