import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from collections import defaultdict
from torch.utils.data import IterableDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
from datasets import load_dataset
from tqdm import tqdm

# ==========================================
# 1. TOPOLOGY EXTRACTION (UNION-FIND)
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
        root1 = self.find(set1)
        root2 = self.find(set2)
        if root1 != root2:
            self.parent[root2] = root1

def extract_topology(sch_data, coords):
    uf = UnionFind()
    
    def hash_pt(x, y):
        return (round(x, 2), round(y, 2))

    wires = sch_data.get('wires', [])
    for wire in wires:
        pts = wire.get('pts', [])
        if len(pts) >= 2:
            pt1 = hash_pt(pts[0]['x'], pts[0]['y'])
            pt2 = hash_pt(pts[1]['x'], pts[1]['y'])
            uf.union(pt1, pt2)
            for i in range(1, len(pts) - 1):
                uf.union(hash_pt(pts[i]['x'], pts[i]['y']), hash_pt(pts[i+1]['x'], pts[i+1]['y']))

    lib_symbols = {sym['uuid']: sym for sym in sch_data.get('lib_symbols', [])}
    net_to_components = defaultdict(list)
    symbols = sch_data.get('symbols', [])
    
    for comp_idx, symbol in enumerate(symbols):
        lib_id = symbol.get('lib_id')
        lib_sym = next((ls for ls in lib_symbols.values() if ls.get('name') == lib_id), None)
        
        comp_x = symbol['at']['x']
        comp_y = symbol['at']['y']
        angle_deg = symbol['at'].get('angle', 0.0)
        angle_rad = math.radians(angle_deg)
        cos_a = round(math.cos(angle_rad), 3)
        sin_a = round(math.sin(angle_rad), 3)

        if lib_sym and 'pins' in lib_sym:
            for pin in lib_sym['pins']:
                pin_rel_x = pin['at']['x']
                pin_rel_y = pin['at']['y']
                
                abs_x = comp_x + (pin_rel_x * cos_a - pin_rel_y * sin_a)
                abs_y = comp_y + (pin_rel_x * sin_a + pin_rel_y * cos_a)
                
                net_root = uf.find(hash_pt(abs_x, abs_y))
                net_to_components[net_root].append(comp_idx)
        else:
            net_root = uf.find(hash_pt(comp_x, comp_y))
            net_to_components[net_root].append(comp_idx)

    edge_sources = []
    edge_targets = []
    for net_root, comp_indices in net_to_components.items():
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
# 2. STREAMABLE PYTORCH DATASET
# ==========================================
class StreamableSchematicDataset(IterableDataset):
    def __init__(self, hf_iterable_dataset):
        self.data = hf_iterable_dataset
        self.comp_types = {'R': 0, 'C': 1, 'L': 2, 'Q': 3, 'U': 4, 'D': 5, 'J': 6, 'GND': 7, 'VCC': 8}

    def _get_feature_vector(self, lib_id):
        vec = torch.zeros(len(self.comp_types))
        ref_prefix = lib_id.split(':')[-1][0].upper() if ':' in lib_id else 'U'
        
        if 'GND' in lib_id.upper():
            vec[self.comp_types['GND']] = 1.0
        elif 'V' in lib_id.upper() or '+5' in lib_id.upper() or '+3' in lib_id.upper():
            vec[self.comp_types['VCC']] = 1.0
        elif ref_prefix in self.comp_types:
            vec[self.comp_types[ref_prefix]] = 1.0
        else:
            vec[self.comp_types['U']] = 1.0 
        return vec

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        
        if worker_info is not None:
            # Shard data so each core handles a unique part of the stream
            sharded_data = self.data.shard(num_shards=worker_info.num_workers, index=worker_info.id)
            worker_id = worker_info.id
        else:
            sharded_data = self.data
            worker_id = 0

        for idx, row in enumerate(sharded_data):
            # Heartbeat (reduced frequency)
            if idx % 50 == 0:
                print(f"[Worker {worker_id}] Fetched row {idx}...")
                
            if row['schematic_json'] is None:
                continue
                
            try:
                sch_data = json.loads(row['schematic_json'])
            except json.JSONDecodeError:
                continue 
                
            symbols = sch_data.get('symbols', [])
            num_nodes = len(symbols)
            
            if num_nodes < 2:
                continue
            
            node_features = []
            coords = []
            power_mask = torch.zeros(num_nodes, dtype=torch.bool)
            ground_mask = torch.zeros(num_nodes, dtype=torch.bool)
            
            for i, symbol in enumerate(symbols):
                lib_id = symbol.get('lib_id', '')
                node_features.append(self._get_feature_vector(lib_id))
                
                x, y = symbol['at']['x'], symbol['at']['y']
                coords.append([x, y])
                
                if 'GND' in lib_id.upper():
                    ground_mask[i] = True
                elif 'V' in lib_id.upper() or '+5' in lib_id.upper() or '+3' in lib_id.upper():
                    power_mask[i] = True
                    
            edge_index = extract_topology(sch_data, coords)

            x_tensor = torch.stack(node_features)
            y_tensor = torch.tensor(coords, dtype=torch.float)
            
            data = Data(x=x_tensor, edge_index=edge_index, y=y_tensor)
            data.power_mask = power_mask
            data.ground_mask = ground_mask
            
            yield data

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
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def forward(self, x, edge_index):
        h = F.relu(self.conv1(x, edge_index))
        mu = self.conv_mu(h, edge_index)
        logvar = self.conv_logvar(h, edge_index)
        
        z = self.reparameterize(mu, logvar)
        pred_pos = self.decoder(z)
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
        voltage_penalty = F.relu(power_y.mean() - ground_y.mean())
        
    return recon_loss + (0.01 * kl_loss) + (2.0 * voltage_penalty)

# ==========================================
# 5. MAIN TRAINING LOOP
# ==========================================
def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("Connecting to Hugging Face Stream (Text/JSON only)...")
    hf_dataset = load_dataset(
        "bshada/open-schematics", 
        split="train", 
        streaming=True
    ).select_columns(['schematic_json'])
    
    dataset = StreamableSchematicDataset(hf_dataset)
    
    # 14 workers spun up using the 'spawn' method
    loader = DataLoader(
        dataset, 
        batch_size=32,
        num_workers=14,
        prefetch_factor=2 
    ) 
    
    model = SchematicGVAE(in_channels=9, hidden_channels=64, latent_dim=32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    model.train()
    print("Streaming initialized. Starting training...")
    
    epochs = 50
    steps_per_epoch = 1000 
    
    data_iterator = iter(loader)
    
    for epoch in range(epochs):
        total_loss = 0
        valid_batches = 0
        
        pbar = tqdm(range(steps_per_epoch), desc=f"Epoch {epoch + 1}/{epochs}")
        
        for step in pbar:
            try:
                batch = next(data_iterator)
            except StopIteration:
                data_iterator = iter(loader)
                batch = next(data_iterator)
                
            batch = batch.to(device)
            
            if batch.num_nodes == 0:
                continue
                
            optimizer.zero_grad()
            pred_pos, mu, logvar = model(batch.x, batch.edge_index)
            
            loss = compute_loss(
                pred_pos, 
                batch.y, 
                mu, 
                logvar, 
                batch.power_mask, 
                batch.ground_mask
            )
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            valid_batches += 1
            
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        avg_loss = total_loss / valid_batches if valid_batches > 0 else 0
        print(f"=== Epoch {epoch + 1} Complete | Avg Loss: {avg_loss:.4f} ===")

# Force PyTorch to spawn fresh processes to avoid network socket corruption
if __name__ == "__main__":
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    
    train()