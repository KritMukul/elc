import json
import torch
import math
from collections import defaultdict
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader

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

class KiCadJSONDataset(Dataset):
    def __init__(self, json_strings_list):
        """
        Args:
            json_strings_list: List of raw JSON strings from the dataset.
        """
        super().__init__()
        self.data_strings = json_strings_list
        
        # Expanded vocabulary based on your JSON dump
        self.comp_types = {
            'R': 0, 'C': 1, 'L': 2, 'D': 3, 'Q': 4, 'U': 5, 'J': 6, 
            'GND': 7, 'VCC': 8, 'LED': 9, 'SW': 10, 'BAT': 11, 'SPK': 12
        }

    def __len__(self):
        return len(self.data_strings)

    def _hash_pt(self, x, y):
        # Round to 1 decimal place to handle KiCad floating point inaccuracies
        return (round(x, 1), round(y, 1))

    def _get_node_feature(self, nickname, entry_name):
        vec = torch.zeros(len(self.comp_types))
        entry_upper = entry_name.upper()
        
        # Heuristics to classify the component type
        if 'GND' in entry_upper: vec[self.comp_types['GND']] = 1.0
        elif 'VCC' in entry_upper or '+5V' in entry_upper or '+3V3' in entry_upper: vec[self.comp_types['VCC']] = 1.0
        elif 'LED' in entry_upper or 'WS2812' in entry_upper: vec[self.comp_types['LED']] = 1.0
        elif 'SW_' in entry_upper or 'BUTTON' in entry_upper: vec[self.comp_types['SW']] = 1.0
        elif 'BATTERY' in entry_upper: vec[self.comp_types['BAT']] = 1.0
        elif 'SPEAKER' in entry_upper or 'BUZZER' in entry_upper: vec[self.comp_types['SPK']] = 1.0
        elif entry_upper.startswith('R'): vec[self.comp_types['R']] = 1.0
        elif entry_upper.startswith('C'): vec[self.comp_types['C']] = 1.0
        elif entry_upper.startswith('D'): vec[self.comp_types['D']] = 1.0
        elif entry_upper.startswith('Q'): vec[self.comp_types['Q']] = 1.0
        elif 'CONN' in entry_upper or entry_upper.startswith('J'): vec[self.comp_types['J']] = 1.0
        else: vec[self.comp_types['U']] = 1.0 # Default to IC / Microcontroller
        
        return vec

    def __getitem__(self, idx):
        raw_json = self.data_strings[idx]
        sch = json.loads(raw_json)
        
        # 1. EXTRACT TOPOLOGY (WIRES TO NETS)
        uf = UnionFind()
        for item in sch.get('graphicalItems', []):
            if item.get('type') == 'wire':
                pts = item.get('points', [])
                if len(pts) >= 2:
                    p1 = self._hash_pt(pts[0]['x'], pts[0]['y'])
                    p2 = self._hash_pt(pts[1]['x'], pts[1]['y'])
                    uf.union(p1, p2)

        # 2. EXTRACT COMPONENTS (NODES)
        symbols = sch.get('schematicSymbols', [])
        num_nodes = len(symbols)
        
        node_features = []
        coords = []
        power_mask = torch.zeros(num_nodes, dtype=torch.bool)
        ground_mask = torch.zeros(num_nodes, dtype=torch.bool)
        
        net_to_components = defaultdict(list)
        
        for i, symbol in enumerate(symbols):
            nickname = symbol.get('libraryNickname', '')
            entry = symbol.get('entryName', '')
            
            # Node features
            node_features.append(self._get_node_feature(nickname, entry))
            
            # Ground truth targets
            pos = symbol.get('position', {'x': 0, 'y': 0})
            coords.append([pos['x'], pos['y']])
            
            # Map component to a Net based on its center coordinate
            # (In a production system, you offset this by actual pin locations)
            comp_hash = self._hash_pt(pos['x'], pos['y'])
            net_root = uf.find(comp_hash)
            net_to_components[net_root].append(i)
            
            # Physics Loss Masks
            if 'GND' in entry.upper(): ground_mask[i] = True
            elif 'VCC' in entry.upper() or '+' in entry: power_mask[i] = True

        # 3. BUILD EDGE INDEX (CLIQUES FOR EACH NET)
        edge_sources, edge_targets = [], []
        for net_root, comp_indices in net_to_components.items():
            unique_comps = list(set(comp_indices))
            if len(unique_comps) > 1:
                for src in unique_comps:
                    for tgt in unique_comps:
                        if src != tgt:
                            edge_sources.append(src)
                            edge_targets.append(tgt)

        if not edge_sources:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        else:
            edge_index = torch.tensor([edge_sources, edge_targets], dtype=torch.long)

        # 4. COMPILE PYG DATA OBJECT
        x_tensor = torch.stack(node_features) if node_features else torch.empty((0, len(self.comp_types)))
        y_tensor = torch.tensor(coords, dtype=torch.float)
        
        data = Data(x=x_tensor, edge_index=edge_index, y=y_tensor)
        data.power_mask = power_mask
        data.ground_mask = ground_mask
        
        return data