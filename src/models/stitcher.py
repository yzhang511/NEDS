import numpy as np
import torch
from torch import nn

STATIC_VARS = ["choice", "block"]
DYNAMIC_VARS = ["wheel", "whisker"]
OUTPUT_DIM = {"choice": 2, "block": 3, "wheel": 1, "whisker": 1}

class StitchEncoder(nn.Module):
    def __init__(self, 
         eid_list: dict,
         n_channels: int,
         scale: int=1,
         mod: str="spike",
         max_F: int=100,
    ):
        super().__init__()

        self.mod = mod
        self.P = n_channels
        self.max_F = max_F
        self.N = max(list(eid_list.values()))
        stitcher_dict, project_dict = {}, {}
        for key, val in eid_list.items():
            val = 1 if mod in STATIC_VARS + DYNAMIC_VARS else self.N
            mult = max_F if mod in STATIC_VARS else 1
            # token embedding layer
            stitcher_dict[str(key)] = nn.Linear(int(val), int(val) * 2 * mult)
            # projection layer
            project_dict[str(key)] = nn.Linear(int(val) * 2, n_channels)
        self.stitcher_dict = nn.ModuleDict(stitcher_dict)
        self.project_dict = nn.ModuleDict(project_dict)
        self.scale = scale
        self.act = nn.Softsign()

    def forward(self, x, eid):
        eid = np.array(eid)
        unique_eids = np.unique(eid)
        out = torch.zeros((len(x), self.max_F, self.P), device=x.device)
        for group_eid in unique_eids:
            mask = torch.tensor(np.argwhere(eid==group_eid), device=x.device).squeeze()
            x_group = x[mask]
            stitched = self.stitcher_dict[group_eid](x_group)
            if self.mod in STATIC_VARS:
                stitched = stitched.reshape(stitched.shape[0], -1, 2)
            stitched = self.act(stitched) * self.scale
            out[mask] = self.project_dict[group_eid](stitched)
        return out


class StitchDecoder(nn.Module):
    def __init__(self,
         eid_list: list,
         n_channels: int,
         mod:str="spike",
         max_F: int=100,
    ):
        super().__init__()
        
        self.mod = mod
        self.max_F = max_F
        self.P = n_channels
        max_num_neuron = max(list(eid_list.values()))
        stitch_decoder_dict = {}
        for key, val in eid_list.items():
            if mod in STATIC_VARS:
                val, mult = OUTPUT_DIM[mod], 1
            elif mod in DYNAMIC_VARS:
                val, mult = OUTPUT_DIM[mod], 1
            else:
                val, mult = max_num_neuron, 1
            stitch_decoder_dict[str(key)] = nn.Linear(n_channels * mult, val)
        self.stitch_decoder_dict = nn.ModuleDict(stitch_decoder_dict)
        self.N = max_num_neuron if mod == "spike" else val

    def forward(self, x, eid):
        x = x.reshape((len(eid), -1, self.P))
        B, T, _ = x.size()
        eid = np.array(eid)
        unique_eids = np.unique(eid)
        out = torch.zeros((B,T,self.N), device=x.device)
        for group_eid in unique_eids:
            mask = torch.tensor(np.argwhere(eid==group_eid), device=x.device).squeeze()
            x_group = x[mask]
            out[mask] = self.stitch_decoder_dict[group_eid](x_group)
        return out
    
    

