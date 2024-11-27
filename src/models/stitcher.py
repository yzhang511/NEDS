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
        out = []
        for idx in range(len(x)):
            tmp = self.stitcher_dict[eid[idx]](x[idx].unsqueeze(0))
            if self.mod in STATIC_VARS:
                tmp = tmp.reshape(1, -1, 2)
            tmp = self.act(tmp) * self.scale
            out.append(self.project_dict[eid[idx]](tmp))
        return torch.cat(out, dim=0).to(x.device)


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
        self.N = max(list(eid_list.values()))
        stitch_decoder_dict = {}
        for key, val in eid_list.items():
            if mod in STATIC_VARS:
                val, mult = OUTPUT_DIM[mod], 1
            elif mod in DYNAMIC_VARS:
                val, mult = OUTPUT_DIM[mod], 1
            else:
                val, mult = self.N, 1
            stitch_decoder_dict[str(key)] = nn.Linear(n_channels * mult, val)
        self.stitch_decoder_dict = nn.ModuleDict(stitch_decoder_dict)

    def forward(self, x, eid):
        x = x.reshape(len(eid), -1, self.P)
        out = []
        for idx in range(len(x)):
            out.append(self.stitch_decoder_dict[eid[idx]](x[idx]))
        return torch.cat(out, dim=0).to(x.device)

