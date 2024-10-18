import torch
from torch import nn

class StitchEncoder(nn.Module):

    def __init__(self, 
                 eid_list:dict,
                 n_channels:int,
                 scale:int = 1,
                 mod:str = 'ap',
        ):
        super().__init__()

        stitcher_dict = {}
        project_dict = {}
        self.mod = mod
        # iterate key, value pairs in the dictionary
        for key, val in eid_list.items():
            if mod in ['wheel', 'whisker', 'choice', 'block']:
                val = 1
            mult = 100 if mod in ['choice', 'block'] else 1
            # token embedding layer
            stitcher_dict[str(key)] = nn.Linear(int(val), int(val) * 2 * mult)
            # projection layer
            project_dict[str(key)] = nn.Linear(int(val) * 2, n_channels)
        self.stitcher_dict = nn.ModuleDict(stitcher_dict)
        self.project_dict = nn.ModuleDict(project_dict)
        self.scale = scale
        self.act = nn.Softsign()

    def forward(self, x, block_idx):
        x = self.stitcher_dict[block_idx](x)
        if self.mod in ['choice', 'block']:
            x = x.reshape(x.shape[0], -1, 2)
        x = self.act(x) * self.scale
        x = self.project_dict[block_idx](x)
        return x
    
class StitchDecoder(nn.Module):

    def __init__(self,
                 eid_list:list,
                 n_channels:int,
                 mod:str = 'ap',
        ):
        super().__init__()
        self.mod = mod
        stitch_decoder_dict = {}
        for key, val in eid_list.items():
            if mod in ["choice"]:
                val = 2
                mult = 100
            elif mod == "block":
                val = 3
                mult = 100
            elif mod == "wheel" or mod == "whisker":
                val = 1
                mult = 1
            else:
                mult=1
            stitch_decoder_dict[str(key)] = nn.Linear(n_channels * mult, val)
        self.stitch_decoder_dict = nn.ModuleDict(stitch_decoder_dict)

    def forward(self, x, block_idx):
        if self.mod in ["choice", "block"]:
            x = x.flatten(1)
            if len(x.shape) == 2:
                x = x.reshape(-1,x.shape[1] * 100)
        return self.stitch_decoder_dict[block_idx](x)

