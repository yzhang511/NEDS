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
        # iterate key, value pairs in the dictionary
        for key, val in eid_list.items():
            if mod == "behavior":
                if len(eid_list) == 1:
                    # include discrete and continuous variables in single session
                    val = 4
                else:
                    # include only continuous variables in multi-session
                    val = 2
            # token embedding layer
            stitcher_dict[str(key)] = nn.Linear(int(val), int(val) * 2)
            # projection layer
            project_dict[str(key)] = nn.Linear(int(val) * 2, n_channels)
        self.stitcher_dict = nn.ModuleDict(stitcher_dict)
        self.project_dict = nn.ModuleDict(project_dict)
        self.scale = scale
        self.act = nn.Softsign()

    def forward(self, x, block_idx):
        x = self.stitcher_dict[block_idx](x)
        x = self.act(x) * self.scale
        x = self.project_dict[block_idx](x)
        return x
    
class StitchDecoder(nn.Module):

    def __init__(self,
                 eid_list:list,
                 n_channels:int,
                 mod:str = 'ap',
                 p:float = 0.5,
        ):
        super().__init__()

        stitch_decoder_dict = {}
        for key, val in eid_list.items():
            if mod in ["behavior", "choice"]:
                val = 2
            elif mod == "block":
                val = 3
            stitch_decoder_dict[str(key)] = nn.Linear(n_channels, val)
        self.stitch_decoder_dict = nn.ModuleDict(stitch_decoder_dict)
        # self.dropout = nn.Dropout(p) if mod in ["choice", "block"] else nn.Identity()

    def forward(self, x, block_idx):
        # x = self.dropout(x)
        return self.stitch_decoder_dict[block_idx](x)

