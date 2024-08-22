import torch
from torch import nn

class StitchEncoder(nn.Module):

    def __init__(self, 
                 eid_list:dict,
                 n_channels:int,
                 scale:int = 1,
                 stitcher_type:str = 'spike',
                 behavior_channel:int = 2):
        super().__init__()

        stitcher_dict = {}
        project_dict = {}
        # iterate key, value pairs in the dictionary
        if stitcher_type == 'spike':
            for key, val in eid_list.items():
                # token embedding layer
                stitcher_dict[str(key)] = nn.Linear(int(val), int(val) * 2)
                # projection layer
                project_dict[str(key)] = nn.Linear(int(val) * 2, n_channels)
        elif stitcher_type == 'behavior':
            for key, _ in eid_list.items():
                # token embedding layer
                stitcher_dict[str(key)] = nn.Linear(behavior_channel, behavior_channel * 2)
                # projection layer
                project_dict[str(key)] = nn.Linear(behavior_channel * 2, n_channels)
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
                 stitcher_type:str = 'spike',
                 behavior_channel:int = 2):
        super().__init__()

        stitch_decoder_dict = {}
        if stitcher_type == 'spike':
            for key, val in eid_list.items():
                stitch_decoder_dict[str(key)] = nn.Linear(n_channels, val)
        elif stitcher_type == 'behavior':
            for key, _ in eid_list.items():
                stitch_decoder_dict[str(key)] = nn.Linear(n_channels, behavior_channel)
        else:
            raise ValueError(f"stitcher_type: {stitcher_type} is not supported.")
        self.stitch_decoder_dict = nn.ModuleDict(stitch_decoder_dict)

    def forward(self, x, block_idx):
        return self.stitch_decoder_dict[block_idx](x)