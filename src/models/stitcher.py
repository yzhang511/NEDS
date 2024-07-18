import torch
from torch import nn

class StitchEncoder(nn.Module):

    def __init__(self, 
                 eid_list:dict,
                 n_channels:int,):
        super().__init__()

        stitcher_dict = {}
        # iterate key, value pairs in the dictionary
        for key, val in eid_list.items():
            stitcher_dict[str(key)] = nn.Linear(int(val), n_channels)
        self.stitcher_dict = nn.ModuleDict(stitcher_dict)

    def forward(self, x, block_idx):
        return self.stitcher_dict[block_idx](x)
    
class StitchDecoder(nn.Module):

    def __init__(self,
                 eid_list:list,
                 n_channels:int):
        super().__init__()

        stitch_decoder_dict = {}
        for key, val in eid_list.items():
            stitch_decoder_dict[str(key)] = nn.Linear(n_channels, val)
        self.stitch_decoder_dict = nn.ModuleDict(stitch_decoder_dict)

    def forward(self, x, block_idx):
        return self.stitch_decoder_dict[block_idx](x)