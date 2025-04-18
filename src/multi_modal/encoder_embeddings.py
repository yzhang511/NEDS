import os
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN
ACT2FN["softsign"] = nn.Softsign
from utils.config_utils import DictConfig, update_config
from multi_modal.mm_utils import ScaleNorm, MLP, Attention
from models.stitcher import StitchEncoder, StitchDecoder

DEFAULT_CONFIG = "src/configs/multi_modal/mm.yaml"

PROJ_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
with open(f"{PROJ_DIR}/data/train_eids.txt") as file:
    INCLUDE_EIDS = [line.rstrip() for line in file]
with open(f"{PROJ_DIR}/data/test_eids.txt") as file:
    INCLUDE_EIDS += [line.rstrip() for line in file]

STATIC_VARS = ["choice", "block"]


class EncoderEmbeddingLayer(nn.Module):
    def __init__(
        self, hidden_size, n_channels, config: DictConfig, stitching=False, eid_list=None, mod=None, max_F=100
    ):
        super().__init__()

        self.bias = config.bias
        self.n_channels = n_channels
        self.input_dim = self.n_channels*config.mult
        self.max_F = max_F

        self.mod_emb = nn.Embedding(config.n_modality, hidden_size)

        self.eid_lookup = INCLUDE_EIDS
        self.eid_to_indx = {r: i for i, r in enumerate(self.eid_lookup)}
        self.session_emb = nn.Embedding(len(self.eid_lookup), hidden_size)

        self.pos = config.pos
        if self.pos:
            self.pos_embed = nn.Embedding(config.max_F, hidden_size)

        self.dropout = nn.Dropout(config.dropout)

        if stitching:
            self.mod_stitch_encoder = StitchEncoder(
                eid_list=eid_list, n_channels=hidden_size, mod=mod, max_F=max_F
            )
        else:
            self.token_embed = nn.Linear(self.n_channels, self.input_dim, bias=self.bias)
            self.projection = nn.Linear(self.input_dim, hidden_size)
            self.act = ACT2FN[config.act] if config.act != "identity" else nn.Identity()
            self.scale = hidden_size ** 0.5 if config.scale == None else config.scale

    
    def forward(self, d: Dict[str, torch.Tensor]) -> Tuple[torch.FloatTensor, torch.FloatTensor]:  

        inputs, inputs_timestamp, inputs_modality, eid = \
        d["inputs"], d["inputs_timestamp"], d["inputs_modality"], d["eid"]
        B, N, D = inputs.size()
        N = self.max_F
        if hasattr(self, "mod_stitch_encoder"):
            x = self.mod_stitch_encoder(inputs, eid)
        else:
            x = self.token_embed(inputs)
            x = self.act(x) * self.scale
            x = self.projection(x)

        x_embed = self.mod_emb(inputs_modality)[None,None,:].expand(B,N,-1).clone()

        if self.pos:
            x_embed += self.pos_embed(inputs_timestamp)

        eid = np.array(eid)
        unique_eids = np.unique(eid)
        for group_eid in unique_eids:
            mask = torch.tensor(np.argwhere(eid==group_eid), device=x.device).squeeze()
            if mask.dim() > 0:
                session_idx = torch.tensor(self.eid_to_indx[group_eid]).to(x.device, torch.int64)
                x_embed[mask] += self.session_emb(session_idx)[None,None,:].expand(mask.size(0),N,-1)

        return self.dropout(x), x_embed


class EncoderEmbedding(nn.Module):
    def __init__(
        self, 
        n_channel,
        output_channel,
        config: DictConfig,
        stitching=False,
        eid_list=None,
        mod=None,
        max_F=100,
        **kwargs
    ):
        super().__init__() 

        self.hidden_size = config.transformer.hidden_size
        self.n_layers = config.transformer.n_layers
        self.max_F = max_F
        self.n_channel = n_channel
        self.output_channel = output_channel

        self.embedder = EncoderEmbeddingLayer(
            self.hidden_size, self.n_channel, config.embedder, stitching, eid_list, mod, max_F
        )

        if stitching:
            self.mod_stitcher_proj_dict = StitchDecoder(
                eid_list = eid_list, n_channels = self.n_channel, mod = mod, max_F = max_F
            )
            if mod in STATIC_VARS:
                mod_static_weight_dict = {}
                for key, val in eid_list.items():
                    mod_static_weight_dict[str(key)] = nn.Parameter(torch.rand(self.max_F))
                self.mod_static_weight_dict = nn.ParameterDict(mod_static_weight_dict)
        else:
            self.out = nn.Linear(self.hidden_size, self.output_channel)

    def forward(self, d : Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:    
                        
        x, x_emb = self.embedder(d)
        d["x"], d["emb"], d["gt"] = x, x_emb, d["targets"]
        
        return d

    def out_proj(self, 
        mod_idx: int, d: Dict[str, torch.Tensor], y: torch.Tensor, 
        mod_mask: torch.Tensor, n_mod: int,
    ) -> Dict[str, torch.Tensor]: 

        B, N, P = y.size()

        y_mod = y[mod_mask == mod_idx]
        
        if hasattr(self, "mod_stitcher_proj_dict"):
            if hasattr(self, "mod_static_weight_dict"):
                weight = torch.zeros_like(y_mod.reshape(B,-1,P), device=y.device) 
                eid = np.array(d["eid"])
                unique_eids = np.unique(eid)
                for group_eid in unique_eids:
                    mask = torch.tensor(np.argwhere(eid==group_eid), device=y.device).squeeze()
                    if mask.dim() > 0:
                        weight[mask] = self.mod_static_weight_dict[group_eid][None,:,None].expand(mask.size(0),-1,P)
                y_mod = torch.sum(
                    y_mod.reshape(B,-1,P) * weight, 1
                ).reshape(B,-1)
            preds = self.mod_stitcher_proj_dict(y_mod, d["eid"])
            d["preds"] = preds.reshape((B,-1,preds.size()[-1])) \
                if not hasattr(self, "mod_static_weight_dict") else preds
        else:
            y_mod = self.out(y_mod).reshape((B,-1,self.output_channel))
            d["preds"] = y_mod
        
        return d
        


class EncoderLayer(nn.Module):
    
    def __init__(self, idx, config: DictConfig):
        super().__init__()

        self.idx = idx
    
        self.ln1 = ScaleNorm(config.hidden_size ** 0.5) \
            if config.use_scalenorm else nn.LayerNorm(config.hidden_size) 
        self.attn = Attention(
            idx, config.hidden_size, config.n_heads, config.attention_bias, 
            config.dropout, config.use_rope, 
        )
        self.ln2 = ScaleNorm(config.hidden_size ** 0.5) \
            if config.use_scalenorm else nn.LayerNorm(config.hidden_size) 
        self.mlp = MLP(
            config.hidden_size, config.inter_size, config.act, 
            config.mlp_bias, config.dropout
        )
        if config.fixup_init:
            self.fixup_initialization(config.n_layers)

    def forward(
        self, x: torch.FloatTensor, 
        mask: Optional[torch.LongTensor] = None, 
        timestamp: Optional[torch.LongTensor] = None,  
    ) -> torch.FloatTensor :                           
        
        x = x + self.attn(self.ln1(x), mask=mask, timestamp=timestamp)

        x = x + self.mlp(self.ln2(x))

        return x

    def fixup_initialization(self, n_layers):
        temp_state_dic = {}
        for name, param in self.named_parameters():
            if name.endswith("_proj.weight"):
                temp_state_dic[name] = (0.67 * (n_layers) ** (- 1./4.)) * param
            elif name.endswith("value.weight"):
                temp_state_dic[name] = (0.67 * (n_layers) ** (- 1./4.)) * (param * (2**0.5))
                
        for name in self.state_dict():
            if name not in temp_state_dic:
                temp_state_dic[name] = self.state_dict()[name]
        self.load_state_dict(temp_state_dic)   
        