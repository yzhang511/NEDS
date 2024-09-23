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
from multi_modal.mm_utils import ScaleNorm, MLP, FactorsProjection, Attention, CrossAttention
from models.stitcher import StitchDecoder, StitchEncoder

DEFAULT_CONFIG = "src/configs/multi_modal/mm.yaml"

with open('data/train_eids.txt') as file:
    include_eids = [line.rstrip() for line in file]


class DecoderEmbeddingLayer(nn.Module):
    def __init__(
        self, hidden_size, n_channels, config: DictConfig, stitching=False, eid_list=None, mod=None,
    ):
        super().__init__()

        self.bias = config.bias
        self.n_channels = n_channels
        self.input_dim = self.n_channels*config.mult

        self.mod_emb = nn.Embedding(config.n_modality, hidden_size)

        #####
        self.n_static_vars = 2
        
        self.pos = config.pos
        if self.pos:
            self.pos_embed = nn.Embedding(config.max_F+self.n_static_vars, hidden_size)
        #####

        self.eid_lookup = include_eids
        self.eid_to_indx = {r: i for i,r in enumerate(self.eid_lookup)}
        self.session_emb = nn.Embedding(len(self.eid_lookup), hidden_size)

        self.dropout = nn.Dropout(config.dropout)

        if stitching:
            self.spike_stitch_decoder = StitchEncoder(
                eid_list=eid_list, n_channels=hidden_size, mod=mod
            )
        else:
            self.token_embed = nn.Linear(self.n_channels, self.input_dim, bias=self.bias)
            self.projection = nn.Linear(self.input_dim, hidden_size)
            self.act = ACT2FN[config.act] if config.act != "identity" else nn.Identity()
            self.scale = hidden_size ** 0.5 if config.scale == None else config.scale

    def forward(self, d : Dict[str, torch.Tensor]) -> Tuple[torch.FloatTensor, torch.FloatTensor]:  

        targets, targets_timestamp, targets_modality, eid = \
        d['inputs'].clone(), d['inputs_timestamp'].clone(), d['inputs_modality'], d['eid']
        
        B, N, D = targets.size()
        
        if hasattr(self, 'spike_stitch_decoder'):
            x = self.spike_stitch_decoder(targets, eid)
        else:
            x = self.token_embed(targets)
            x = self.act(x) * self.scale
            x = self.projection(x)

        x_embed = self.mod_emb(targets_modality)[None,None,:].expand(B,N,-1).clone()

        #####
        if self.pos:
            if N == self.n_static_vars:
                targets_timestamp = torch.arange(self.n_static_vars).unsqueeze(0).expand(B,-1).to(x.device)
            else:
                targets_timestamp += self.n_static_vars
            x_embed += self.pos_embed(targets_timestamp)
        #####

        session_idx = torch.tensor(self.eid_to_indx[eid], dtype=torch.int64, device=targets.device)
        x_embed += self.session_emb(session_idx)[None,None,:].expand(B,N,-1).clone()

        return self.dropout(x), x_embed



class DecoderEmbedding(nn.Module):
    def __init__(
        self, 
        n_channel,
        output_channel,
        config: DictConfig,
        stitching=False,
        eid_list=None,
        mod=None,
        **kwargs
    ):
        super().__init__() 
        
        self.hidden_size = config.transformer.hidden_size
        self.n_layers = config.transformer.n_layers
        self.max_F = config.embedder.max_F
        self.n_channel = n_channel
        self.output_channel = output_channel

        self.embedder = DecoderEmbeddingLayer(
            self.hidden_size, self.n_channel, config.embedder, stitching, eid_list, mod
        )

        if stitching:
            #####
            if mod == 'static':
                self.choice_stitch_decoder = StitchDecoder(eid_list, self.n_channel, mod='choice')
                self.block_stitch_decoder = StitchDecoder(eid_list, self.n_channel, mod='block')
            else:
                self.spike_stitch_proj_decoder = StitchDecoder(eid_list, self.n_channel, mod=mod)
            #####
        else:
            self.out = nn.Linear(self.hidden_size, self.output_channel)
    
    def forward_embed(self, d : Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:    
                        
        x, x_emb = self.embedder(d)

        d['x'] = x
        d['emb'] = x_emb
        d['gt'] = d['targets']

        return d

    def out_proj(self, 
        mod_idx : int,
        d : Dict[str, torch.Tensor],
        y : torch.Tensor, 
        decoder_mod_mask : torch.Tensor,
        n_mod : int,
    ) -> Dict[str, torch.Tensor]:   
        
        B, N, P = y.size()
        
        y_mod, eid = y[decoder_mod_mask == mod_idx], d['eid']

        #####
        # if len(torch.unique(decoder_mod_mask)) > 1:
        #     all_mod_idxs = torch.arange(n_mod)
        #     for _mod_idx in all_mod_idxs[all_mod_idxs != mod_idx]:
        #         y_mod = torch.cat((y_mod, y[decoder_mod_mask==_mod_idx][:,:P//2]), 1)
    
        if hasattr(self, 'spike_stitch_proj_decoder'):
            preds = self.spike_stitch_proj_decoder(y_mod, eid)
            _, N = preds.size()
            d['preds'] = preds.reshape((B, -1, N))
        elif hasattr(self, 'choice_stitch_decoder'):
            preds_choice = self.choice_stitch_decoder(y_mod.reshape(B, -1, P)[:,0], eid)
            preds_block = self.block_stitch_decoder(y_mod.reshape(B, -1, P)[:,1], eid)
            d['preds'] = (preds_choice, preds_block)
        else:
            y_mod = self.out(y_mod).reshape((B, -1, self.output_channel))
            d['preds'] = y_mod
        #####
            
        return d


class DecoderLayer(nn.Module):
    def __init__(self, idx, config: DictConfig):
        super().__init__()

        self.idx = idx
    
        self.ln1 = ScaleNorm(config.hidden_size ** 0.5) if config.use_scalenorm else nn.LayerNorm(config.hidden_size) 
        
        self.attn = Attention(idx, config.hidden_size, config.n_heads, config.attention_bias, config.dropout)
        self.cross_attn = CrossAttention(idx, config.hidden_size, config.n_heads, config.attention_bias, config.dropout)
        
        self.query_norm = ScaleNorm(config.hidden_size ** 0.5) if config.use_scalenorm else nn.LayerNorm(config.hidden_size) 
        self.context_norm = ScaleNorm(config.hidden_size ** 0.5) if config.use_scalenorm else nn.LayerNorm(config.hidden_size) 
        
        self.ln2 = ScaleNorm(config.hidden_size ** 0.5) if config.use_scalenorm else nn.LayerNorm(config.hidden_size) 
        
        self.mlp = MLP(config.hidden_size, config.inter_size, config.act, config.mlp_bias, config.dropout)

        if config.fixup_init:
            self.fixup_initialization(config.n_layers)

    def forward(
        self, 
        x:        torch.FloatTensor, 
        context:  torch.FloatTensor, 
        sa_mask:  Optional[torch.LongTensor] = None,
        xa_mask:  Optional[torch.LongTensor] = None,
    ) -> torch.FloatTensor :                           
        
        x = x + self.attn(self.ln1(x), sa_mask)

        x = x + self.cross_attn(self.query_norm(x), self.context_norm(context), xa_mask)

        x = x + self.mlp(self.ln2(x))

        return x

    def fixup_initialization(self, n_layers):
        temp_state_dic = {}
        for name, param in self.named_parameters():
            if name.endswith("_proj.weight"):
                temp_state_dic[name] = (0.67 * (n_layers) ** (- 1. / 4.)) * param
            elif name.endswith("value.weight"):
                temp_state_dic[name] = (0.67 * (n_layers) ** (- 1. / 4.)) * (param * (2**0.5))
                
        for name in self.state_dict():
            if name not in temp_state_dic:
                temp_state_dic[name] = self.state_dict()[name]
        self.load_state_dict(temp_state_dic)   
        
