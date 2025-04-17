import os
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

from einops import rearrange

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.activations import ACT2FN
ACT2FN["softsign"] = nn.Softsign

from utils.config_utils import DictConfig, update_config

# Copied from hf Llama
# Precompute cos and sin for RoPE
def get_cos_sin(dim, max_F, base=10000, dtype=torch.get_default_dtype(), device=None):

    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
    t = torch.arange(max_F, device=device, dtype=inv_freq.dtype)
    freqs = torch.einsum("i,j->ij", t, inv_freq)

    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)

# Rotates half the hidden dims of the input.
def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), -1)

# Applies RoPE to the query and key tensors.
def apply_rotary_pos_emb(q, k, pos_ids, cos, sin, unsqueeze_dim=1):

    cos = cos[pos_ids].unsqueeze(unsqueeze_dim)
    sin = sin[pos_ids].unsqueeze(unsqueeze_dim)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    
    return q_embed, k_embed
    

def create_context_mask(context_forward, context_backward, max_F) -> torch.LongTensor: 
    
    if context_forward == -1 and context_backward == -1:
        return torch.ones(max_F, max_F).to(torch.int64)

    context_forward = context_forward if context_forward >= 0 else max_F
    context_backward = context_backward if context_backward >= 0 else max_F
    mask = (torch.triu(torch.ones(max_F, max_F), diagonal=-context_forward).to(torch.int64)).transpose(0, 1)
    if context_backward > 0:
        back_mask = (torch.triu(torch.ones(max_F, max_F), diagonal=-context_backward).to(torch.int64))
        mask = mask & back_mask
    return mask


class ScaleNorm(nn.Module):
    def __init__(self, scale, eps=1e-5):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(scale))
        self.eps = eps

    def forward(self, x):
        norm = self.scale / torch.norm(x, dim=-1, keepdim=True).clamp(min=self.eps)
        return x * norm
        

class MLP(nn.Module):
    def __init__(self, hidden_size, inter_size, act, use_bias, dropout):
        super().__init__()
        self.up_proj    = nn.Linear(hidden_size, inter_size, bias=use_bias)
        self.act        = ACT2FN[act]
        self.down_proj  = nn.Linear(inter_size, hidden_size, bias=use_bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.act(self.up_proj(x))
        return self.dropout(self.down_proj(x))


class FactorsProjection(nn.Module):
    def __init__(self, hidden_size, config):
        super().__init__()
        self.out_size = config.size if config.active else hidden_size        
        self.dropout = nn.Dropout(config.dropout)
        if config.active:
            self.proj = nn.Sequential(
                nn.Linear(hidden_size, config.size, config.bias),
                ACT2FN[config.act]
            )
            if config.fixup_init:
                self.proj[0].weight.data.uniform_(-config.init_range, config.init_range)
                if config.bias:
                    self.proj[0].bias.data.zero_()
        else:
            self.proj = nn.Identity()

    def forward(self, x):
        return self.proj(self.dropout(x))


class Attention(nn.Module):
    def __init__(
        self, idx, hidden_size, n_heads, use_bias, dropout, 
        use_rope=False, base=10000., max_F=100., n_mod=2,
    ):
        super().__init__()
        
        self.idx = idx

        self.hidden_size = hidden_size
        self.n_heads = n_heads
        assert self.hidden_size % self.n_heads == 0, "Hidden dim is not multiple of head size"
        self.head_size = self.hidden_size // self.n_heads

        # Attention parameters
        self.query = nn.Linear(self.hidden_size, self.hidden_size, bias=use_bias)
        self.key = nn.Linear(self.hidden_size, self.hidden_size, bias=use_bias)
        self.value  = nn.Linear(self.hidden_size, self.hidden_size, bias=use_bias)

        torch.backends.cuda.enable_flash_sdp(True)
        self.attn_dropout = dropout

        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=use_bias)

        # RoPE parameters
        self.use_rope = use_rope
        if self.use_rope:
            cos, sin = get_cos_sin(
                self.head_size, max_F*n_mod, base=base, 
                dtype=self.query.weight.dtype, device=self.query.weight.device
            )
            self.register_buffer("cos", cos, persistent=False)
            self.register_buffer("sin", sin, persistent=False)

    def forward(
        self,       
        x:          torch.FloatTensor,                      
        mask:       Optional[torch.LongTensor] = None,    
        timestamp:  Optional[torch.LongTensor] = None,  # (bs, seq_len)
    ) -> torch.FloatTensor:                                

        B, T, _  = x.size()    

        if mask is not None:
            mask = mask.unsqueeze(1).expand(B,self.n_heads,T,T).bool()

        # Compute query, key, value for attention
        q = self.query(x).view(B, T, self.n_heads, self.head_size).transpose(1, 2)      
        k = self.key(x).view(B, T, self.n_heads, self.head_size).transpose(1, 2)        
        v = self.value(x).view(B, T, self.n_heads, self.head_size).transpose(1, 2)     

        # Apply rotations to encode relative positions
        if self.use_rope:
            q, k = apply_rotary_pos_emb(q, k, timestamp, self.cos, self.sin, 1)  # (B,n_heads,T,head_size)

        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=(self.attn_dropout if self.training else 0.0), is_causal=False
        )
        out = out.transpose(1, 2).contiguous().view(B, T, self.hidden_size) 

        return self.out_proj(self.dropout(out)) 


class CrossAttention(nn.Module):
    def __init__(
        self, idx, hidden_size, n_heads, use_bias, dropout, 
        use_rope=False, base=10000., max_F=100., n_mod=2,
    ):
        super().__init__()

        self.idx = idx

        self.hidden_size = hidden_size
        self.n_heads = n_heads
        assert self.hidden_size % self.n_heads == 0, "Hidden dim is not multiple of head size"
        self.head_size = self.hidden_size // self.n_heads

        # Compute query, key, value for cross-attention
        self.query = nn.Linear(self.hidden_size, self.hidden_size, bias=use_bias)
        self.key = nn.Linear(self.hidden_size, self.hidden_size, bias=use_bias)
        self.value  = nn.Linear(self.hidden_size, self.hidden_size, bias=use_bias)

        torch.backends.cuda.enable_flash_sdp(True)
        self.attn_dropout = dropout

        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=use_bias)

        # RoPE parameters
        self.use_rope = use_rope
        if self.use_rope:
            cos, sin = get_cos_sin(
                self.head_size, max_F*n_mod, base=base, 
                dtype=self.query.weight.dtype, device=self.query.weight.device
            )
            self.register_buffer("cos", cos, persistent=False)
            self.register_buffer("sin", sin, persistent=False)

    def forward(
        self, x, context, mask=None, timestamp=None,
    ):
        B, T, _ = x.size()
        _, M, _ = context.size()

        if mask is not None:
            mask = mask.unsqueeze(1).expand(B,self.n_heads,T,M).bool()
        
        q = self.query(x).view(B, T, self.n_heads, self.head_size).transpose(1, 2)      
        k = self.key(context).view(B, M, self.n_heads, self.head_size).transpose(1, 2)        
        v = self.value(context).view(B, M, self.n_heads, self.head_size).transpose(1, 2)   

        # Apply rotations to encode relative positions
        if self.use_rope:
            q, k = apply_rotary_pos_emb(q, k, timestamp, self.cos, self.sin, 1)  # (B,n_heads,T,head_size)

        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=(self.attn_dropout if self.training else 0.0), is_causal=False
        )
        out = out.transpose(1, 2).contiguous().view(B, T, self.hidden_size) 

        return self.out_proj(self.dropout(out)) 

    
        