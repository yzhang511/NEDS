import os
import numpy as np
from dataclasses import dataclass
from einops import repeat
from typing import (
    Any, 
    List, 
    Tuple, 
    Dict, 
    Union,
    Optional, 
)
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN
ACT2FN["softsign"] = nn.Softsign
from utils.config_utils import DictConfig, update_config
from models.masker import Masker
from multi_modal.encoder_embeddings import EncoderLayer
from models.stitcher import StitchDecoder
from models.model_output import ModelOutput
from multi_modal.mm_utils import create_context_mask

DEFAULT_CONFIG = "src/configs/multi_modal/mm.yaml"

STATIC_VARS = ["choice", "block"]
DYNAMIC_VARS = ["wheel", "whisker"]

@dataclass
class MultiModalOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    mod_loss: Optional[torch.FloatTensor] = None
    mod_n_examples: Optional[torch.LongTensor] = None
    mod_preds: Optional[torch.FloatTensor] = None
    mod_targets: Optional[torch.FloatTensor] = None
    static_preds: Optional[torch.LongTensor] = None
    static_targets: Optional[torch.LongTensor] = None
    
class MultiModal(nn.Module):
    def __init__(
        self, 
        encoder_embeddings:        Dict[str, nn.Module],
        avail_mod:                 List,
        avail_beh:                 List,
        model_mode:                List,
        config:                    DictConfig,
        **kwargs
    ):
        super().__init__()

        self.avail_mod = avail_mod
        self.avail_beh = avail_beh
        self.model_mode = model_mode
        self.eid_list = kwargs["eid_list"]
        self.mod_to_indx = {r: i for i, r in enumerate(self.avail_mod)}
        self.n_mod = len(self.avail_mod)

        self.n_layers = config.encoder.transformer.n_layers
        self.hidden_size = config.encoder.transformer.hidden_size
        self.max_F = config.encoder.embedder.max_F

        self.encoder_modalities = set(encoder_embeddings.keys())
        self.encoder_embeddings = nn.ModuleDict(encoder_embeddings)

        # context_mask = create_context_mask(config.context.forward, config.context.backward, self.max_F)
        # self.register_buffer("context_mask", context_mask, persistent=False)

        self.mask = config.masker.force_active
        if self.mask:
            assert config.masker.mode in ["temporal", "causal"], "only allow temporal / causal token masking."
            self.masker = Masker(config.masker)

        self.encoder = nn.ModuleList(
            [EncoderLayer(idx, config.encoder.transformer) for idx in range(self.n_layers)]
        )
        self.encoder_norm = nn.LayerNorm(self.hidden_size) 

        self.num_class = {
            "spike": None, "wheel": 1, "whisker": 1, "choice": 2, "block": 3,
        }
        self.mod_type = {
            "spike": "spike", 
            "choice": "static", 
            "block": "static",
            "wheel": "dynamic", 
            "whisker": "dynamic",
        }

        self.mod_loss = {
            "spike": nn.PoissonNLLLoss(reduction="none", log_input=True),
            "dynamic": nn.MSELoss(reduction="none"),
            "static": nn.CrossEntropyLoss(reduction="none") if "choice" in self.avail_beh else nn.MSELoss(reduction="none")
        }

        if self.model_mode in ["encoding", "decoding"]:
            self.init_unimodal_stitcher()
        

    def init_unimodal_stitcher(self):
        # Trick to handle incompatibility between unimodal and multimodal outputs
        # Can we improve this in the future?
        if self.model_mode == "encoding":
            mod_list = ["spike"]
            _eid_list = {k: v for k, v in self.eid_list.items()}
            n_channels = self.hidden_size * len(self.avail_beh)
        else:
            mod_list, _eid_list, n_channels = self.avail_beh, self.eid_list, self.hidden_size
            
        device = "cuda" if torch.cuda.is_available() else "cpu"

        self.mod_stitcher_proj_dict, self.mod_static_weight_dict = {}, {}
        self.mod_token_weight_dict = {}
        for mod in mod_list:
            self.mod_stitcher_proj_dict[mod] = StitchDecoder(
                eid_list = _eid_list, n_channels = n_channels, mod = mod,
            ).to(device)
            if mod in STATIC_VARS:
                tmp_dict = {}
                for key, val in _eid_list.items():
                    tmp_dict[str(key)] = nn.Parameter(torch.rand(self.max_F))
                self.mod_static_weight_dict[mod] = nn.ParameterDict(tmp_dict).to(device)
                
    
    def cat_encoder_tensors(self, mod_dict: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor]:
        encoder_tokens, encoder_emb, input_timestamp = [], [], []
        encoder_mask, mod_mask = [], []

        for mod, d in mod_dict.items():
            encoder_tokens.append(d["x"])
            encoder_emb.append(d["emb"])
            input_timestamp.append(d["inputs_timestamp"])
            encoder_mask.append(d["inputs_mask"])
            mod_mask.append(torch.full_like(d["inputs_mask"], self.mod_to_indx[mod]))
    
        encoder_tokens = torch.cat(encoder_tokens, dim=1)
        encoder_emb = torch.cat(encoder_emb, dim=1)
        input_timestamp = torch.cat(input_timestamp, dim=1)
        encoder_mask = torch.cat(encoder_mask, dim=1)
        mod_mask = torch.cat(mod_mask, dim=1).to(torch.int16)
        return encoder_tokens, encoder_emb, input_timestamp, encoder_mask, mod_mask

    
    def forward_mask_encoder(self, mod_dict: Dict[str, Dict[str, torch.Tensor]]) -> Tuple[torch.Tensor]:
        
        encoder_tokens, encoder_emb, input_timestamp, encoder_mask, mod_mask = \
        self.cat_encoder_tensors(mod_dict)

        B, N, _ = encoder_tokens.size()

        encoder_mask_ids = torch.argwhere(encoder_mask[0] == 1).squeeze()
        
        encoder_tokens[:,encoder_mask_ids,:] = 0.        

        return encoder_tokens, encoder_emb, input_timestamp, encoder_mask, mod_mask


    def forward_encoder(
        self, 
        x: torch.Tensor, 
        input_timestamp: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        
        B, N, _ = x.size()

        # context_mask = self.context_mask.repeat(N//self.max_F, N//self.max_F).unsqueeze(0).expand(B,N,N)
        
        for layer in self.encoder:
            x = layer(
                x, mask=None, timestamp=input_timestamp
            )

        x = self.encoder_norm(x)

        return x

    
    def forward_loss(self, 
        output_mod_dict: Dict[str, Any]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

        mod_loss, mod_n_examples, mod_preds, mod_targets, static_targets, static_preds = \
        {}, {}, {}, {}, {}, {}

        for mod, d in output_mod_dict.items():
            targets = output_mod_dict[mod]["gt"]
            B, T, N = targets.size()
            targets_mask = output_mod_dict[mod]["targets_mask"].unsqueeze(-1).expand(B, self.max_F, N)
            preds = output_mod_dict[mod]["preds"]

            mod_type = self.mod_type[mod]
            if mod_type != "static":
                pad_mask = targets != -1.
                targets_mask = torch.mul(targets_mask, pad_mask)
                n_examples = targets_mask.sum()
                if n_examples != 0:                        
                    assert preds.shape == targets.shape == targets_mask.shape, \
                    f"shape mismatch in computing loss: preds ({preds.shape}) vs. targets ({targets.shape})."
                    loss = (self.mod_loss[mod_type](preds, targets)*targets_mask).sum()/n_examples
                else:
                    loss = torch.zeros(1, device=targets.device, requires_grad=True).squeeze()
            else:
                preds, targets = preds.squeeze(1), targets.squeeze(1)
                targets_mask = targets_mask.squeeze(1)
                n_examples = targets_mask.sum()
                if n_examples == 0:
                    mod_loss[mod], mod_n_examples[mod], mod_preds[mod], mod_targets[mod] = \
                    torch.zeros(1, device=targets.device, requires_grad=True).squeeze(), \
                    n_examples, preds, targets
                    continue                
                static_targets[mod], static_preds[mod] = targets.squeeze(1), preds.argmax(-1) 
                if mod in ["choice", "block"]:
                    targets = F.one_hot(
                        targets.to(torch.int64), num_classes=self.num_class[mod]
                    ).squeeze(1)          
                else:
                    targets = targets.reshape(-1, 1)
                loss = self.mod_loss[mod_type](preds.float(), targets.float()).sum() / n_examples
                if mod in ["choice", "block"]:
                    preds, targets = preds.argmax(-1), targets.argmax(-1)
            
            mod_loss[mod] = loss
            mod_n_examples[mod] = n_examples
            mod_preds[mod] = preds
            mod_targets[mod] = targets
                
        loss = sum(mod_loss.values())

        return loss, mod_loss, mod_n_examples, mod_preds, mod_targets, static_targets, static_preds


    def forward_unimodal_output(
        self, mod_dict: Dict[str, Dict[str, torch.Tensor]], y
    ) -> MultiModalOutput:
        # Trick to handle incompatibility between unimodal and multimodal outputs
        # Can we improve this in the future?
        output_mod_dict = {}
        eid = mod_dict["spike"]["eid"]

        if self.model_mode == "encoding":
            mod_list = ["spike"]
        else:
            mod_list = self.avail_beh

        B, N, P = y.size()

        for mod in mod_list:

            output_mod_dict[mod] = {}
            if hasattr(self, "mod_stitcher_proj_dict"):
                y_mod = y.clone()
                if hasattr(self, "mod_static_weight_dict") and (mod in STATIC_VARS):
                    weight = torch.zeros_like(y, device=y.device) 
                    eid = np.array(eid)
                    unique_eids = np.unique(eid)
                    for group_eid in unique_eids:
                        mask = torch.tensor(np.argwhere(eid==group_eid), device=y.device).squeeze()
                        if mask.dim() > 0:
                            weight[mask] = self.mod_static_weight_dict[mod][group_eid][None,:,None].expand(mask.size(0),N,P)
                    y_mod = torch.sum(y.reshape(B,N,P) * weight, 1).reshape(B,-1)

                if self.model_mode == "encoding":
                    chunks = []
                    for beh_idx in range(len(self.avail_beh)): 
                        start_idx = beh_idx * self.max_F
                        end_idx = start_idx + self.max_F
                        chunks.append(y_mod[:, start_idx:end_idx])
                    y_mod = torch.cat(chunks, dim=2)
                preds = self.mod_stitcher_proj_dict[mod](y_mod, eid) 
                output_mod_dict[mod]["preds"] = preds.reshape((B,self.max_F,-1)) \
                    if mod not in STATIC_VARS else preds

            output_mod_dict[mod]["targets_mask"] = mod_dict[mod]["targets_mask"]
            output_mod_dict[mod]["gt"] = mod_dict[mod]["targets"]
        
        return output_mod_dict


    def _prepare_mixed_masking(self, mod_dict):
                    
        tmp = mod_dict["spike"]["inputs"]
        
        masking_schemes = [
            "encoding", "decoding", "self-spike", "self-behavior", "random_token"
        ]
        selected_schemes = np.random.choice(
            masking_schemes, size=tmp.size()[0], replace=True
        )
        all_ones = torch.ones_like(tmp).to(tmp.device, torch.int64)
        all_zeros = all_ones * 0.
        
        mask_map = {}
        for mod in self.avail_mod:
            if mod == "spike":
                mask_map[mod] = {
                    "encoding": all_ones,
                    "decoding": all_zeros,
                    "self-spike": self.masker(tmp, None, "temporal")[1],
                    "self-behavior": all_zeros,
                    "random_token": self.masker(tmp, None, "temporal")[1],
                }
            elif mod in DYNAMIC_VARS+STATIC_VARS:
                mask_map[mod] = {
                    "encoding": 1 - mask_map["spike"]["encoding"],
                    "decoding": 1 - mask_map["spike"]["decoding"],
                    "self-spike": all_zeros,
                    "self-behavior": self.masker(tmp, None, "temporal")[1],
                    "random_token": mask_map["spike"]["random_token"],
                }
        return mask_map, selected_schemes

    
    def forward(self, mod_dict: Dict[str, Dict[str, torch.Tensor]]) -> MultiModalOutput:

        if self.model_mode == "mm" and mod_dict["spike"]["training_mode"] == "mixed":
            mask_map, selected_schemes = self._prepare_mixed_masking(mod_dict)

        for mod, d in mod_dict.items():

            for name in ["inputs", "targets"]:
                if len(mod_dict[mod][name].size()) == 2:
                    mod_dict[mod][name] = mod_dict[mod][name].unsqueeze(-1)
                        
            if mod_dict[mod]["eval_mask"] is None:
                _, mask = self.masker(mod_dict[mod]["inputs"].clone(), None)
            else:
                mask = mod_dict[mod]["eval_mask"]

            mask = mask[...,0].to(torch.int64) & mod_dict[mod]["inputs_attn_mask"]

            if mod_dict[mod]["training_mode"] == "mixed":
                mask_list = []
                for scheme in selected_schemes:
                    tmp = mask_map[mod][scheme][0,:,0].to(torch.int64) & mod_dict[mod]["inputs_attn_mask"][0]
                    mask_list.append(tmp.unsqueeze(0))
                mask = torch.cat(mask_list, dim=0)
            
            # Mask selected modalities for encoding
            if "inputs_token_mask" in mod_dict[mod]:
                mod_dict[mod]["inputs_mask"] = mod_dict[mod]["inputs_token_mask"][...,0]
            else:
                mod_dict[mod]["inputs_mask"] = mask
            mod_dict[mod]["targets_mask"] = mask

        encoder_mod_dict = {
            mod: self.encoder_embeddings[mod](d)
            for mod, d in mod_dict.items() if mod in self.encoder_embeddings
        }
        encoder_tokens, encoder_emb, input_timestamp, encoder_mask, encoder_mod_mask = \
        self.forward_mask_encoder(encoder_mod_dict)

        x = encoder_tokens + encoder_emb
        x = self.forward_encoder(x, input_timestamp=input_timestamp)

        if self.model_mode == "mm":
            output_mod_dict = {
                mod: self.encoder_embeddings[mod].out_proj(
                    self.mod_to_indx[mod], d, x, encoder_mod_mask, len(self.avail_mod)
                )
                for mod, d in encoder_mod_dict.items() if mod in self.encoder_embeddings
            }
        else:
            output_mod_dict = self.forward_unimodal_output(mod_dict, x)
            
        loss, mod_loss, mod_n_examples, mod_preds, mod_targets, static_targets, static_preds = \
        self.forward_loss(output_mod_dict)

        return MultiModalOutput(
            loss=loss,
            mod_loss=mod_loss,
            mod_n_examples=mod_n_examples,
            mod_preds=mod_preds,
            mod_targets=mod_targets,
            static_preds=static_preds,
            static_targets=static_targets,
        )

    