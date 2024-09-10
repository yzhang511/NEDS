import os
import numpy as np
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Dict, Union

from einops import rearrange, repeat

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.activations import ACT2FN
ACT2FN["softsign"] = nn.Softsign

from utils.config_utils import DictConfig, update_config
from utils.metric_utils import clip_contrastive_loss, top_k_accuracy
from models.model_output import ModelOutput
from multi_modal.encoder_embeddings import EncoderLayer
from multi_modal.decoder_embeddings import DecoderLayer
from models.masker import Masker
from multi_modal.mm_utils import create_context_mask

DEFAULT_CONFIG = "src/configs/multi_modal/mm.yaml"

@dataclass
class MultiModalOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    mod_loss: Optional[torch.FloatTensor] = None
    mod_n_examples: Optional[torch.LongTensor] = None
    mod_preds: Optional[torch.FloatTensor] = None
    mod_targets: Optional[torch.FloatTensor] = None
    contrastive_dict: Optional[Dict[str, torch.Tensor]] = None


class MultiModal(nn.Module):
    def __init__(
        self, 
        encoder_embeddings: Dict[str, nn.Module],
        decoder_embeddings: Dict[str, nn.Module],
        avail_mod:          List,
        config: DictConfig,
        share_modality_embeddings: bool = True,
        **kwargs
    ):
        super().__init__()

        self.avail_mod = avail_mod
        self.mod_to_indx = {r: i for i,r in enumerate(self.avail_mod)}
        self.decoder_sep_mask = config.decoder.decoder_sep_mask
        self.decoder_causal_mask = config.decoder.decoder_causal_mask

        self.n_enc_layers = config.encoder.transformer.n_layers
        self.n_dec_layers = config.decoder.transformer.n_layers
        self.hidden_size = config.encoder.transformer.hidden_size
        self.max_F = config.encoder.embedder.max_F
        self.context_forward = config.context.forward
        self.context_backward = config.context.backward

        self.encoder_modalities = set(encoder_embeddings.keys())
        self.encoder_embeddings = nn.ModuleDict(encoder_embeddings)

        self.decoder_modalities = set(decoder_embeddings.keys())
        self.decoder_embeddings = nn.ModuleDict(decoder_embeddings)

        if share_modality_embeddings:
            self.share_modality_embeddings()

        self.mask = config.masker.force_active
        if self.mask:
            assert config.masker.mode in ['temporal'], "Only token-wise masking is allowed for multi-modal model for now."
            self.masker = Masker(config.masker)

        self.encoder = nn.ModuleList([EncoderLayer(idx, config.encoder.transformer) for idx in range(self.n_enc_layers)])
        self.encoder_norm = nn.LayerNorm(self.hidden_size) 

        self.decoder_proj_context = nn.Linear(self.hidden_size, self.hidden_size)
        
        self.decoder = nn.ModuleList([DecoderLayer(idx, config.decoder.transformer) for idx in range(self.n_dec_layers)])
        self.decoder_norm = nn.LayerNorm(self.hidden_size) 

        self.use_contrastive, self.use_prompt, self.use_moco = config.use_contrastive, config.use_prompt, config.use_moco
        if self.use_contrastive:
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
            self.spike_projection = nn.Linear(100 * self.hidden_size, 768)
            self.behavior_projection = nn.Linear(100 * self.hidden_size, 768)
        if self.use_prompt:
            self.spike_decoder_prompt = nn.Linear(self.hidden_size, self.hidden_size)
            self.behavior_decoder_prompt = nn.Linear(self.hidden_size, self.hidden_size)
            self.spike_encoder_prompt = nn.Linear(self.hidden_size, self.hidden_size)
            self.behavior_encoder_prompt = nn.Linear(self.hidden_size, self.hidden_size)
        if self.use_contrastive and self.use_moco:
            # create momentum encoder
            self.encoder_embeddings_m = nn.ModuleDict(encoder_embeddings)
            self.encoder_m = nn.ModuleList([EncoderLayer(idx, config.encoder.transformer) for idx in range(self.n_enc_layers)])
            self.encoder_norm_m = nn.LayerNorm(self.hidden_size)
            self.model_pairs = [[self.encoder_embeddings, self.encoder_embeddings_m], [self.encoder, self.encoder_m], [self.encoder_norm, self.encoder_norm_m]]
            self.spike_projection_m = nn.Linear(100 * self.hidden_size, 768)
            self.behavior_projection_m = nn.Linear(100 * self.hidden_size, 768)
            if self.use_prompt:
                self.spike_encoder_prompt_m = nn.Linear(self.hidden_size, self.hidden_size)
                self.behavior_encoder_prompt_m = nn.Linear(self.hidden_size, self.hidden_size)
                self.model_pairs.append([self.spike_encoder_prompt, self.spike_encoder_prompt_m])
                self.model_pairs.append([self.behavior_encoder_prompt, self.behavior_encoder_prompt_m])
            # create the queue
            self.queue_size = kwargs['queue_size']
            self.register_buffer("spike_queue", torch.randn(768, self.queue_size))
            self.register_buffer("behavior_queue", torch.randn(768, self.queue_size))
            self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))  
                                
            self.spike_queue = nn.functional.normalize(self.spike_queue, dim=0)
            self.behavior_queue = nn.functional.normalize(self.behavior_queue, dim=0)
            self.copy_params()
            self.momentum = kwargs['momentum']
        self.loss_mod = {
            'ap': nn.PoissonNLLLoss(reduction="none", log_input=True),
            'behavior': nn.MSELoss(reduction="none"),
        }

    @torch.no_grad()    
    def copy_params(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient    
            
    @torch.no_grad()        
    def _momentum_update(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)
        
    def share_modality_embeddings(self):
        shared_modalities = self.encoder_modalities & self.decoder_modalities
        for mod in shared_modalities:
            self.decoder_embeddings[mod].embedder.mod_emb = self.encoder_embeddings[mod].embedder.mod_emb

    
    def cat_encoder_tensors(self, mod_dict: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor]:
        encoder_tokens = []
        encoder_emb = []
        encoder_mask = []
        attention_mask = []
        mod_mask = []

        for mod, d in mod_dict.items():
            encoder_tokens.append(d['x'])
            encoder_emb.append(d['emb'])
            encoder_mask.append(d['inputs_mask'])
            attention_mask.append(d['encoder_attn_mask'])
            mod_mask.append(torch.full_like(d['inputs_mask'], self.mod_to_indx[mod], dtype=torch.int16))

    
        encoder_tokens = torch.cat(encoder_tokens, dim=1)
        encoder_emb = torch.cat(encoder_emb, dim=1)
        encoder_mask = torch.cat(encoder_mask, dim=1)
        attention_mask = torch.cat(attention_mask, dim=1)
        mod_mask = torch.cat(mod_mask, dim=1)

        return encoder_tokens, encoder_emb, encoder_mask, attention_mask, mod_mask

    
    def cat_decoder_tensors(self, mod_dict: Dict[str, Dict[str, torch.Tensor]]) -> Tuple[torch.Tensor]:
        decoder_tokens = []
        target_gts = {}
        decoder_emb = []
        decoder_mask = []
        attention_mask = []
        mod_mask = []

        # shuffle order in which modalities are provided (useful for modality causal mask)
        # mod_dict = {mod: d for mod, d in random.sample(mod_dict.items(), len(mod_dict))}

        for mod, d in mod_dict.items():
            decoder_tokens.append(d['x'])
            target_gts[mod] = d['gt']
            decoder_emb.append(d['emb'])
            decoder_mask.append(d['targets_mask'])
            attention_mask.append(d['decoder_attn_mask'])
            mod_mask.append(torch.full_like(d['targets_mask'], self.mod_to_indx[mod], dtype=torch.int16))
        
        decoder_tokens = torch.cat(decoder_tokens, dim=1)
        decoder_emb = torch.cat(decoder_emb, dim=1)
        decoder_mask = torch.cat(decoder_mask, dim=1)
        attention_mask = torch.cat(attention_mask, dim=1)
        mod_mask = torch.cat(mod_mask, dim=1)

        return decoder_tokens, target_gts, decoder_emb, decoder_mask, attention_mask, mod_mask

    
    def forward_mask_encoder(self, mod_dict: Dict[str, Dict[str, torch.Tensor]]) -> Tuple[torch.Tensor]:
        
        encoder_tokens, encoder_emb, encoder_mask, encoder_attn_mask, mod_mask = self.cat_encoder_tensors(mod_dict)

        B, N, _ = encoder_tokens.size()

        encoder_mask_ids = torch.argwhere(encoder_mask[0] == 1).squeeze()
        
        encoder_tokens[:,encoder_mask_ids,:] = 0.
        # encoder_emb[:,encoder_mask_ids,:] = 0.

        encoder_attn_mask = encoder_attn_mask.unsqueeze(1).expand(B,N,N)
        self_mask = torch.eye(N).to(encoder_attn_mask.device, torch.int64).expand(B,N,N)
        ###
        context_mask = torch.ones_like(encoder_attn_mask).to(encoder_attn_mask.device, torch.int64)
        # context_mask = create_context_mask(0, -1, N).to(encoder_tokens.device)
        # context_mask = repeat(context_mask, "n1 n2 -> b n1 n2", b=B)
        ###
        encoder_attn_mask = self_mask | (context_mask & encoder_attn_mask)
        return encoder_tokens, encoder_emb, encoder_mask, encoder_attn_mask, mod_mask

    
    def forward_mask_decoder(self, mod_dict: Dict[str, Dict[str, torch.Tensor]]) -> Tuple[torch.Tensor]:
        
        decoder_tokens, target_gts, decoder_emb, decoder_mask, decoder_attn_mask, mod_mask = self.cat_decoder_tensors(mod_dict)

        B, N, _ = decoder_tokens.size()

        decoder_mask_ids = torch.argwhere(decoder_mask[0] == 1).squeeze()

        decoder_tokens[:,decoder_mask_ids,:] = 0.
        # decoder_emb[:,decoder_mask_ids,:] = 0.
        decoder_attn_mask = self.adapt_decoder_attention_mask(decoder_attn_mask, mod_mask)

        return decoder_tokens, target_gts, decoder_emb, decoder_mask, decoder_attn_mask, mod_mask
        
    # TO DO
    def adapt_decoder_attention_mask(self, decoder_attn_mask: torch.Tensor, mod_mask=Optional[torch.Tensor]) -> torch.Tensor:

        B, N = decoder_attn_mask.shape

        if self.decoder_causal_mask:
            causal_mask = create_context_mask(0, -1, N).to(decoder_attn_mask.device)
            causal_mask = repeat(causal_mask, "n1 n2 -> b n1 n2", b=B)
            adapted_attn_mask = causal_mask
        else:
            adapted_attn_mask = decoder_attn_mask.unsqueeze(1).expand(B,N,N)
            
        if self.decoder_sep_mask:
            # separate attention between tokens based on their modality using mod_mask.
            sep_mask = repeat(mod_mask, "b n2 -> b n1 n2", n1=N) != repeat(mod_mask, "b n1 -> b n1 n2", n2=N)
            adapted_attn_mask = adapted_attn_mask | sep_mask

        return adapted_attn_mask

    
    def forward_encoder(self, x: torch.Tensor, encoder_attn_mask: torch.Tensor) -> torch.Tensor:
        
        for layer in self.encoder:
            x = layer(x, mask=encoder_attn_mask)

        x = self.encoder_norm(x)

        return x

    
    def forward_decoder(self, y: torch.Tensor, context: torch.Tensor, encoder_attn_mask: torch.Tensor, decoder_attn_mask: torch.Tensor) -> torch.Tensor:

        for layer in self.decoder:
            y = layer(y, context, sa_mask=decoder_attn_mask, xa_mask=encoder_attn_mask)

        y = self.decoder_norm(y)

        return y
    

    def forward_loss(self, 
        decoder_mod_dict: Dict[str, Any], target_gts: torch.Tensor, contrastive_loss_dict=None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

        mod_loss, mod_n_examples, mod_preds, mod_targets = {}, {}, {}, {}
        for mod, d in decoder_mod_dict.items():
            targets = target_gts[mod]
            B, T, N = targets.size()
            preds = decoder_mod_dict[mod]['preds']
            if "spike_mask" in decoder_mod_dict[mod]:
                targets_mask = decoder_mod_dict[mod]['spike_mask']
            else:
                targets_mask = decoder_mod_dict[mod]['targets_mask'].unsqueeze(-1).expand(B,T,N)
            loss = (self.loss_mod[mod](preds, targets) * targets_mask).sum()
            n_examples = targets_mask.sum()
            mod_loss[mod] = loss
            mod_n_examples[mod] = n_examples
            mod_preds[mod] = preds
            mod_targets[mod] = targets

        loss = sum(mod_loss.values()) / sum(mod_n_examples.values())
        if contrastive_loss_dict is not None:
            loss += contrastive_loss_dict["loss"]
            loss /= 2
            contrastive_dict = {k: v for k, v in contrastive_loss_dict.items() if k != "loss"}
        else:
            contrastive_dict = None

        return loss, mod_loss, mod_n_examples, mod_preds, mod_targets, contrastive_dict
    
    @torch.no_grad()
    def forward_moco_logits(self, x: torch.Tensor, x_m: torch.Tensor, decoder_mod_mask: torch.Tensor, alpha=0.4) -> torch.Tensor:
        B, _, _ = x.shape
        assert 'ap' in self.mod_to_indx and 'behavior' in self.mod_to_indx, "AP and behavior modalities must be present in the model."
        spike_features_m = x_m[decoder_mod_mask == self.mod_to_indx['ap']]
        spike_features_m = spike_features_m.reshape(B, -1)
        spike_features_m = self.spike_projection(spike_features_m)
        spike_features_m = spike_features_m / spike_features_m.norm(dim=1, keepdim=True)
        spike_features_all = torch.cat([spike_features_m.t(), self.spike_queue.clone().detach()], dim=1)
        
        behavior_features_m = x_m[decoder_mod_mask == self.mod_to_indx['behavior']]
        behavior_features_m = behavior_features_m.reshape(B, -1)
        behavior_features_m = self.behavior_projection(behavior_features_m)
        behavior_features_m = behavior_features_m / behavior_features_m.norm(dim=1, keepdim=True)
        behavior_features_all = torch.cat([behavior_features_m.t(), self.behavior_queue.clone().detach()], dim=1) 
        # compute logits
        logit_scale = self.logit_scale.exp()
        sim_s2b_m = spike_features_m @ behavior_features_all / logit_scale
        sim_b2s_m = behavior_features_m @ spike_features_all / logit_scale

        sim_targets = torch.zeros(sim_s2b_m.size()).to(sim_s2b_m.device)
        sim_targets.fill_diagonal_(1)

        # sim_s2b_targets = alpha * sim_targets + (1 - alpha) * (torch.eye(sim_s2b_m.size(0)).to(sim_s2b_m.device) - 1)
        # update queue
        self._dequeue_and_enqueue(spike_features_m, behavior_features_m)
        return {
            "sim_s2b_m": sim_s2b_m,
            "sim_b2s_m": sim_b2s_m,
            "spike_feat_all": spike_features_all,
            "behavior_feat_all": behavior_features_all,
            "sim_targets": sim_targets
        }

    def forward_logits(self, x: torch.Tensor, decoder_mod_mask: torch.Tensor, use_moco=False) -> torch.Tensor:
        B, _, _ = x.shape
        assert 'ap' in self.mod_to_indx and 'behavior' in self.mod_to_indx, "AP and behavior modalities must be present in the model."
        spike_features = x[decoder_mod_mask == self.mod_to_indx['ap']]
        spike_features = spike_features.reshape(B, -1)
        spike_features = self.spike_projection(spike_features)

        behavior_features = x[decoder_mod_mask == self.mod_to_indx['behavior']]
        behavior_features = behavior_features.reshape(B, -1)
        behavior_features = self.behavior_projection(behavior_features)
        
        # normalize features
        spike_features = spike_features / spike_features.norm(dim=1, keepdim=True)
        behavior_features = behavior_features / behavior_features.norm(dim=1, keepdim=True)
        if use_moco:
            return spike_features, behavior_features
        # compute logits
        logit_scale = self.logit_scale.exp()
        logits_per_spike = logit_scale * spike_features @ behavior_features.T
        logits_per_behavior = logit_scale * behavior_features @ spike_features.T
        
        logits = torch.stack([logits_per_spike, logits_per_behavior], dim=0)
        return logits

    def forward_contrastive_loss(self, logits):
        spike_logits = logits[0]
        behavior_logits = logits[1]

        loss_spike, s2b_acc = clip_contrastive_loss(spike_logits)
        loss_behavior, b2s_acc = clip_contrastive_loss(behavior_logits)

        return {
            "loss_spike": loss_spike.item(),
            "loss_behavior": loss_behavior.item(),
            "loss": (loss_spike + loss_behavior) / 2,
            "s2b_acc": s2b_acc.item(),
            "b2s_acc": b2s_acc.item(),
        }

    def forward_moco_loss(self, sim_s2b, sim_b2s, sim_targets):
        loss_s2b = -torch.sum(F.log_softmax(sim_s2b, dim=1) * sim_targets, dim=1).mean()
        loss_b2s = -torch.sum(F.log_softmax(sim_b2s, dim=1) * sim_targets, dim=1).mean()
        loss = (loss_s2b + loss_b2s) / 2

        s2b_acc = top_k_accuracy(sim_s2b, sim_targets, k=1)
        b2s_acc = top_k_accuracy(sim_b2s, sim_targets, k=1)
        return {
            "loss_spike": loss_s2b.item(),
            "loss_behavior": loss_b2s.item(),
            "loss": loss,
            "s2b_acc": s2b_acc,
            "b2s_acc": b2s_acc,
        }

    def forward_decoder_prompt(self, context):
        # assert mod has 2
        assert len(self.avail_mod) == 2, "Only two modalities are supported for contrastive loss."
        B, N, D = context.size()
        spike_context = context[:, :N//2, :]
        behavior_context = context[:, N//2:, :]
        spike_adapt = self.spike_decoder_prompt(spike_context)
        behavior_adapt = self.behavior_decoder_prompt(behavior_context)
        # concatenate adapted features and swap modalities
        adapt = torch.cat([behavior_adapt, spike_adapt], dim=1)
        return adapt

    def forward_encoder_prompt(self, context):
        # assert mod has 2
        assert len(self.avail_mod) == 2, "Only two modalities are supported for contrastive loss."
        B, N, D = context.size()
        spike_context = context[:, :N//2, :]
        behavior_context = context[:, N//2:, :]
        spike_adapt = self.spike_encoder_prompt(spike_context)
        behavior_adapt = self.behavior_encoder_prompt(behavior_context)
        # concatenate adapted features and swap modalities
        adapt = torch.cat([behavior_adapt, spike_adapt], dim=1)
        return adapt
    
    @torch.no_grad()
    def forward_encoder_prompt_m(self, context):
        # assert mod has 2
        assert len(self.avail_mod) == 2, "Only two modalities are supported for contrastive loss."
        B, N, D = context.size()
        spike_context = context[:, :N//2, :]
        behavior_context = context[:, N//2:, :]
        spike_adapt = self.spike_encoder_prompt_m(spike_context)
        behavior_adapt = self.behavior_encoder_prompt_m(behavior_context)
        # concatenate adapted features and swap modalities
        adapt = torch.cat([behavior_adapt, spike_adapt], dim=1)
        return adapt

    @torch.no_grad()
    def forward_encoder_m(self, x: torch.Tensor, encoder_attn_mask: torch.Tensor) -> torch.Tensor:
        for layer in self.encoder_m:
            x = layer(x, mask=encoder_attn_mask)

        x = self.encoder_norm_m(x)

        return x
    
    @torch.no_grad()
    def _dequeue_and_enqueue(self, spike_feat, behavior_feat):
        # gather keys before updating queue
        spike_feats = spike_feat.detach()
        behavior_feats = behavior_feat.detach()

        batch_size = spike_feats.shape[0]

        ptr = int(self.queue_ptr)        
        assert self.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.spike_queue[:, ptr:ptr + batch_size] = spike_feats.T
        self.behavior_queue[:, ptr:ptr + batch_size] = behavior_feats.T
        ptr = (ptr + batch_size) % self.queue_size  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def forward_moco(self, mod_dict: Dict[str, Dict[str, torch.Tensor]]) -> torch.Tensor:
        self._momentum_update()
        # copy the mod_dict to avoid modifying the original one
        mod_dict_m = {mod: d.copy() for mod, d in mod_dict.items()}
        encoder_mod_dict_m = {mod: self.encoder_embeddings_m[mod](d)
                            for mod, d in mod_dict_m.items()
                            if mod in self.encoder_embeddings}

        encoder_tokens, encoder_emb, encoder_mask, encoder_attn_mask, encoder_mod_mask = self.forward_mask_encoder(encoder_mod_dict_m)

        encoder_tokens = encoder_tokens + self.forward_encoder_prompt_m(encoder_tokens) if self.use_prompt else encoder_tokens
        x = encoder_tokens + encoder_emb
        x = self.forward_encoder_m(x, encoder_attn_mask=encoder_attn_mask)

        return x

    def forward(
            self, mod_dict: Dict[str, Dict[str, torch.Tensor]]
        ) -> MultiModalOutput:
        for mod, d in mod_dict.items():

            # TO DO
            if mod == 'behavior' and len(mod_dict[mod]['inputs'].size()) == 2:
                mod_dict[mod]['inputs'] = mod_dict[mod]['inputs'].unsqueeze(-1)
                mod_dict[mod]['targets'] = mod_dict[mod]['targets'].unsqueeze(-1)
                
            B, N, D = mod_dict[mod]['inputs'].size()
            
            inputs_regions = mod_dict[mod]['inputs_regions'] if mod == 'ap' else None
            
            if mod_dict[mod]['masking_mode']:
                self.masker.mode = mod_dict[mod]['masking_mode']
                # print(f"masking mode: {self.masker.mode}")
                # print(f"unmask inputs: {mod_dict[mod]['inputs'].sum()}")
                mod_dict[mod]['inputs'], spike_mask = self.masker(mod_dict[mod]['inputs'].clone(), inputs_regions)
                # print(f"mask inputs: {mod_dict[mod]['inputs'].sum()}")
                # print(f"spike mask: {spike_mask.sum()}")
                mod_dict[mod]['spike_mask'] = spike_mask
            else:
                # print(f"masking mode: {self.masker.mode}")
                if mod_dict[mod]['eval_mask'] is None:
                    _, mask = self.masker(mod_dict[mod]['inputs'].clone(), inputs_regions)
                else:
                    mask = mod_dict[mod]['eval_mask']
                mask = mask[:,:,0] & mod_dict[mod]['inputs_attn_mask'] 

            mod_dict[mod]['inputs_mask'] = mask
            mod_dict[mod]['targets_mask'] = mask
            mod_dict[mod]['encoder_attn_mask'] = mod_dict[mod]['inputs_attn_mask']
            mod_dict[mod]['decoder_attn_mask'] = mod_dict[mod]['inputs_attn_mask']

        encoder_mod_dict = {mod: self.encoder_embeddings[mod](d)
                            for mod, d in mod_dict.items()
                            if mod in self.encoder_embeddings}
        # moco
        x_m = self.forward_moco(encoder_mod_dict) if self.use_moco else None
        encoder_tokens, encoder_emb, encoder_mask, encoder_attn_mask, encoder_mod_mask = self.forward_mask_encoder(encoder_mod_dict)

        decoder_mod_dict = {mod: self.decoder_embeddings[mod].forward_embed(d)
                            for mod, d in mod_dict.items()
                            if mod in self.decoder_embeddings}

        decoder_tokens, target_gts, decoder_emb, decoder_mask, decoder_attn_mask, decoder_mod_mask = self.forward_mask_decoder(decoder_mod_dict)

        # Encoder
        # prompt for encoder
        encoder_tokens = encoder_tokens + self.forward_encoder_prompt(encoder_tokens) if self.use_prompt else encoder_tokens
        x = encoder_tokens + encoder_emb
        x = self.forward_encoder(x, encoder_attn_mask=encoder_attn_mask)

        # Contrastive loss
        contrastive_loss_dict = None
        if self.use_contrastive and not self.use_moco:
            logits = self.forward_logits(x, decoder_mod_mask, use_moco=False)
            contrastive_loss_dict = self.forward_contrastive_loss(logits)
        elif self.use_contrastive and self.use_moco:
            spike_features, behavior_features = self.forward_logits(x, decoder_mod_mask, use_moco=True)
            moco_logits_dict = self.forward_moco_logits(x, x_m, decoder_mod_mask)
            _, _, spike_features_all, behavior_features_all, sim_targets = moco_logits_dict.values()
            sim_s2b = spike_features @ behavior_features_all / self.logit_scale.exp()
            sim_b2s = behavior_features @ spike_features_all / self.logit_scale.exp()
            contrastive_loss_dict = self.forward_moco_loss(sim_s2b, sim_b2s, sim_targets)

        # Decoder
        # prompt for decoder
        decoder_tokens = decoder_tokens + self.forward_decoder_prompt(x) if self.use_prompt else decoder_tokens # feature adaptation
        # decoder_tokens = decoder_tokens + self.forward_decoder_prompt(decoder_tokens) if self.use_prompt else decoder_tokens # token adaptation
        context = self.decoder_proj_context(x) + encoder_emb
        y = decoder_tokens + decoder_emb
        y = self.forward_decoder(y, context, encoder_attn_mask=encoder_attn_mask, decoder_attn_mask=decoder_attn_mask)
    
        decoder_mod_dict = {mod: self.decoder_embeddings[mod].out_proj(self.mod_to_indx[mod], d, y, decoder_mod_mask, len(self.avail_mod))
                            for mod, d in decoder_mod_dict.items()
                            if mod in self.decoder_embeddings}

        loss, mod_loss, mod_n_examples, mod_preds, mod_targets, contrastive_dict = self.forward_loss(decoder_mod_dict, target_gts, contrastive_loss_dict)

        return MultiModalOutput(
            loss=loss,
            mod_loss=mod_loss,
            mod_n_examples=mod_n_examples,
            mod_preds=mod_preds,
            mod_targets=mod_targets,
            contrastive_dict=contrastive_dict
        )

    
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output