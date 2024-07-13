import os
import numpy as np
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Dict, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.model_output import ModelOutput

@dataclass
class DecoderOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    n_examples: Optional[torch.LongTensor] = None
    preds: Optional[torch.FloatTensor] = None
    targets: Optional[torch.FloatTensor] = None


class BaselineDecoder(nn.Module):
    def __init__(
        self, in_channel, out_channel, **kwargs
    ):
        super().__init__()

        self.in_channel = in_channel
        self.out_channel = out_channel
        self.layer = nn.Linear(self.in_channel, self.out_channel)
        self.loss = nn.MSELoss(reduction="none")

    def forward_loss(
            self, preds: torch.Tensor, targets: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.LongTensor]:
        n_examples, _, _ = targets.size()
        loss = self.loss(preds, targets).sum() / n_examples
        return loss, n_examples

    def forward(
            self, data_dict: Dict[str, Dict[str, torch.Tensor]]
        ) -> DecoderOutput:

        inputs, targets = data_dict['inputs'], data_dict['targets']
        preds = self.layer(inputs)
        loss, n_examples = self.forward_loss(preds, targets)

        return DecoderOutput(
            loss=loss,
            n_examples=n_examples,
            preds=preds,
            targets=targets
        )
