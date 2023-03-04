import math
import sys
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import torchaudio
from typing import Tuple

class SpeakerNetwork(nn.Module):
    def __init__(self, encoder, in_channels:int, out_channels:int, kernel_size:int, num_speakers:int) -> None:
        super(SpeakerNetwork, self).__init__()
        from sepformer import Encoder
        self.encoder = encoder
        self.conv = nn.Conv1d(
            in_channels,      # 256
            out_channels,
            kernel_size,      # 16
            1,
            kernel_size//2, # padding
            1
        )
        self.act = nn.ReLU()
        self.linear=nn.Linear(out_channels, num_speakers)

    def forward(self, x:Tensor) -> Tuple[Tensor,Tensor]:
        y = self.encoder(x)
        y = self.conv(y)
        y = torch.mean(y, dim=-1) # (B, C)
        z = self.linear(self.act(y)) # (B, S)
        return y, z

class SpeakerAdaptationLayer(nn.Module):
    def __init__(self) -> None:
        super(SpeakerAdaptationLayer, self).__init__()

    def forward(self, x:Tensor, s:Tensor) -> Tensor:
        # (B, C, T) x (B, C)
        y = x * s.unsqueeze(-1)
        return y
