from __future__ import annotations
import math
import sys
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import torchaudio
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from sepformer import Encoder

class SpeakerNetwork(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_speakers):
        super(SpeakerNetwork, self).__init__()
        self.encoder = Encoder(config)
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

    def forward(self, x):
        y = self.encoder(x)
        y = self.conv(y)
        y = torch.mean(y, dim=-1) # (B, C)
        z = self.linear(self.act(y)) # (B, S)
        return y, z

class SpeakerAdaptationLayer(nn.Module):
    def __init__(self):
        super(SpeakerAdaptationLayer, self).__init__()

    def forward(self, x, s):
        # (B, C, T) x (B, C)
        y = x * s.unsqueeze(-1)
        return y
