import sys
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from model import PositionEncoding
from asteroid.masknn.norm import GlobLN # non-causal

'''
    (TODO) Conv-TasNet
'''
class Conv1dBlock(nn.Module):
    def __init__(self, config, dilation_factor=0):
        super(Conv1dBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv1d(
                config['in_channels'],
                config['block_channels'],
                kernel_size=1, stride=1, padding=0, dilation=1
                ),
            nn.PReLU(),
            GlobLN(),
            nn.Conv1d(
                config['block_channels'],
                config['block_channels'],
                kernel_size=1, stride=1, padding=0, dilation=dilation_factor,
                groups=config['block_channels']
            ),
            nn.PReLU(),
            GlobLN()
        )
        self.conv1 = nn.Conv1d(
                config['block_channels'],
                config['in_channels'],
                kernel_size=1, stride=1, padding=0, dilation=1
        )
        self.conv2 = nn.Conv1d(
            config['block_channels'],
            config['in_channels'],
            kernel_size=1, stride=1, padding=0, dilation=1
        )

    def forward(self, x):
        assert x.dims() == 3
        y = self.block(x)
        out = self.conv1(y)+x
        skip = self.conv2(y)

        return out, skip
