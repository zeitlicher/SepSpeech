import pytorch_lightning as pt
from typing import Tuple
import torch
import torch.nn.functional as F
import torch.nn as nn

def rescale_conv(conv, reference):
    std = conv.weight.std().detach()
    scale = (std / reference)**0.5
    conv.weight.data /= scale
    if conv.bias is not None:
        conv.bias.data /= scale

def rescale_module(module, reference):
    for sub in module.modules():
        if isinstance(sub, (nn.Conv1d, nn.ConvTranspose1d)):
            rescale_conv(sub, reference)

class BLSTM(nn.Module):
    def __init__(self, dim, layers=2, bi=True):
        super().__init__()
        klass = nn.LSTM
        self.lstm = klass(bidirectional=bi, num_layers=layers, hidden_size=dim, input_size=dim)
        self.linear = None
        if bi:
            self.linear = nn.Linear(2 * dim, dim)

    def forward(self, x, hidden=None):
        x, hidden = self.lstm(x, hidden)
        if self.linear:
            x = self.linear(x)
        return x, hidden

class EncoderBlock(nn.Module):
    def __init__(self, chin:int, chout:int, kernel_size:int, stride:int) -> None:
        self.block = nn.Sequential(
            nn.Conv1d(chin, chout, kernel_size, stride),
            nn.ReLU(),
            nn.Conv1d(chout, chout, kernel_size, 1),
            nn.ReLU()
        )
    
    def forward(self, x:Tensor) -> Tensor:
        return self.block(x)

class DecoderBlock(nn.Module):
    def __init__(self, chin:int, chout:int, kernel_size:int, stride:int, append=False) -> None:
        block = [
            nn.Conv1d(chin, chin, 1),
            nn.ReLU(),
            nn.ConvTranspose1d(chin, chout, kernel_size, stride),
        ]
        if append:
            block.append(nn.ReLU())
        self.block = nn.Sequential(*block)

    def forward(self, x:Tensor) -> Tensor:
        return self.block(x)

class UNet(nn.Module):
    def __init__(self, config:dict) -> None:
        super().__init__()
        depth = config['unet']['depth']

        in_channels=config['unet']['in_channels']
        mid_channels=config['unet']['mid_channels']
        out_channels=config['unet']['out_channels']
        max_channels=config['unet']['max_channels']
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        for index in range(depth):
            self.encoder.append(EncoderBlock(in_channels, mid_channels, self.kernel_size, self.stride))
            if index > 0:
                self.decoder.append(DecoderBlock(mid_channels, out_channels, self.kernel_size, self.stride, True))
            else:
                self.decoder.append(DecoderBlock(mid_channels, out_channels, self.kernel_size, self.stride, False))
            out_channels = mid_channels
            in_channels = mid_channels
            mid_channels = min(int(growth*mid_channels), max_channels)

            self.lstm = BLSTM(in_channels, bi=not config['unet']['causal'])

            if rescale:
                rescale_module(self, reference=reference)
        from speaker import SpeakerNetwork
        self.speaker = SpeakerNetwork(spk_encoder, config['unet']['in_channels'], config['sepformer']['mid_channels'], config['unet']['kernel_size'], config['unet']['num_speakers'])
        from speaker import SpeakerAdaptationLayer
        self.adpt = SpeakerAdaptationLayer()
            
    def valid_length(self, length):
        length = math.ceil(length * self.resample)
        for idx in range(self.depth):
            length = math.ceil((length - self.kernel_size)/self.stride) + 1
            length = max(length, 1)
        for idx in range(self.depth):
            length = (length - 1) * self.stride + self.kernel_size
            length = int(math.ceil(length/self.resample))
        return int(length)
    
    @property
    def total_stride(self):
        return self.stride ** seld.depth // self.resample

    def forward(self, mix, enr):
        if mix.dim() == 2:
            mix = mix.unsqueeze(1) 
        
        if self.normalize:
            mono = mix.mean(dim=1, keepdim=True)
            std = mono.std(dim=1, keepdim=True)
            mix = mix/(self.floor + std)
        else:
            std = 1
        length = mix.shape[-1]
        x = mix
        x = F.pad(x, (0, self.valid_length(length) - length))
        if self.resqmple == 2:
            x = upsample2(x)
        elif self.resqmple == 4:
            x = upsample2(x)
            x = upsample2(x)
        skips = []
        enc_s, y = self.speaker(enr)
        for n, encode in enumerate(self.encoder):
            if n == 1:
                x = self.adpt(x, enc_s)
            x = encode(x)
            skips.append(x)
        x = x.permute(2, 0, 1)
        x, _ = self.lstm(x)
        x = x.permute(1, 2, 0)
        for decode in self.decoder:
            skip = skips.pop(-1)
            x = x + skip[...,:x.shape[-1]]
            x = decode(x)
        if self.resample == 2:
            x = downsample2(x)
        elif self.resample == 4:
            x = downsample2(x)
            x = downsample2(x)

        x = x[..., :length]
        return std * x, y
        