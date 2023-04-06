import math
import sys
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange
import torchaudio
import yaml
from typing import Tuple

class Reshape:
  def __init__(self, shape):
    self.shape = shape

  def __call__(self, tensor):
    return tensor.reshape(*self.shape)

class PositionEncoding(nn.Module):
    def __init__(self, config:dict) -> None:
        super(PositionEncoding, self).__init__()
        self.dropout = nn.Dropout(config['sepformer']['dropout'])

        reshape = Reshape((config['sepformer']['max_len'], config['sepformer']['d_model']))
        pe = torch.zeros(config['sepformer']['max_len'], config['sepformer']['d_model'])
        position = torch.arange(0, config['sepformer']['max_len'], dtype=torch.float)
        div_term = torch.exp(torch.arange(0, config['sepformer']['d_model'], 2).float() * (-math.log(10000.0)/config['sepformer']['d_model']))
        pe[:, 0::2] = torch.sin(position.unsqueeze(-1) * div_term.unsqueeze(0))
        pe[:, 1::2] = torch.cos(position.unsqueeze(-1) * div_term.unsqueeze(0))

        pe = reshape(pe).unsqueeze(0) # (b, t, c)
        self.register_buffer('pe', pe)

    def forward(self, x:Tensor) -> Tensor:
        # reshape applied here 
        reshape = Reshape((x.size(1), -1))
        position = torch.arange(0, x.size(1)).unsqueeze(-1)
        x = x + reshape(self.pe[:, position, :])
        return self.dropout(x)


class Encoder(nn.Module):
    def __init__(self, config:dict) -> None:
        super(Encoder, self).__init__()
        self.conv = nn.Conv1d(
            1,
            config['sepformer']['channels'],     # 256
            config['sepformer']['kernel_size'],  # 16
            config['sepformer']['stride'],       # 8
            config['sepformer']['kernel_size']//2, # padding
            1,
            bias=True
        )
        self.act = nn.ReLU()

    def forward(self, x:Tensor) -> Tensor:
        assert x.dim() == 2 # (B, T)
        x = x.unsqueeze(1) # add channel
        return self.act(self.conv(x))

class Decoder(nn.Module):
    def __init__(self, config:dict) -> None:
        super(Decoder, self).__init__()
        self.conv = nn.ConvTranspose1d(
            config['sepformer']['channels'],
            1,
            config['sepformer']['kernel_size'],
            config['sepformer']['stride'],
            config['sepformer']['kernel_size']//4,
            output_padding=0 #padding
        )
    def forward(self, x:Tensor) -> Tensor:
        assert x.dim() == 3 # (B, C, T)
        x = self.conv(x)
        x = rearrange(x, 'b c t -> (b c) t') # channels=1
        return x

class ChunkLayer(nn.Module):
    def __init__(self, config:dict) -> None:
        super(ChunkLayer,self).__init__()
        self.chunk_size=config['sepformer']['chunk_size']
        assert self.chunk_size % 2 == 0

    def forward(self, x:Tensor)->Tensor:
        # x: (B, T, C)
        _batch, _time, _channel = x.shape
        assert _time % self.chunk_size == 0

        x = torch.reshape(x, (_batch, self.chunk_size//2, _time//(self.chunk_size//2), _channel))
        y = torch.cat((x[:, :, ::2, :], x[:, :, 1::2, :]), 1)  # optimized version 

        return y

class OverlapAddLayer(nn.Module):
    def __init__(self,config:dict) -> None:
        super(OverlapAddLayer, self).__init__()
        self.chunk_size=config['sepformer']['chunk_size']
        assert self.chunk_size % 2 == 0

    def forward(self,x:Tensor) -> Tensor:
        y = torch.zeros(x.shape).cuda()
        _batch, _, _, _channel= x.shape
        y[:, :self.chunk_size//2,:, :] =  x[:, :self.chunk_size//2,:, :]
        y[:, self.chunk_size//2:,:, :] += x[:, self.chunk_size//2:,:, :]
        y = torch.reshape(y, (_batch, _channel, -1))

        return y

class SepFormerLayer(nn.Module):
    def __init__(self, config:dict) -> None:
        super(SepFormerLayer, self).__init__()
        self.pos_encoder = PositionEncoding(config)

        self.intra_T = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                                d_model=config['sepformer']['d_model'],
                                nhead=config['sepformer']['nhead'],
                                dim_feedforward=config['sepformer']['dim_feedforward'],
                                dropout=config['sepformer']['dropout'],
                                layer_norm_eps=config['sepformer']['layer_norm_eps'],
                                batch_first=True,norm_first=True
            ) ,
            config['sepformer']['num_layers'],
            nn.LayerNorm(
                config['sepformer']['d_model'],
                eps=config['sepformer']['layer_norm_eps']
            )
        )
        self.inter_T = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                                d_model=config['sepformer']['d_model'],
                                nhead=config['sepformer']['nhead'],
                                dim_feedforward=config['sepformer']['dim_feedforward'],
                                dropout=config['sepformer']['dropout'],
                                layer_norm_eps=config['sepformer']['layer_norm_eps'],
                                batch_first=True,norm_first=True
            ) ,
            config['sepformer']['num_layers'],
            nn.LayerNorm(
                config['sepformer']['d_model'],
                eps=config['sepformer']['layer_norm_eps']
            )
        )

    def forward(self, x:Tensor) -> Tensor:
        # x: (B, Intra, Inter, C)
        _batch, _intra, _inter, _channel = x.shape
        x = rearrange(x, 'b a r c-> (b r) a c')
        x = self.pos_encoder(x)
        x = self.intra_T(x)
        x = torch.reshape(x, (_batch, _inter, _intra, _channel))
        x = rearrange(x, 'b r a c -> (b a) r c')
        self.pos_encoder(x)
        x = self.inter_T(x)
        x = torch.reshape(x, (_batch, _intra, _inter, _channel))

        return x

class SepFormer(nn.Module):
    def __init__(self, config:dict) -> None:
        super(SepFormer, self).__init__()
        self.layers = nn.ModuleList()
        for n in range(config['sepformer']['num_sepformer_layers']):
            self.layers.append(SepFormerLayer(config))

    def forward(self, x:Tensor) -> Tensor:
        for mod in self.layers:
            x = mod(x)

        return x

class MaskingNetwork(nn.Module):
    def __init__(self,config:dict) -> None:
        super(MaskingNetwork, self).__init__()
        self.norm = nn.LayerNorm(
            config['sepformer']['channels'],
            eps=config['sepformer']['layer_norm_eps']
        )
        self.linear1 = nn.Linear(config['sepformer']['channels'], config['sepformer']['d_model'])
        self.chunk = ChunkLayer(config)
        self.sepformer = SepFormer(config)
        self.act1 = nn.PReLU()
        self.linear2 = nn.Linear(config['sepformer']['d_model'], config['sepformer']['channels'])
        self.overlapadd = OverlapAddLayer(config)
        self.act2 = nn.ReLU()

    def forward(self, x:Tensor) -> Tensor:
        y = rearrange(x, 'b c t -> b t c')
        y = self.norm(y)
        y = self.linear1(y)
        y = self.chunk(y)
        y = self.sepformer(y)
        y = self.linear2(self.act2(y))
        y = self.overlapadd(y)
        y = self.act2(y)
        y = rearrange(x, 'b t c -> b c t')
        return y

class Separator(nn.Module):
    def __init__(self, config:dict) -> None:
        super(Separator, self).__init__()
        self.encoder = Encoder(config)
        self.masking = MaskingNetwork(config)
        self.decoder = Decoder(config)
        spk_encoder = Encoder(config)
        from models.speaker import SpeakerNetwork
        self.speaker = SpeakerNetwork(spk_encoder, config['sepformer']['channels'], config['sepformer']['d_model'], config['sepformer']['kernel_size'], config['sepformer']['num_speakers'])
        from models.speaker import SpeakerAdaptationLayer
        self.adpt = SpeakerAdaptationLayer()

    def forward(self, x:Tensor, s:Tensor) -> Tuple[Tensor, Tensor]:
        enc_x = self.encoder(x)
        enc_s, y = self.speaker(s)
        enc_a = self.adpt(enc_x, enc_s)
        mask = self.masking(enc_a)
        mask = rearrange(mask, 'b t c -> b c t')
        out = self.decoder(enc_x*mask)

        return out, y

'''
    for module testing
'''
def padding(x:Tensor, padding:Tensor) -> Tensor:
    return F.pad(x, padding)

if __name__ == '__main__':
    with open("../config.yaml", 'r') as yf:
        config = yaml.safe_load(yf)
    mix_path = '../samples/sample.wav'
    spk_path = '../samples/sample.wav'

    mix, sr = torchaudio.load(mix_path)
    original_len = mix.shape[-1]

    len = config['sepformer']['stride'] * config['sepformer']['chunk_size']
    pad_value = len - mix.shape[-1] % len -1
    mix = padding(mix, (0, pad_value))

    spk, sr = torchaudio.load(spk_path)
    len = config['sepformer']['stride']
    pad_value = len - spk.shape[-1] % len - 1
    spk = padding(spk, (0, pad_value))

    model = Separator(config)

    out, y = model(mix, spk)
    out = out[:, :original_len]
