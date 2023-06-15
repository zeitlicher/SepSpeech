import torch
from torch import Tensor
import torch.nn as nn
import pytorch_lightning as pl
from models.unet import UNet
from models.unet2 import UNet2
from models.conv_tasnet import ConvTasNet
from loss.stft_loss import MultiResolutionSTFTLoss
from typing import Tuple
from lite.solver import LitSepSpeaker
from loss.pesq_loss import PesqLoss
from loss.stoi_loss import NegSTOILoss
from loss.sdr_loss import NegativeSISDR

'''
 PyTorch Lightning 用 solver
'''
class LitMultiSepSpeaker(LitSepSpeaker):
    def __init__(self, config:dict) -> None:
        super().__init__(config)

        self.ce_loss = self.stft_loss = self.pesq_loss = self.stoi_loss = self.sdr_loss = None
        self.ce_loss_weight = self.stft_loss_weight = self.pesq_loss_weight = self.stoi_loss_weight = self.sdr_loss_weight = 0.

        if config['loss']['ce_loss']['active']:
            self.ce_loss = nn.CrossEntropyLoss(reduction='sum')
            self.ce_loss_weight = config['loss']['ce_loss']['weight']
        if config['loss']['stft_loss']['active']:
            self.stft_loss = MultiResolutionSTFTLoss()
            self.stft_loss_weight = config['loss']['stft_loss']['weight']
        if config['loss']['pesq_loss']['active']:
            self.pesq_loss = PesqLoss(factor=1.,
                                      sample_rate=16000)
            self.pesq_loss_weight = config['loss']['pesq_loss']['weight']
        if config['loss']['stoi_loss']['active']:
            self.stoi_loss = NegSTOILoss(sample_rate=16000)
            self.stoi_loss_weight = config['loss']['stoi_loss']['weight']
        if config['loss']['sdr_loss']['active']:
            self.sdr_loss = NegativeSISDR()
            self.sdr_loss_weight = config['loss']['sdr_loss']['weight']
            
        self.save_hyperparameters()
        
    def forward(self, mix:Tensor, enr:Tensor) -> Tuple[Tensor, Tensor]:
        return self.model(mix, enr)

    def compute_loss_values(self, estimates, targets, est_speakers, tgt_speakers, valid=False):
        d = {}
        _loss = 0.
        if self.ce_loss:
            _ce_loss = self.ce_loss(est_speakers, tgt_speakers)
            if valid:
                d['valid_ce_loss'] = _ce_loss
            else:
                d['train_ce_loss'] = _ce_loss
            _loss += self.ce_loss_weight * _ce_loss
            
        if self.stft_loss:
            _stft_loss1, _stft_loss2 = self.stft_loss(estimates, targets)
            _stft_loss = _stft_loss1 + _stft_loss2
            if valid:
                d['valid_stft_loss'] = _stft_loss
            else:
                d['train_stft_loss'] = _stft_loss
            _loss += self.stft_loss_weight * _stft_loss
                
        if self.pesq_loss:
            _pesq_loss = torch.mean(self.pesq_loss(targets, estimates.to(torch.float32)))
            if valid:
                d['valid_pesq_loss'] = _pesq_loss
            else:
                d['train_pesq_loss'] = _pesq_loss
            _loss += self.pesq_loss_weight * _pesq_loss
                
        if self.stoi_loss:
            _stoi_loss = torch.mean(self.stoi_loss(estimates, targets))
            if valid:
                d['valid_stoi_loss'] = _stoi_loss
            else:
                d['train_stoi_loss'] = _stoi_loss
            _loss += self.stoi_loss_weight * _stoi_loss
                
        if self.sdr_loss:
            _sdr_loss = torch.mean(self.sdr_loss(estimates, targets))
            if valid:
                d['valid_sdr_loss'] = _sdr_loss
            else:
                d['train_sdr_loss'] = _sdr_loss
            _loss += self.sdr_loss_weight * _sdr_loss

        return _loss, d
                
    def training_step(self, batch, batch_idx:int) -> Tensor:
        mixtures, sources, enrolls, lengths, speakers = batch

        src_hat, spk_hat = self.forward(mixtures, enrolls)

        _loss, d = self.compute_loss_values(src_hat,
                                            sources,
                                            spk_hat,
                                            speakers)
        d['train_loss'] = _loss
        self.log_dict(d)
        
        return _loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        mixtures, sources, enrolls, lengths, speakers = batch

        src_hat, spk_hat = self.forward(mixtures, enrolls)
        _loss, d = self.compute_loss_values(src_hat,
                                            sources,
                                            spk_hat,
                                            speakers,
                                            valid=True)
        d['valid_loss'] = _loss
        self.log_dict(d)
        
        return _loss
