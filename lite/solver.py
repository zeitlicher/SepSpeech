import torch
from torch import Tensor
import torch.nn as nn
import pytorch_lightning as pl
from models.unet import UNet
from models.unet2 import UNet2
from models.conv_tasnet import ConvTasNet
from loss.stft_loss import MultiResolutionSTFTLoss
from typing import Tuple

'''
 PyTorch Lightning 用 solver
'''
class LitSepSpeaker(pl.LightningModule):
    def __init__(self, config:dict, model_type=None) -> None:
        super().__init__()
        if model_type is None:
            model_type = config['model_type']
        self.config = config
        if model_type == 'unet':
            self.model = UNet(config)
        elif model_type == 'unet2':
            self.model = UNet2(config)
        elif model_type == 'tasnet':
            self.model = ConvTasNet(config)
        else:
            raise ValueError('wrong parameter: '+config['model_type'])

        self.lambda1 = config['loss']['lambda1']
        self.lambda2 = config['loss']['lambda2']

        self.ce_loss = nn.CrossEntropyLoss(reduction='sum')
        self.stft_loss = MultiResolutionSTFTLoss()
        self.padding_value = self.get_padding_value()
        
        self.save_hyperparameters()

    def forward(self, mix:Tensor, enr:Tensor) -> Tuple[Tensor, Tensor]:
        return self.model(mix, enr)

    def training_step(self, batch, batch_idx:int) -> Tensor:
        mixtures, sources, enrolls, lengths, speakers = batch

        src_hat, spk_hat = self.forward(mixtures, enrolls)
        _stft_loss1, _stft_loss2 = self.stft_loss(src_hat, sources)
        _stft_loss = _stft_loss1 + _stft_loss2
        _ce_loss = self.ce_loss(spk_hat, speakers)
        _loss = self.lambda1 * _stft_loss + self.lambda2 * _ce_loss
        self.log_dict({'train_loss': _loss,
                       'train_stft_loss': _stft_loss,
                       'train_ce_loss': _ce_loss})
        return _loss

    '''
    def on_train_epoch_end(outputs:Tensor):
        #agv_loss = torch.stack([x['loss'] for x in outputs]).mean()
        #tensorboard_logs={'loss': agv_loss}
        #return {'avg_loss': avg_loss, 'log': tensorboard_logs}
    '''

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        mixtures, sources, enrolls, lengths, speakers = batch

        src_hat, spk_hat = self.forward(mixtures, enrolls)
        _stft_loss1, _stft_loss2 = self.stft_loss(src_hat, sources)
        _stft_loss = _stft_loss1 + _stft_loss2
        _ce_loss = self.ce_loss(spk_hat, speakers)
        _loss = self.lambda1 * _stft_loss + self.lambda2 * _ce_loss
        self.log_dict({'valid_loss': _loss,
                       'valid_stft_loss': _stft_loss,
                       'valid_ce_loss': _ce_loss})
        return _loss

    '''
    def on_validation_epoch_end(outputs:Tensor):
        #agv_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        #tensorboard_logs={'val_loss': agv_loss}
        #return {'avg_loss': avg_loss, 'log': tensorboard_logs}
    '''
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),
                                     **self.config['optimizer'])
        return optimizer

    def get_padding_value(self):
        self.model.cuda()
        start = -1
        end = -1
        s = torch.rand(4, 1000).cuda()
        with torch.no_grad():
            for n in range(1000, 1100):
                x = torch.rand(4, n)
                o, _ = self.model(x.cuda(), s)
                if x.shape[-1] == o.shape[-1]:
                    if start < 0:
                        start = n
                    else:
                        end = n
                        break
        return end - start
