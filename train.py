import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter
import torchaudio
import conventional.speech_dataset
from conventional.speech_dataset import SpeechDataset
from models.unet import UNet
from models.conv_tasnet import ConvTasNet
from loss.sdr import NegativeSISDR
from loss.stft_loss import MultiResolutionSTFTLoss
import conventional.solver
from argparse import ArgumentParser
import yaml
import sys, os

class IterMeter(object):
    """keeps track of total iterations"""
    def __init__(self):
        self.val = 0

    def step(self):
        self.val += 1

    def get(self):
        return self.val
    
def main(config:dict):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if config['train']['model_type'] == 'unet':
        model = UNet(config)
    elif config['train']['model_type'] == 'tasnet':
        model = ConvTasnet(config)
    else:
        raise ValueError('wrong parameter: '+config['train']['model_type'])
    
    if os.path.exists(config['train']['saved']):
        model.load_state_dict(torch.load(config['train']['saved'], map_location=torch.device('cpu')), strict=False)
    model.to(device)
        
    ce = nn.CrossEntropyLoss(reduction='none').to(device)
    #sdr = NegativeSISDR()
    sdr = MultiResolutionSTFTLoss().to(device)
    
    train_dataset = SpeechDataset(config['dataset']['train'],
                                  config['dataset']['train_enroll'],
                                  segment=config['train']['segment'])
    train_loader = data.DataLoader(dataset=train_dataset,
                                   batch_size=config['train']['batch_size'],
                                   shuffle=True, collate_fn=lambda x: conentional.speech_dataset.data_processing(x))
    valid_dataset = SpeechDataset(config['dataset']['valid'],
                                  config['dataset']['valid_enroll'],
                                  segment=config['train']['segment'])
    valid_loader = data.DataLoader(dataset=valid_dataset,
                                   batch_size=config['train']['batch_size'],
                                   shuffle=True, collate_fn=lambda x: conventional.speech_dataset.data_processing(x))

    optimizer = optim.Adam(model.parameters(), config['train']['learning_rate'])
    iterm = IterMeter()
    writer = SummaryWriter(log_dir=config['train']['logdir'])
    
    min_loss = 1.e10
    for epoch in range(1, config['train']['max_epochs']+1):
        conventional.solver.train(model, train_loader, optimizer,
                                  [sdr, ce], iterm, epoch, writer, config)
        avg_loss = conventional.solver.test(model, valid_loader,
                                            [sdr, ce], iterm,
                                            epoch, writer, config)
        if min_loss > avg_loss:
            min_loss = avg_loss
            torch.save(model.to('cpu').state_dict(), config['train']['output'])
            model.to(device)

        chk_point = os.path.join(os.path.dir_name(config['train']['output']), 'checkpoint.pt')
        torch.save(model.to('cpu').state_dict(), chk_point)
        model.to(device)
        
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args=parser.parse_args()

    with open(args.config, 'r') as yf:
        config = yaml.safe_load(yf)

    main(config)
