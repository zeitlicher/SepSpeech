import torch
import torch.nn as nn
from argparse import ArgumentParser
import yaml
import speech_dataset
from speech_dataset import SpeechDataset
import torch.utils.data as data
import torchaudio
from models.sepformer import Separator
from sdr import NegativeSISDR
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import classic_solver

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

    model = Separator(config)
    model.to(device)
    ce = nn.CrossEntropyLoss(reduce=False)
    sdr = NegativeSISDR()

    train_dataset = SpeechDataset(config['dataset']['train'], config['dataset']['train_enroll'], segment=config['train']['segment'])
    train_loader = data.DataLoader(dataset=train_dataset, batch_size=config['train']['batch_size'],
                                   shuffle=True, collate_fn=lambda x: speech_dataset.data_processing(x))
    valid_dataset = SpeechDataset(config['dataset']['valid'], config['dataset']['valid_enroll'], segment=config['train']['segment'])
    valid_loader = data.DataLoader(dataset=valid_dataset, batch_size=config['train']['batch_size'],
                                   shuffle=True, collate_fn=lambda x: speech_dataset.data_processing(x))

    optimizer = optim.Adam(model.parameters(), config['train']['learning_rate'])
    iterm = IterMeter()
    writer = SummaryWriter(log_dir=config['train']['logdir'])
    
    min_loss = 1.e10
    for epoch in range(1, config['train']['max_epochs']+1):
        classic_solver.train(model, train_loader, optimizer, [sdr, ce], iterm, epoch, writer, config)
        avg_loss = classic_solver.test(model, valid_loader, [sdr, ce], iterm, epoch, writer, config)
        if min_loss > avg_loss:
            min_loss = avg_loss
            torch.save(model.to('cpu').state_dict(), config['train']['output'])
            model.to(device)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args=parser.parse_args()

    with open(args.config, 'r') as yf:
        config = yaml.safe_load(yf)

    main(config)
