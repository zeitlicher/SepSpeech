import torch
import torch.utils.data as data
import pytorch_lightning as pl
from lite.solver import LitSepSpeaker
import conventional
from conventional.speech_dataset import SpeechDataset
import models.sepformer
from argparse import ArgumentParser
import yaml

'''
 PyTorch Lightning用 将来変更する予定
'''
def main(config:dict, checkpoint_path=None):

    if checkpoint_path is not None:
        model = LitSepSpeaker.load_from_checkpoint(checkpoint_path, config=config)
    else:
        model = LitSepSpeaker(config)
    
    train_dataset = SpeechDataset(config['dataset']['train'],
                                  config['dataset']['train_enroll'],
                                  segment=config['train']['segment'])
    train_loader = data.DataLoader(dataset=train_dataset,
                                   batch_size=config['train']['batch_size'], num_workers=2, pin_memory=True,
                                   shuffle=True, collate_fn=lambda x: conventional.speech_dataset.data_processing(x))
    valid_dataset = SpeechDataset(config['dataset']['valid'],
                                  config['dataset']['valid_enroll'],
                                  segment=config['train']['segment'])
    valid_loader = data.DataLoader(dataset=valid_dataset,
                                   batch_size=config['train']['batch_size'], num_workers=2, pin_memory=True,
                                   shuffle=True, collate_fn=lambda x: conventional.speech_dataset.data_processing(x))
    trainer = pl.Trainer(
        accelerator='auto',
        accumulate_grad_batches=config['train']['accumulate_grad_batches'],
        default_root_dir=config['train']['default_root_dir'],
        max_epochs=config['train']['max_epochs'],
        precision=config['train']['precision'],
        profiler='simple',
    )
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=valid_loader)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, default=None)
    args=parser.parse_args()

    with open(args.config, 'r') as yf:
        config = yaml.safe_load(yf)

    main(config, args.checkpoint)
