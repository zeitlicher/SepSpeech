import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger
from lite.solver import LitSepSpeaker
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

    train_dataset = SpeechDataset(**config['dataset']['train'], 
                                  **config['dataset']['segment'])
    train_loader = data.DataLoader(dataset=train_dataset,
                                   **config['dataset']['process'],
                                   pin_memory=True,
                                   shuffle=True, 
                                   collate_fn=lambda x: conventional.speech_dataset.data_processing(x))
    valid_dataset = SpeechDataset(**config['dataset']['valid'],
                                  **config['dataset']['segment'])
    valid_loader = data.DataLoader(dataset=valid_dataset,
                                   **config['dataset']['process'],
                                   pin_memory=True,
                                   shuffle=False, 
                                   collate_fn=lambda x: conventional.speech_dataset.data_processing(x))
    callbacks = [
        pl.callbacks.ModelCheckpoint( **config['checkpoint'])
    ]
    logger = TensorBoardLogger(**config['logger'])
    trainer = pl.Trainer( callbacks=callbacks, **config['trainer'] )
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=valid_loader, logger=logger)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, default=None)
    args=parser.parse_args()

    with open(args.config, 'r') as yf:
        config = yaml.safe_load(yf)

    main(config, args.checkpoint)
