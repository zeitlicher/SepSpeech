import torch
import pytorch_lightning as pl
from lightning.pytorch.loggers import TensorBoardLogger
import torch.utils.data as data
from lite.solver_ctc import LitSepSpeaker
from lite.multi_solver_ctc import LitMultiSepSpeaker
import torch.utils.data as dat
import conventional
from utils.asr_tokenizer_phone import ASRTokenizerPhone
from conventional.speech_dataset_ctc import SpeechDatasetCTC
from argparse import ArgumentParser
import yaml
import warnings
warnings.filterwarnings('ignore')

'''
 PyTorch Lightning用 将来変更する予定
'''
def main(config:dict, checkpoint_path=None):

    #if checkpoint_path is not None:
    #    model = LitMultiSepSpeaker.load_from_checkpoint(checkpoint_path, config=config)
    #else:
    model = LitMultiSepSpeaker(config)

    padding_value = 0
    if config['model_type'] == 'tasnet':
        padding_value = model.get_padding_value()
        
    tokenizer = ASRTokenizerPhone(config['dataset']['tokenizer'])
    
    train_dataset = SpeechDatasetCTC(**config['dataset']['train'], 
                                     **config['dataset']['segment'],
                                     padding_value=padding_value,
                                     tokenizer=tokenizer)
    train_loader = data.DataLoader(dataset=train_dataset,
                                   **config['dataset']['process'],
                                   pin_memory=True,
                                   shuffle=True, 
                                   collate_fn=lambda x: conventional.speech_dataset_ctc.data_processing(x))
    valid_dataset = SpeechDatasetCTC(**config['dataset']['valid'],
                                     **config['dataset']['segment'],
                                     padding_value=padding_value,
                                     tokenizer=tokenizer)
    valid_loader = data.DataLoader(dataset=valid_dataset,
                                   **config['dataset']['process'],
                                   pin_memory=True,
                                   shuffle=False, 
                                   collate_fn=lambda x: conventional.speech_dataset_ctc.data_processing(x))
    callbacks = [
        pl.callbacks.ModelCheckpoint( **config['checkpoint'])
    ]
    logger = TensorBoardLogger(**config['logger'])
    trainer = pl.Trainer( callbacks=callbacks,
                          logger=logger,
                          **config['trainer'] )
    trainer.fit(model=model, ckpt_path=args.checkpoint, train_dataloaders=train_loader,
                val_dataloaders=valid_loader)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, default=None)
    args=parser.parse_args()

    torch.set_float32_matmul_precision('high')
    with open(args.config, 'r') as yf:
        config = yaml.safe_load(yf)

    main(config, args.checkpoint)
