import pytorch_lightning as pl
import torchaudio
from lightning.solver import LitSepSpeaker
from conventional.speech_dataset import SpeechDataset, SpeechDataModule
import models.sepformer
from argparse import ArgumentParser
import yaml

def main(config:dict):
    model = LitSepSpeaker(config)
    '''
    nnet = model.get_model()

    mix_path = './samples/sample.wav'
    spk_path = './samples/sample.wav'

    mix, sr = torchaudio.load(mix_path)
    original_len = mix.shape[-1]

    len = config['sepformer']['stride'] * config['sepformer']['chunk_size']
    pad_value = len - mix.shape[-1] % len -1
    print(mix.shape)
    mix = models.sepformer.padding(mix, (0, pad_value))

    spk, sr = torchaudio.load(spk_path)
    len = config['sepformer']['stride']
    print(spk.shape)
    pad_value = len - spk.shape[-1] % len - 1
    spk = models.sepformer.padding(spk, (0, pad_value))

    out, y = nnet(mix, spk)
    out = out[:, :original_len]
    print(out.shape)

    exit(1)
    '''
    data = SpeechDataModule(config)
    trainer = pl.Trainer(
        max_epochs=config['train']['max_epochs'],
        gradient_clip_val=config['train']['gradient_clip_val'],
        precision=config['train']['precision']
    )
    trainer.fit(model, data)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args=parser.parse_args()

    with open(args.config, 'r') as yf:
        config = yaml.safe_load(yf)

    main(config)
