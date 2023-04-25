import pytorch_lightning as pl
import torchaudio
from lightning.solver import LitSepSpeaker
from conventional.speech_dataset import SpeechDataset, SpeechDataModule
import models.sepformer
from argparse import ArgumentParser
import yaml

'''
 PyTorch Lightning用 将来変更する予定
'''
def main(config:dict):
    #torch.backends.cudnn.benchmark=True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if config['train']['model_type'] == 'unet':
        model = UNet(config)
    elif config['train']['model_type'] == 'tasnet':
        model = ConvTasNet(config)
    else:
        raise ValueError('wrong parameter: '+config['train']['model_type'])
    
    if os.path.exists(config['train']['saved']):
        model.load_state_dict(torch.load(config['train']['saved'], map_location=torch.device('cpu')), strict=False)
    model.to(device)
    torch.compile(model)
            
    ce = nn.CrossEntropyLoss(reduction='none').to(device)
    stft_loss = MultiResolutionSTFTLoss().to(device)
    
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
           
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args=parser.parse_args()

    with open(args.config, 'r') as yf:
        config = yaml.safe_load(yf)

    main(config)
