import torch
import torch.nn.functional as F
import torchaudio
#from models.sepformer import Separator
from models.unet import UNet
from models.conv_tasnet import ConvTasNet
from argparse import ArgumentParser
import yaml
  
def main(config, args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #model = Separator(config)
    if config['model_type'] == 'unet':
        model = UNet(config)
    else:
        model = ConvTasNet(config)
    
    model.load_state_dict(torch.load(args.saved_params, map_location=torch.device('cpu')), strict=False)
    model.to(device)
    model.eval()

    mixture, sr = torchaudio.load(args.mixture)
    mx = torch.max(mixture)
    std, mean = torch.std_mean(mixture, dim=-1)
    mixture = (mixture-mean)/std
    enroll, sr = torchaudio.load(args.enroll)
    std, mean = torch.std_mean(enroll, dim=-1)
    enroll = (enroll - mean)/std
    length = len(mixture.t())
    #normalized = config['sepformer']['stride'] * config['sepformer']['chunk_size']
    #pad_value = normalized - mixture.shape[-1] % normalized -1
    #mixture = F.pad(mixture, (0, pad_value))
    with torch.no_grad():
        output, _ = model(mixture.cuda(), enroll.cuda())
    std, mean = torch.std_mean(output, dim=-1)
    #output = (output[:, :length]-mean)/std
    output /= std
    output = output/torch.max(output) * mx
    torchaudio.save(filepath=args.output, src=output.to('cpu'), sample_rate=sr)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--saved-params', type=str, required=True)
    parser.add_argument('--mixture', type=str, required=True)
    parser.add_argument('--enroll', type=str, required=True)
    parser.add_argument('--output', type=str, default='output.wav')
    args=parser.parse_args()

    with open(args.config, 'r') as yf:
        config = yaml.safe_load(yf)

    main(config, args)
