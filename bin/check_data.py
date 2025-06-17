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
import numpy as np
warnings.filterwarnings('ignore')

'''
 PyTorch Lightning用 将来変更する予定
'''
def main(config:dict):

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

    number_of_model_classes=config['unet']['ctc']['output_class']
    max_class_id=0
    for idx, batch in enumerate(train_loader):
        mixtures, sources, enrolls, lengths, speakers, labels, label_lengths = batch
     
        labels = labels.cpu().detach().numpy().ravel().tolist()
        for class_id in labels:
            max_class_id = max(max_class_id, class_id)
        assert max_class_id < number_of_model_classes

        print("***")
        print(type(mixtures))
        print(type(sources))
        print(type(enrolls))
        print(type(lengths))
        print(type(speakers))
        print(type(labels))
        print(type(label_lengths))

        print(labels.shape)
        print('***')
        valid_lengths = np.array([ model.model.valid_length_encoder(l) for l in label_lengths ])
        extend_label_lengths = np.array([ 2*l+1 for l in label_lengths ])
        for n in range(len(valid_lengths)):
            assert  extend_label_lengths[n] < valid_lengths[n]
            assert  extend_label_lengths[n] >= 1
        
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args=parser.parse_args()

    torch.set_float32_matmul_precision('high')
    with open(args.config, 'r') as yf:
        config = yaml.safe_load(yf)

    main(config)
