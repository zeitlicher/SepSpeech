import numpy as np
import sys, os, re, gzip, struct
import random
import torch
import torch.nn as nn
import torch.utils.data as data
import pandas as pd
from typing import Tuple
from torch import Tensor
import torchaudio

class SpeechDataset(torch.utils.data.Dataset):

    def __init__(self, csv_path:str, enroll_path:str, sample_rate=16000, segment=0, enroll_segment=0) -> None:
        super(SpeechDataset, self).__init__()

        self.df = pd.read_csv(csv_path)
        self.segment = segment if segment > 0 else None
        self.sample_rate = sample_rate
        if self.segment is not None:
            max_len = len(self.df)
            self.seg_len = int(self.segment * self.sample_rate)
            self.df = self.df[self.df['length'] <= self.seg_len]
            print(
                f"Drop {max_len - len(self.df)} utterances from {max_len} "
                f"(shorter than {segment} seconds)"
            )
        else:
            self.seg_len = None

        self.enroll_df = pd.read_csv(enroll_path)
        self.enroll_segment = enroll_segment if enroll_segment > 0 else None
        if self.enroll_segment is not None:
            #pass
            max_len = len(self.enroll_df)
            self.seg_len = int(self.enroll_segment * self.sample_rate)
            self.enroll_df = self.enroll_df[self.enroll_df['length'] <= self.seg_len]
            print(
                f"Drop {max_len - len(self.enroll_df)} utterances from {max_len} "
                f"(shorter than {enroll_segment} seconds)"
            )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx:int) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        row = self.df.iloc[idx]
        # mixture path
        self.mixture_path = row['mixture']
        #if self.seg_len is not None:
        #    start = random.randint(0, row["length"] - self.seg_len)
        #    stop = start + self.seg_len
        #else:
        start = 0
        stop = -1

        source_path = row['source']
        if os.path.exists(source_path):
            source, sr = torchaudio.load(source_path)
            std, mean = torch.std_mean(source, dim=-1)
            source = (source - mean)/std
            source = source[:, start:stop]
        else:
            source = None
        mixture, sr = torchaudio.load(self.mixture_path)
        std, mean = torch.std_mean(mixture, dim=-1)
        mixture = (mixture - mean)/std
        mixture = mixture[:, start:stop]
        source_speaker = int(row['index'])
        filtered = self.enroll_df.query('index == @source_speaker')
        enroll_path = filtered.iloc[random.randint(0, len(filtered)-1)]['source']
        assert os.path.exists(enroll_path)
        enroll, sr = torchaudio.load(enroll_path)
        std, mean = torch.std_mean(enroll, dim=-1)
        enroll = (enroll - mean)/std
        #enroll = enroll[:, start:stop]
        #if enroll.shape[1] == 0:
        #    print("%s %d %d" % (enroll_path, start, stop))
        #print('%s %s %s' % (mixture.shape, source.shape, enroll.shape))
        return torch.t(mixture), torch.t(source), torch.t(enroll), source_speaker

'''
class SpeechDatasetLive(SpeechDataset):
    def __init__(self, csv_path:str, noise_csv_path:str, enroll_path:str, sample_rate=16000, segment=None) -> None:
        super().__init__(csv_path, enroll_path, sample_rate=16000, segment=None)
        self.noise_df = pd.read_csv(noise_csv_path)

    def __getitem__(self, idx:int) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        row = self.df.iloc[idx]
        source_path = row['source']
        if self.seg_len is not None:
            start = random.randint(0, row["length"] - self.seg_len)
            stop = start + self.seg_len
        else:
            start = 0
            stop = -1
        assert os.path.exists(source_path)
        source, sr = torchaudio.load(source_path)
        source = source[start:stop]

        source_speaker = int(row['index'])
        filtered = self.enroll_df.query('index == @source_speaker')
        enroll = filtered.iloc[randint(0, len(filtered)-1)]['source']
        enroll = enroll[start:stop]

        noise_row = self.noise_df[random.randint(0, len(self.noise_df))]
        self.noise_path = noise_row['noise']
        if self.seg_len is not None:
            start = random.randint(0, row["length"] - self.seg_len)
            stop = start + sele.seg_len
        else:
            start = random.randint(0, row["length"] - len(source))
            stop = start + len(source)
        if os.path.exists(noise_path):
            noise, sr = torchaudio.load(noise_path)
            noise = noise[start:stop]
        else:
            noise=None

        return source, noise, enroll, source_speaker
'''

def data_processing(data:Tuple[Tensor,Tensor,Tensor,Tensor]) -> Tuple[Tensor, Tensor, Tensor, list, list]:
    mixtures = []
    sources = []
    enrolls = []
    lengths = []
    speakers = []

    for mixture, source, enroll, speaker in data:
        # w/o channel
        mixtures.append(mixture)
        if source is not None:
            sources.append(source)
        enrolls.append(enroll)
        lengths.append(len(mixture))
        #speakers.append(torch.from_numpy(speaker.astype(np.int)).clone())
        speakers.append(speaker)

    mixtures = nn.utils.rnn.pad_sequence(mixtures, batch_first=True)
    if len(sources) > 0:
        sources = nn.utils.rnn.pad_sequence(sources, batch_first=True)
    enrolls = nn.utils.rnn.pad_sequence(enrolls, batch_first=True)
    speakers = torch.from_numpy(np.array(speakers)).clone()

    mixtures = mixtures.squeeze()
    sources = sources.squeeze()
    enrolls = enrolls.squeeze()

    if mixtures.dim() == 1:
        mixtures = mixtures.unsqueeze(0)
        sources = sources.unsqueeze(0)
        enrolls = enrolls.unsqueeze(0)
        
    return mixtures, sources, enrolls, lengths, speakers

'''
class SpeechDataLiveModule(SpeechDataModule):
    def __init__(self, config:dict) -> None:
        super().__init__(config)
    
    def setup(self) -> None:
        self.train_dataset=SpeechDatasetLive(
            self.config['dataset']['train'],
            self.config['dataset']['noise'],
            self.config['dataset']['train_enroll'],
            self.config['train']['sample_rate'],
            self.config['train']['segment']
        )
        self.valid_dataset=SpeechDatasetLive(
            self.config['dataset']['valid'],
            self.config['dataset']['valid_noise'],
            self.config['dataset']['valid_enroll'],
            self.config['train']['sample_rate'],
            self.config['train']['segment']
        )
        self.test_dataset=SpeechDatasetLive(
        self.config['dataset']['test'],
        self.config['dataset']['test_noise'],
        self.config['dataset']['test_enroll'],
        self.config['train']['sample_rate'],
        self.config['train']['segment']
        )
'''
'''
class SpeechDataModule(pl.LightningModule):
    def __init__(self, config:dict) -> None:
        super().__init__()
        self.config=config
        self.batch_size=8
        
    def setup(self, stage:str) -> None:
        self.train_dataset=SpeechDataset(
            self.config['dataset']['train'],
            self.config['dataset']['train_enroll'],
            self.config['train']['sample_rate'],
            self.config['train']['segment']
        )
        self.valid_dataset=SpeechDataset(
            self.config['dataset']['valid'],
            self.config['dataset']['valid_enroll'],
            self.config['train']['sample_rate'],
            self.config['train']['segment']
        )
        self.test_dataset=SpeechDataset(
        self.config['dataset']['test'],
        self.config['dataset']['test_enroll'],
        self.config['train']['sample_rate'],
        self.config['train']['segment']
        )

    def train_dataloader(self) -> data.DataLoader:
        return data.DataLoader(
                self.train_dataset,
                batch_size = self.config['train']['batch_size'],
                shuffle = True,
                collate_fn=lambda x: data_processing(x)
            )

    def val_dataloader(self) -> data.DataLoader:
        return data.DataLoader(
                self.valid_dataset,
                batch_size = self.config['train']['batch_size'],
                collate_fn=lambda x: data_processing(x)
            )
    def test_dataloader(self) -> data.DataLoader:
        return data.DataLoader(
                self.test_dataset,
                batch_size = self.config['train']['batch_size'],
                collate_fn=lambda x: data_processing(x)
            )
'''
