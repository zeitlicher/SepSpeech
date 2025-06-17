import numpy as np
import sys, os, re, gzip, struct
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import pandas as pd
from typing import Tuple
from torch import Tensor
import torchaudio

'''
    音声強調用データの抽出
    入力: 音声CSV，エンロールCSV
    出力: 混合音声，ソース音声，エンロール音声，話者インデックス
        音声データはtorch.Tensor
'''
class SpeechDataset(torch.utils.data.Dataset):

    def __init__(self, csv_path:str,
                 enroll_path:str,
                 sample_rate=16000,
                 segment=0,
                 enroll_segment=0,
                 padding_value=0) -> None:
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
        self.padding_value = padding_value

    def __len__(self) -> int:
        return len(self.df)

    def get_padded_value(self, x):
        v = self.padding_value - x.shape[-1] % self.padding_value
        x = F.pad(x, pad=(1, v), value=0.)
        return x
    
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

        if self.padding_value > 1:
            mixture = self.get_padded_value(mixture)
            source = self.get_padded_value(source)
            
        return torch.t(mixture), torch.t(source), torch.t(enroll), source_speaker

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
