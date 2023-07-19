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
from conventional.speech_dataset import SpeechDataset

'''
    音声強調用データの抽出
    入力: 音声CSV，エンロールCSV
    出力: 混合音声，ソース音声，エンロール音声，話者インデックス
        音声データはtorch.Tensor
'''
class SpeechDatasetCTC(SpeechDataset):

    def __init__(self, csv_path:str,
                 enroll_path:str,
                 sample_rate=16000,
                 segment=0,
                 enroll_segment=0,
                 padding_value=0,
                 tokenizer=None) -> None:
        super().__init__(csv_path,
                 enroll_path,
                 sample_rate,
                 segment,
                 enroll_segment,
                 padding_value)
        self.tokenizer = tokenizer
        
    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx:int) -> Tuple[Tensor, Tensor, Tensor, int, Tensor]:
        mixture, source, enroll, speaker = super().__getitem__(idx)
        row = self.df.iloc[idx]
        label = row['label']
        label = self.tokenizer.text2token(label)
        #with open(label_path, 'r') as f:
        #    line = f.readline().strip()
        #    label = self.tokenizer.token2id(line)
            
        return mixture, source, enroll, speaker, label

def data_processing(data:Tuple[Tensor,Tensor,Tensor,int, Tensor]) -> Tuple[Tensor, Tensor, Tensor, list, Tensor, list]:
    mixtures = []
    sources = []
    enrolls = []
    lengths = []
    speakers = []
    labels = []
    label_lengths = []
    
    for mixture, source, enroll, speaker, label in data:
        # w/o channel
        mixtures.append(mixture)
        if source is not None:
            sources.append(source)
        enrolls.append(enroll)
        lengths.append(len(mixture))
        #speakers.append(torch.from_numpy(speaker.astype(np.int)).clone())
        speakers.append(speaker)

        labels.append(torch.from_numpy(np.array(label)))
        label_lengths.append(len(labels))

    mixtures = nn.utils.rnn.pad_sequence(mixtures, batch_first=True)
    if len(sources) > 0:
        sources = nn.utils.rnn.pad_sequence(sources, batch_first=True)
    enrolls = nn.utils.rnn.pad_sequence(enrolls, batch_first=True)
    speakers = torch.from_numpy(np.array(speakers)).clone()

    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)
    
    mixtures = mixtures.squeeze()
    sources = sources.squeeze()
    enrolls = enrolls.squeeze()

    if mixtures.dim() == 1:
        mixtures = mixtures.unsqueeze(0)
        sources = sources.unsqueeze(0)
        enrolls = enrolls.unsqueeze(0)
        
    return mixtures, sources, enrolls, lengths, speakers, labels, label_lengths
