import numpy as np
import sys, os, re, gzip, struct
import random
import torch
import torch.nn as nn
import torch.utils.data as data
import pytorch_lightning as pt
import pandas as pd

class SpeechDataset(torch.utils.data.Dataset):

    def __init__(self, csv_path, enroll_path, sample_rate=16000, segment=None):
        super(SpeechDataset, self).__init__()

        self.df = pd.read_csv(csv_path)
        self.segment = segment if segment > 0 else None
        if self.segment is not None:
            max_len = len(self.df)
            self.seg_len = int(self.segment * self.sample_rate)
            self.df = self.df[self.df["length"] >= self.seg_len]
            print(
                f"Drop {max_len - len(self.df)} utterances from {max_len} "
                f"(shorter than {segment} seconds)"
            )
        else:
            self.seg_len = None

        self.enroll_df = pd.read_csv(enroll_path)
        if self.segment is not None:
            max_len = len(self.enroll_df)
            self.seg_len = int(self.segment * self.sample_rate)
            self.enroll_df = self.enroll_df[self.enroll_df['length'] >= self.seg_len]
            print(
                f"Drop {max_len - len(self.enroll_df)} utterances from {max_len} "
                f"(shorter than {segment} seconds)"
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # mixture path
        self.mixture_path = row['mixture_path']
        if self.seg_len is not None:
            start = random.randint(0, row["length"] - self.seg_len)
            stop = start + self.seg_len
        else:
            start = 0
            stop = -1

        source_path = row['source_path']
        if os.path.exists(source_path):
            source, sr = torchaudio.load(source_path)
            source = source[start:stop]
        else:
            source = None
        mixture, sr = torchaudio.load(mixture_path)
        mixture = mixture[start:stop]

        source_speaker = int(row['source_speaker_index'])
        filtered = self.enroll_df.query('source_speaker_index == @source_speaker')

        enroll = filtered.iloc[randint(0, len(filtered)-1)]['source_path']
        enroll = enroll[start:stop]

        return mixture, source, enroll, source_speaker

def data_processing(data):
    mixtures = []
    sources = []
    enrolls=[]
    lengths=[]
    speakers = []

    for mixture, source, enroll, speaker in data:
        # w/o channel
        mixtures.append(mixture)
        if source is not None:
            sources.append(source)
        enrolls.append(enroll)
        lengths.append(len(mixture))
        speakers.append(torch.from_numpy(speaker.astype(np.int)).clone())

    mixtures = nn.utils.rnn.pad_sequence(mixtures, batch_first=True)
    if sources.empty() is not True:
        sources = nn.utils.rnn.pad_sequence(sources, batch_first=True)
    enrolls = nn.utils.rnn.pad_sequence(enrolls, batch_first=True)

    return mixtures, sources, enrolls, lengths, speakers

class SpeechDataModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config=config

    def setup():
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

    def train_dataloader(self):
        return data.DataLoader(
                self.train_dataset,
                batch_size = self.config['train']['batch_size'],
                shuffle = True,
                collate_fn=lambda x: data_processing(x)
            )

    def val_dataloader(self):
        return data.DataLoader(
                self.valid_dataset,
                batch_size = self.config['train']['batch_size'],
                collate_fn=lambda x: data_processing(x)
            )
    def test_dataloader(self):
        return data.DataLoader(
                self.test_dataset,
                batch_size = self.config['train']['batch_size'],
                collate_fn=lambda x: data_processing(x)
            )
