import argparse
import json
import logging
import sys, os
from argparse import ArgumentParser
import pandas as pd
import yaml
import numpy as np

parser = ArgumentParser()
parser.add_argument('--csv', type=str, required=True)
args = parser.parse_args()

df = pd.read_csv(args.csv)

mean_mix_pesq=mean_mix_stoi=mean_mix_sdr=0.
mean_est_pesq=mean_est_stoi=mean_est_sdr=0.

for index, row in df.iterrows():
    mean_mix_pesq += row['mix_pesq']
    mean_est_pesq += row['est_pesq']
    mean_mix_stoi += row['mix_stoi']
    mean_est_stoi += row['est_stoi']
    mean_mix_sdr  += row['mix_sdr']
    mean_est_sdr  += row['est_sdr']

mean_mix_pesq /= len(df)
mean_est_pesq /= len(df)
mean_mix_stoi /= len(df)
mean_est_stoi /= len(df)
mean_mix_sdr  /= len(df)
mean_est_sdr  /= len(df)

print(f'PESQ   mix: {mean_mix_pesq:.3f} est: {mean_est_pesq:.3f}')
print(f'STOI   mix: {mean_mix_stoi:.3f} est: {mean_est_stoi:.3f}')
print(f'SI-SDR mix: {mean_mix_sdr:.3f} est: {mean_est_sdr:.3f}')
