import os, sys
import argparse
import array
import math
import numpy as np
import random
import wave
import pandas as pd

def main(args):
    df_speech = pd.read_csv(args.csv)

    total_length_in_sec = 0.
    for index, row in df_speech.iterrows():
        sample=row['length']
        total_length_in_sec += sample / 16000
    total_length_in_hour = total_length_in_sec/3600
    print(total_length_in_hour)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, required=True)
    args=parser.parse_args()

    main(args)
