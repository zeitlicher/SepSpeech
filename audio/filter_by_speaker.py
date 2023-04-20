import argparse
import array
import math
import numpy as np
import random
import wave
import pandas as pd


def main(args):

    speakers=[]
    with open(args.speakers, 'r') as f:
        lines = f.readlines()
        for line in lines:
           speakers.append(line.strip())

    df = pd.read_csv(args.csv)
    for index, row in df.iterrows():
        speaker = row['speaker']
        if args.remove is True:
            if speaker in speakers:
                df.drop(index=index, inplace=True)
                continue
        else:
            if speaker not in speakers:
                df.drop(index=index, inplace=True)
                continue
    df.to_csv(args.output_csv, index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, required=True)
    parser.add_argument('--output-csv', type=str, required=True)
    parser.add_argument('--remove', action='store_true')
    parser.add_argument('--speakers', type=str, required=True)
    args=parser.parse_args()

    main(args)
