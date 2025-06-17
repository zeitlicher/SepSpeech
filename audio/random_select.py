import argparse
import array
import math
import numpy as np
import random
import wave
import pandas as pd
import os, sys

def main(args):

    df = pd.read_csv(args.csv)
    df_out=None
    for speaker in args.speakers:
        df_spk = df.query('speaker==@speaker')
        if df_out is None:
            df_out = df_spk
        else:
            df_out = pd.concat([df_out, df_spk])
    df_out.to_csv(args.output, index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, required=True)
    parser.add_argument('--speakers', type=str, nargs='*')
    parser.add_argument('--output', type=str, required=True)
    args=parser.parse_args()

    main(args)
