import argparse
import array
import math
import numpy as np
import random
import wave
import pandas as pd


'''
発話集合により発話を振り分ける：例 ATR Bset & Cset
'''
def main(args):

    df_new = None
    df = pd.read_csv(args.csv)
    for tag in args.utterance_set:
        #utt = tag+'01'
        #df_utt = df.query('utt==@utt')
        df_utt = df.query('utt.str.contains(@tag)')
        if args.remove is True:
            df.drop(index=df_utt.index, inplace=True)
        else:
            if df_new is None:
                df_new = df_utt
            else:
                df_new = pd.concat([df_new, df_utt])
    if args.remove is True:
        df.to_csv(args.output_csv, index=False)
    else:
        df_new.to_csv(args.output_csv, index=False)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, required=True)
    parser.add_argument('--output-csv', type=str, required=True)
    parser.add_argument('--remove', action='store_true')
    parser.add_argument('--utterance-set', type=str, nargs='*')
    args=parser.parse_args()

    main(args)
