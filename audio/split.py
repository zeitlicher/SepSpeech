import argparse
import array
import math
import numpy as np
import random
import wave
import pandas as pd
import os, sys

#speakers=['M060', 'M061', 'M062', 'F060', 'F061', 'F062']
speakers=[286, 295, 300, 233, 265, 271]
def main(args):

    df_mix = pd.read_csv(args.mixture_csv)
    df_enr = pd.read_csv(args.enroll_csv)

    df_valid=None
    df_enr_valid=None
    for speaker in speakers:
        df_filt = df_mix.query('index==@speaker')
        if df_valid is None:
            df_valid = df_filt
        else:
            df_valid = pd.concat([df_valid, df_filt])
        df_filt = df_enr.query('index==@speaker')
        if df_enr_valid is None:
            df_enr_valid = df_filt
        else:
            df_enr_valid = pd.concat([df_enr_valid, df_filt])
    df_train = df_mix.drop(df_valid.index)
    df_enr_train = df_enr.drop(df_enr_valid.index)

    output_dir = os.path.dirname(args.mixture_csv)
    df_train.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    df_valid.to_csv(os.path.join(output_dir, 'valid.csv'), index=False)
    df_enr_train.to_csv(os.path.join(output_dir, 'train_enr.csv'), index=False)
    df_enr_valid.to_csv(os.path.join(output_dir, 'valid_enr.csv'), index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mixture-csv', type=str, required=True)
    parser.add_argument('--enroll-csv', type=str, required=True)
    parser.add_argument('--num-valid', type=int, default=500)
    args=parser.parse_args()

    main(args)
