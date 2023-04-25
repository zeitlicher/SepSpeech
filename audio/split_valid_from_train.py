import argparse
import array
import math
import numpy as np
import random
import wave
import pandas as pd
import os, sys

def main(args):

    df_train = pd.read_csv(args.train_csv)
    df_src = pd.read_csv(args.source_csv)

    done=[]

    df_enr_thru=None
    df_valid_thru=None
    df_train_thru=None
    for index, row in df_train.iterrows():
        speaker = row['speaker']
        if speaker not in done:
            df_part = df_train.query('speaker==@speaker')
            df_valid = df_part.sample(args.num_valid, replace=False)
            df_part.drop(df_valid.index, inplace=True)
            df_removed_enr = df_part.sample(args.num_enroll, replace=False)
            df_part.drop(df_removed_enr.index, inplace=True)
            df_enr_filt = df_src.query('speaker==@speaker')
            for _, rrow in df_removed_enr.iterrows():
                source = rrow['source']
                df_enr_source = df_src.query('source==@source')
                if df_enr_thru is None:
                    df_enr_thru = df_enr_source
                else:
                    df_enr_thru=pd.concat([df_enr_thru, df_enr_source])
            if df_train_thru is None:
                df_train_thru = df_part
            else:
                df_train_thru = pd.concat([df_train_thru, df_part])
            if df_valid_thru is None:
                df_valid_thru = df_valid
            else:
                df_valid_thru = pd.concat([df_valid_thru, df_valid])
                
            done.append(speaker)

    df_train_thru.to_csv(args.output_train, index=False)
    df_valid_thru.to_csv(args.output_valid, index=False)
    df_enr_thru.to_csv(args.output_enroll,  index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-csv', type=str, required=True)
    parser.add_argument('--source-csv', type=str, required=True)
    parser.add_argument('--num-valid', type=int, default=1)
    parser.add_argument('--num-enroll', type=int, default=5)
    parser.add_argument('--output-train', type=str, default='train.csv')
    parser.add_argument('--output-valid', type=str, default='valid.csv')
    parser.add_argument('--output-enroll', type=str, default='enroll.csv')
    args=parser.parse_args()

    main(args)
