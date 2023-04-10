import os, sys
import argparse
import array
import math
import numpy as np
import random
import wave
import pandas as pd

def main(args):
    df_speech = pd.read_csv(args.speech_csv)

    for index, row in df_speech.iterrows():
        speech_path=row['source']
        speech = wave.open(speech_path, 'r')
        buffer = speech.readframes(speech.getnframes())
        amplitude = (np.frombuffer(buffer, dtype="int16")).astype(np.float64)
        print(amplitude.shape)
        amplitude=np.append(amplitude, np.zeros(1000,))
        print(amplitude.shape)
        exit(1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--speech-csv', type=str, required=True)
    args=parser.parse_args()

    main(args)
