import os, sys
import argparse
import array
import math
import numpy as np
import random
import wave
import pandas as pd

# 音声ファイルから振幅を計算する
def cal_amp(wf):
    buffer = wf.readframes(wf.getnframes())
    # The dtype depends on the value of pulse-code modulation. The int16 is set for 16-bit PCM.
    amptitude = (np.frombuffer(buffer, dtype="int16")).astype(np.float64)
    return amptitude

# 音声ファイルをWAVEフォーマットで保存する
def save_waveform(output_path, params, amp):
    output_file = wave.Wave_write(output_path)
    output_file.setparams(params) #nchannels, sampwidth, framerate, nframes, comptype, compname
    output_file.writeframes(array.array('h', amp.astype(np.int16)).tobytes() )
    output_file.close()

def main(args):
    part_noise_list = {'source': [], 'noise': [], 'length': [], 'speaker': [], 'index': []}
    df_speech = pd.read_csv(args.speech_csv)
    df_noise = pd.read_csv(args.noise_csv)
    rand_noise = df_noise.sample(len(df_speech), replace=True)

    for index, row in df_speech.iterrows():
        speech_path, speaker, id = row['source'], row['speaker'], row['index']
        noise_path = rand_noise.iloc[rand_noise.index[index]]['noise']

        speech = wave.open(speech_path, 'r')
        speech_amp = cal_amp(speech)
        noise = wave.open(noise_path, 'r')
        noise_amp = cal_amp(noise)

        # 仮定：len(speech) << len(noise)
        start = random.randint(0, len(noise_amp) - len(speech_amp))
        part_noise_amp = noise_amp[start:start+len(speech_amp)]
        
        part_noise_path = os.path.join(args.output_dir,
                                       os.path.splitext(os.path.basename(noise_path))[0] + '_' + 
                                       os.path.splitext(os.path.basename(speech_path))[0])+ '.wav'
        save_waveform(part_noise_path, speech.getparams(), part_noise_amp)
        part_noise_list['source'].append(speech_path)
        part_noise_list['noise'].append(part_noise_path)
        part_noise_list['length'].append(len(speech_amp))
        part_noise_list['speaker'].append(speaker)
        part_noise_list['index'].append(id)

    df_mix = pd.DataFrame.from_dict(part_noise_list, orient='columns')
    df_mix.to_csv(args.output_csv, index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--speech-csv', type=str, required=True)
    parser.add_argument('--noise-csv', type=str, required=True)
    parser.add_argument('--output-csv', type=str, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    args=parser.parse_args()

    main(args)
