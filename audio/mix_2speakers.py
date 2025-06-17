import os, sys
import argparse
import array
import math
import numpy as np
import random
import wave
import pandas as pd

# 音声とSNRから二乗平均平方根(root mean square)を計算する
def cal_adjusted_rms(clean_rms, snr):
    noise_rms = clean_rms / (10**(float(snr) / 20))
    return noise_rms

# 音声ファイルから振幅を計算する
def cal_amp(wf):
    buffer = wf.readframes(wf.getnframes())
    # The dtype depends on the value of pulse-code modulation. The int16 is set for 16-bit PCM.
    amptitude = (np.frombuffer(buffer, dtype="int16")).astype(np.float64)
    return amptitude

# 振幅から二乗平均平方根を計算する
def cal_rms(amp):
    return np.sqrt(np.mean(np.square(amp), axis=-1))

# 音声ファイルをWAVEフォーマットで保存する
def save_waveform(output_path, params, amp):
    output_file = wave.Wave_write(output_path)
    output_file.setparams(params) #nchannels, sampwidth, framerate, nframes, comptype, compname
    output_file.writeframes(array.array('h', amp.astype(np.int16)).tobytes() )
    output_file.close()

'''
  JNASなどの二人の話者音声を混合する
'''    
def main(args):
    assert args.max_snr >= args.min_snr

    range = args.max_snr - args.min_snr
    df_mix = pd.DataFrame(index=None, columns=['mixture', 'source', 'noise', 'length', 'speaker', 'index', 'snr'])
    df_speech = pd.read_csv(args.speech_csv)

    # split train/enroll
    df_enr_thru=None
    df_train_thru=None

    done=[]
    for index, row in df_speech.iterrows():
        speaker = row['speaker']
        if speaker not in done:
            df_filt = df_speech.query('speaker==@speaker')
            df_enr = df_filt.sample(args.num_enroll, replace=False)
            #df_filt.drop(df_enr.index)

            if df_train_thru is None:
                df_train_thru = df_filt
            else:
                df_train_thru=pd.concat([df_train_thru, df_filt])
            if df_enr_thru is None:
                df_enr_thru = df_enr
            else:
                df_enr_thru=pd.concat([df_enr_thru, df_enr])
            done.append(speaker)
    df_train_thru.drop(df_enr_thru.index, inplace=True)

    for index, row in df_train_thru.iterrows():
        source_speech_path, source_speaker, source_length, source_id = row['source'], row['speaker'], row['length'], row['index']

        df_interf = df_train_thru.query('speaker!=@source_speaker').sample(1)
        rrow = df_interf.iloc[0]
        interf_speech_path, interf_speaker, interf_length, interf_id = rrow['source'], rrow['speaker'], rrow['length'], rrow['index']


        source_speech = wave.open(source_speech_path, 'r')
        source_speech_amp = cal_amp(source_speech)
        source_speech_rms = cal_rms(source_speech_amp)
        
        interf_speech = wave.open(interf_speech_path, 'r')
        interf_speech_amp = cal_amp(interf_speech)

        if len(source_speech_amp) < len(interf_speech_amp):
            start = random.randint(0, len(interf_speech_amp) - len(source_speech_amp))
            div_interf_speech_amp = interf_speech_amp[start:start+len(source_speech_amp)]
        else:
            start = random.randint(0, len(source_speech_amp) - len(interf_speech_amp))
            end = len(source_speech_amp) - len(interf_speech_amp) - start
            div_interf_speech_amp = np.append(np.append(np.zeros(start,), interf_speech_amp), np.zeros(end, ))

        interf_speech_rms = cal_rms(div_interf_speech_amp)
        snr = round(random.random() * range + args.min_snr, 2)
        adj_interf_speech_rms = cal_adjusted_rms(source_speech_rms, snr)
        adj_interf_speech_amp = div_interf_speech_amp * (adj_interf_speech_rms / interf_speech_rms)
        mixed_amp = source_speech_amp + adj_interf_speech_amp
            
        max_int16 = np.iinfo(np.int16).max
        min_int16 = np.iinfo(np.int16).min
        if mixed_amp.max(axis=0) > max_int16 or mixed_amp.min(axis=0) < min_int16:
            if mixed_amp.max(axis=0) >= abs(mixed_amp.min(axis=0)):
                reduction_rate = max_int16 / mixed_amp.max(axis=0)
            else :
                reduction_rate = min_int16 / mixed_amp.min(axis=0)
            mixed_amp = mixed_amp * (reduction_rate)
        mix_path = os.path.join(args.output_dir, os.path.splitext(os.path.basename(source_speech_path))[0])+ '_mix2.wav'
        save_waveform(mix_path, source_speech.getparams(), mixed_amp)
        df2 = pd.DataFrame(index=None, columns=['mixture', 'source', 'interfare', 'length', 'speaker', 'index', 'snr'],
                           data=[[mix_path, source_speech_path, interf_speech_path, len(source_speech_amp),
                                  source_speaker, source_id, snr]])
        df_mix = pd.concat([df_mix, df2], ignore_index=True)
    df_mix.to_csv(args.output_csv, index=False)
    df_enr_thru.to_csv(args.enroll_csv, index=False)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--speech-csv', type=str, required=True)
    parser.add_argument('--enroll-csv', type=str, required=True)
    parser.add_argument('--output-csv', type=str, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--max-snr', type=float, default=30.0)
    parser.add_argument('--min-snr', type=float, default=-5.0)
    parser.add_argument('--num-enroll', type=int, default=5)
    args=parser.parse_args()

    main(args)
