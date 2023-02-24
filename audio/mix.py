import argparse
import array
import math
import numpy as np
import random
import wave
import pandas as pd

# 音声とSNRから二乗平均平方根(root mean square)を計算する
def cal_adjusted_rms(clean_rms, snr):
    a = float(snr) / 20
    noise_rms = clean_rms / (10**a)
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

def main(args):
    assert args.max_snr >= args.min_snr

    range = args.max_snr - args.min_snr
    df_mix = pd.DataFrame(index=None, columns=['mixture', 'source', 'noise', 'length', 'speaker', 'index', 'snr'])
    with pd.read_csv(args.speech_csv) as df_speech:
        with pd.read_csv(args.noise_csv) as df_noise:
            rand_noise = df_noise.sample(len(df_speech), replace=True)
            for index, tuple in enumerate(df_speech.itertuples()):
                speech_path, speaker, id = row['source'], row['speaker'], row['index']
                noise_path = rand_noise[rand_noise.index[index], 'noise']

                speech = wave.open(speech_path, 'r')
                speech_amp = cal_amp(speech)
                speech_rms = cal_rms(speech)
                noise = wave.open(noise_path, 'r')
                noise_amp = cal_amp(noise)

                # len(speech) << len(noise)
                start = random.randint(0, len(noise_amp) - len(speech_amp))
                div_noise_amp = noise_amp[start:start+len(speech_amp)]
                noise_rms = cal_rms(div_noise_amp)
                snr = round(random.random() * range + args.min_snr, 2)
                adj_noise_rms = cal_adjusted_rms(speech_rms, snr)
                adj_noise_amp = div_noise_amp * (adj_noise_rms / noise_rms)
                mixed_amp = speech_amp + adj_noise_amp

                max_int16 = np.iinfo(np.int16).max
                min_int16 = np.iinfo(np.int16).min
                if mixed_amp.max(axis=0) > max_int16 or mixed_amp.min(axis=0) < min_int16:
                  if mixed_amp.max(axis=0) >= abs(mixed_amp.min(axis=0)):
                    reduction_rate = max_int16 / mixed_amp.max(axis=0)
                  else :
                    reduction_rate = min_int16 / mixed_amp.min(axis=0)
                    mixed_amp = mixed_amp * (reduction_rate)
                mix_path = os.path.join(args.output_dir, os.path.splitext(os.path.basename(speech_path))[0])+ '_mix.wav'
                save_waveform(mix_path, speach_wav.getparams(), mixed_amp)
                df_mix.append(data=[[mix_path, speech_path, noise_path, len(speech_amp), speaker, id, snr]])
    df_mix.to_csv(args.output_csv)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--speech-csv', type=str, required=True)
    parser.add_argument('--noise-csv', type=str, required=True)
    parser.add_argument('--output-csv', type=str, required=True)
    parser.add_argument('--output-dir', type=str, required=True)

    args.parser.parse_args()

    main(args)
