import os, sys
import argparse
import array
import math
import numpy as np
import random
import wave
import pandas as pd
import mix


def main(args):

    snr = args.snr
    df_mix = pd.DataFrame(index=None,
                          columns=['mixture', 'source', 'noise', 'length', 'speaker', 'index', 'snr'])

    speech = wave.open(args.speech, 'r')
    speech_amp = mix.cal_amp(speech)
    speech_rms = mix.cal_rms(speech_amp)
    noise = wave.open(args.noise, 'r')
    noise_amp = mix.cal_amp(noise)

    # 仮定：len(speech) << len(noise)
    start = random.randint(0, len(noise_amp) - len(speech_amp))
    div_noise_amp = noise_amp[start:start+len(speech_amp)]
    noise_rms = mix.cal_rms(div_noise_amp)
    adj_noise_rms = mix.cal_adjusted_rms(speech_rms, snr)
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
    mix_path = os.path.join(args.output_dir,
                            os.path.splitext(os.path.basename(args.speech))[0])+ '_mix.wav'
        
    mix.save_waveform(mix_path, speech.getparams(), mixed_amp)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--speech', type=str, required=True)
    parser.add_argument('--noise', type=str, required=True)
    parser.add_argument('--output-dir', type=str, default='./')
    parser.add_argument('--snr', type=float, default=20.0)
    args=parser.parse_args()

    main(args)
