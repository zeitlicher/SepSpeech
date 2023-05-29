import argparse
import json
import logging
import sys, os
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality
from torchmetrics.audio.stoi import ShortTImeObjectiveIntelligibility
from torchmetrics import ScaleInvariantSignalDistortionRatio
import torch
import torch.nn.functional as F
import torchaudio
import lightning.pytorch as pl
from lite.solver import LitSepSpeaker
import conventional.speech_dataset as sd
from argparse import ArgumentParser
import pandas as pd
import yaml
import numpy as np
import whisper

def get_divisor(model):

    start = -1
    end = -1
    with torch.no_grad():
        for n in range(1000, 1100):
            x = torch.rand(4, n)
            o = model(x)
            if x.shape[-1] == o.shape[-1] :
                if start < 0:
                    start = n
                else:
                    end = n
                    break
    return end - start

def padding(x, divisor):
    pad_value = divisor - x.shape[-1] % divisor -1
    return F.pad (x, pad=(1, pad_value ), value=0.)

def read_audio(path):
    wave, sr = torchaudio.load(path)
    std, mean = torch.std_mean(wave, dim=-1)
    wave = (wave - mean)/std
    return torch.t(wave)

def main(args):

    decoder = whisper.load_model("large")

    with open(args.config, 'r') as yf:
        config = yaml.safe_load(yf)

    config['model_type'] = args.model_type
    assert args.checkpoint is not None
    model = LitSepSpeaker.load_from_checkpoint(args.checkpoint,
                                               config=config)
    divisor = get_divisor(model)
    model.eval()
    
    sample_rate = config['segment']['sample_rate']
    _pesq = PerceptualEvaluationSpeechQuality(sample_rate, 'wb')
    _stoi = ShortTImeObjectiveIntelligibility(sample_rate, extended=False)
    _sdr = ScaleInvariantSignalDistortionRatio()
    
    df_out = pd.DataFrame(index=None, 
                          columns=['key', 'mixture', 'source', 'enroll', 'estimate', 'mix_result', 'est_result', 'mix_pesq', 'mix_stoi', 'mix_sdr', 'est_pesq', 'est_stoi', 'est_sdr'])
    keys = []
    est_pesq, est_stoi, est_sdr = [], [], []
    mix_pesq, mix_stoi, mix_sdr = [], [], []
    mixtures, sources, enrolls, estimates = [], [], [], []
    mix_decoded, est_decoded = [], [], []
    df = pd.read_csv(args.input_csv)
    df_enroll = pd.read_csv(args.enroll_csv)
    rand_enroll = df_enroll.sample(len(df), replace=True)
    with torch.no_grad():
        for index, row in df.iterrows():

            key = os.path.splitext(os.path.basename(row['source']))[0]
            keys.append(key)
            mixtures.append(row['mixture'])
            sources.append(row['source'])

            mixture, source = read_audio(row['mixture']),read_audio(row['source'])
            mixture_original_length = mixture.shape[-1]
            source_original_length = source.shape[-1]
            assert mixture_original_length == source_original_length

            enroll_path = df_enroll.iloc[df_enroll[index]]['source']
            enroll = read_audio(enroll_path)
            enrolls.append(enroll)

            # padding
            if divisor > 0 and mixture_original_length % divisor > 0:
                mixture = sd.padding(mixture, divisor)
                
            estimate, _ = model(mixture.cuda(), enroll.cuda())
            estimate = estimate[:, :mixture_original_length]
            
            estimate_outdir = args.output_dir
            if not os.path.exists(estimate_outdir):
                os.path.mkdir(estimate_outdir)
            outpath = os.path.join(estimate_outdir, key) + '_estimate.wav'
            torchaudio.save(filepath=outpath, src=estimate.to('cpu'), sample_rate=sample_rate)
            estimates.append(outpath)

            decoded = decoder.transcribe(outpath, verbose=False, language='ja')
            est_decoded.append(decoded['text'])
            decoded = decoder.transcribe(row['mixture'], verbose=False, language='ja')
            mix_decoded.append(decoded['text'])
            
            source = source.cpu().detach().numpy()
            estimate = estimate.cpu().detach().numpy()

            pesq_val = _pesq(estimate, source)
            stoi_val = _stoi(estimate, source)
            sdr = _sdr(estimate, source)

            est_pesq.append(_pesq(estimate, source).cpu().detach().numpy())
            est_stoi.append(_stoi(estimate, source).cpu().detach().numpy())
            est_sdr.append(_sdr(estimate, source).cpu().detach().numpy())

            mix_pesq.append(_pesq(mixture[:, :mixture_original_length], source).cpu().detach().numpy())
            mix_stoi.append(_stoi(mixture[:, :mixture_original_length], source).cpu().detach().numpy())
            mix_sdr.append(_sdr(mixture[:, :mixture_original_length], source).cpu().detach().numpy())
            
    df_out['key'], df_out['mixture'], df_out['source'], df_out['enroll'], df_out['estimate'] = keys, mixtures, sources, enrolls, estimates
    df_out['mix_pesq'], df_out['mix_stoi'], df_out['mix_sdr'] = mix_pesq, mix_stoi, mix_sdr
    df_out['est_pesq'], df_out['est_stoi'], df_out['est_sdr'] = est_pesq, est_stoi, est_sdr
    df_out['mix_result'], df_out['est_result'] = mix_decoded, est_decoded

    pesq, stoi, sdr = np.mean(est_pesq), np.mean(est_stoi), np.mean(est_sdr)
    print("mixture:  PESQ = %.4f , STOI = %.4f, SI-SDR = %.4f" % (pesq, stoi, sdr))
    
    pesq, stoi, sdr = np.mean(est_pesq), np.mean(est_stoi), np.mean(est_sdr)
    print("estimate: PESQ = %.4f , STOI = %.4f, SI-SDR = %.4f" % (pesq, stoi, sdr))

    df.to_csv(args.output_csv, index=False)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--input_csv', type=str)
    parser.add_argument('--enroll_csv', type=str)
    parser.add_argument('--output_csv', type=str)
    parser.add_argument('--output_dir')
    parser.add_argument('--model_type', type=str, default='tasnet')
    args = parser.parse_args()

    main(args)

