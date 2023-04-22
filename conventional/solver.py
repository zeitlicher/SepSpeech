import sys,os
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import torch.nn.functional as F
from torch import Tensor
import torchaudio
import conventional.speech_dataset
from conventional.speech_dataset import SpeechDataset
import numpy as np
from tqdm import tqdm
from typing import Tuple

def padding(x:Tensor, stride:int, chunk_size:int) -> Tuple[Tensor, int]:
    length = x.shape[-1]
    len = stride * chunk_size
    pad_value = len - x.shape[-1] % len -1
    return F.pad(x, (0, pad_value)), length

def truncate(x:Tensor, length:int) -> Tensor:
    return x[:, :length]

'''
    音声強調モデルの学習
    ルートディレクトリのtrain.pyから呼び出される学習プログラムの実体
    2023.4.23 モデルの実装はConvTasNet，DenoiserスタイルのUNet
'''
def train(model, loader, optimizer, loss_funcs, iterm, epoch, writer, config) -> None:
    model.train()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loss_seq=[]
    stft_loss_seq=[]
    with tqdm(total=len(loader.dataset), leave=False) as bar:
        for batch_idx, _data in enumerate(loader):
            mixtures, sources, enrolls, lengths, speakers = _data
            #pd_mix, length = padding(mixtures,
            #config['sepformer']['stride'],
            #config['sepformer']['chunk_size'])
            optimizer.zero_grad()
            est_src, est_spk = model(mixtures.cuda(), enrolls.cuda())
            #est_src = truncate(est_src, length)

            stft_loss1, stft_loss2 = loss_funcs[0](est_src, sources)
            stft_loss = stft_loss1 + stft_loss2
            ce_loss = loss_funcs[1](est_spk, speakers.cuda())

            loss = torch.mean(float(config['train']['lambda1']) * stft_loss
                              + float(config['train']['lambda2']) * ce_loss)
            stft_loss_mean = torch.mean(stft_loss)
            ce_loss_mean = torch.mean(ce_loss)
            loss.backward()
            loss_seq.append(loss.item())
            stft_loss_seq.append(stft_loss_mean.item())
            optimizer.step()
            iterm.step()
            
            if writer:
                writer.add_scalar('loss', loss.item(), iterm.get())

            if iterm.get() % config['train']['checkpoint'] == 0:
                ckpt_path = os.path.join(os.path.dirname(config['train']['output']), 'checkpoint_'+'{:09d}'.format(iterm.get()))
                torch.save(model.to('cpu').state_dict(), ckpt_path)
                model.to(device)

            bar.set_description("[Epoch %d]" % epoch)
            bar.set_postfix_str(f'{loss.item():.3e}, stft={stft_loss_mean.item():.3e}, ce={ce_loss_mean.item():.3e}')
            bar.update(len(mixtures))

            del loss

            torch.cuda.empty_cache()
    print(f"train loss: {np.mean(loss_seq):.3e} {np.mean(stft_loss_seq):.3e}")

'''
    音声強調モデルの開発データ評価...モデルの実装はConvTasNet，DenoiserスタイルのUNet
'''    
def test(model, loader, loss_funcs, iterm, epoch, writer, config):
    model.eval()

    loss_seq=[]
    stft_loss_seq=[]
    with tqdm(total=len(loader.dataset), leave=False) as bar:
        with torch.no_grad():
            for i, _data in enumerate(loader):
                mixtures, sources, enrolls, lengths, speakers = _data
                #pd_mix, length = padding(mixtures, config['sepformer']['stride'],
                #                         config['sepformer']['chunk_size'])

                est_src, est_spk = model(mixtures.cuda(), enrolls.cuda())
                #est_src = truncate(est_src, length)

                stft_loss1, stft_loss2 = loss_funcs[0](est_src, sources)
                stft_loss = stft_loss1+stft_loss2
                ce_loss = loss_funcs[1](est_spk, speakers.cuda())

                loss = torch.mean(float(config['train']['lambda1']) * stft_loss
                                  + float(config['train']['lambda2']) * ce_loss)
                loss_seq.append(loss.item())
                stft_loss_seq.append(torch.mean(stft_loss).item())
                bar.set_description("[Epoch %d]" % epoch)
                bar.set_postfix_str(f'{np.mean(loss_seq):.3e}')
                bar.update(len(mixtures))

    avg_loss = np.mean(loss_seq)
    avg_stft_loss = np.mean(stft_loss_seq)
    print(f"valid loss: {np.mean(loss_seq):.3e} {np.mean(stft_loss_seq):.3e}")
    if writer:
        writer.add_scalar('test_loss', avg_loss, iterm.get())
    return avg_loss

def evaluate(model, mixtures, enrolls, config):
    model.eval()
    with torch.no_grad():
        #d_mix, length = padding(mixtures, config['sepformer']['stride'])
        est_src, est_spk = model(mixtures.cuda(), enrolls.cuda())
        #est_src = truncate(est_src, length)

        return est_src, est_spk
