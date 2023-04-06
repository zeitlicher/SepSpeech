import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import torch.nn.functional as F
import speech_dataset
from speech_dataset import SpeechDataset
import numpy as np
from tqdm import tqdm
from torch import Tensor
from typing import Tuple
import torchaudio

def padding(x:Tensor, stride:int, chunk_size:int) -> Tuple[Tensor, int]:
    length = x.shape[-1]
    len = stride * chunk_size
    pad_value = len - x.shape[-1] % len -1
    return F.pad(x, (0, pad_value)), length

def truncate(x:Tensor, length:int) -> Tensor:
    return x[:, :length]

def train(model, loader, optimizer, loss_funcs, iterm, epoch, writer, config) -> None:
    model.train()

    with tqdm(total=len(loader.dataset), leave=False) as bar:
        for batch_idx, _data in enumerate(loader):
            mixtures, sources, enrolls, lengths, speakers = _data
            pd_mix, length = padding(mixtures, config['sepformer']['stride'], config['sepformer']['chunk_size'])
            optimizer.zero_grad()
            est_src, est_spk = model(pd_mix.cuda(), enrolls.cuda())
            est_src = truncate(est_src, length)

            sdr_loss = loss_funcs[0](est_src, sources)
            ce_loss = loss_funcs[1](est_spk, speakers.cuda())

            loss = torch.mean(float(config['sepformer']['lambda1']) * sdr_loss + float(config['sepformer']['lambda2']) * ce_loss)
            sdr_loss_mean = torch.mean(sdr_loss)
            ce_loss_mean = torch.mean(ce_loss)
            loss.backward()
            optimizer.step()

            if writer:
                writer.add_scalar('loss', loss.item(), iterm.get())

            bar.set_description("[Epoch %d]" % epoch)
            bar.set_postfix_str(f'{loss.item():.3f}, sdr={sdr_loss_mean.item():.3f}, ce={ce_loss_mean.item():.3f}')
            bar.update(len(mixtures))

            del loss

            torch.cuda.empty_cache()

def test(model, loader, loss_funcs, iterm, epoch, writer, config):
    model.eval()

    loss_seq=[]
    with tqdm(total=len(loader.dataset), leave=False) as bar:
        with torch.no_grad():
            for i, _data in enumerate(loader):
                mixtures, sources, enrolls, lengths, speakers = _data
                pd_mix, length = padding(mixtures, config['sepformer']['stride'], config['sepformer']['chunk_size'])

                est_src, est_spk = model(pd_mix.cuda(), enrolls.cuda())
                est_src = truncate(est_src, length)

                sdr_loss = loss_funcs[0](est_src, sources)
                ce_loss = loss_funcs[1](est_spk, speakers.cuda())

                loss = torch.mean(float(config['sepformer']['lambda1']) * sdr_loss + float(config['sepformer']['lambda2']) * ce_loss)
                loss_seq.append(loss.item())

                bar.set_description("[Epoch %d]" % epoch)
                bar.set_postfix_str(f'{np.mean(loss_seq):.3f}')
                bar.update(len(mixtures))

    avg_loss = np.mean(loss_seq)
    if writer:
        writer.add_scalar('test_loss', avg_loss, iterm.get())
    return avg_loss

def evaluate(model, mixtures, enrolls, config):
    model.eval()
    with torch.no_grad():
        d_mix, length = padding(mixtures, config['sepformer']['stride'])
        est_src, est_spk = model(pd_mix.cuda(), enrolls.cuda())
        est_src = truncate(est_src, length)

        return est_src, est_spk
