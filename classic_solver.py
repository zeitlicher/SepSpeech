import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import torch.nn.functional as F
from generator import SpeechDataset
import metric
import numpy as np
from tqdm import tqdm

def padding(x:Tensor, stride:int) -> Tuple[Tensor, int]:
    length = x.shape[-1]
    pads=stride - length % stride - 1
    return F.pad(x, (0, pads)), length

def truncate(x:Tensor, length:int) -> Tensor:
    return x[:, :length]

def train(model, loader, optimizer, loss_funcs, iterm, epoch, writer, config) -> None:
    model.train()

    with tqdm(total=len(loader.dataset), leave=False) as bar:
        for batch_idx, _data in enumerate(loader):
            mixtures, sources, enrolls, lengths, speakers = _data
            pd_mix, length = padding(mixtures, config['sepformer']['stride'])

            optimizer.zero_grad()
            est_src, est_spk = model(pd_mix.cuda(), enrolls.cuda())
            est_src = truncate(est_src, length)

            sdr_loss = loss_funcs[0](est_src, sources)
            ce_loss = loss_funcs[1](est_spk, speakers)

            loss = float(config['sepformer']['lambda1']) * sdr_loss + float(config['sepformer']['lambda2']) * ce_loss
            loss.backward()
            optimizer.step()

            if writer:
                writer.add_scalar('loss', loss.item(), iterm.get())

            bar.set_description("[Epoch %d]" % epoch)
            bar.set_postfix_str(f'{loss:.3f} ({sdr_loss:.3f}, {ce_loss:.3f})')
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
                pd_mix, length = padding(mixtures, config['sepformer']['stride'])

                est_src, est_spk = model(pd_mix.cuda(), enrolls.cuda())
                est_src = truncate(est_src, length)

                sdr_loss = loss_funcs[0](est_src, sources)
                ce_loss = loss_funcs[1](est_spk, speakers)

                loss = float(config['sepformer']['lambda1']) * sdr_loss + float(config['sepformer']['lambda2']) * ce_loss                
                loss_seq.append(loss_value.item())

                bar.set_description("[Epoch %d]" % epoch)
                bar.set_postfix_str(f'{np.mean(loss):.3f}')
                bar.update(len(mixtures))

                if writer:
                    writer.add_scalar('test_loss', np.mean(loss), iterm.get())

def evaluate(model, mixtures, enrolls, config):
    model.eval()
    with torch.no_grad():
        d_mix, length = padding(mixtures, config['sepformer']['stride'])
        est_src, est_spk = model(pd_mix.cuda(), enrolls.cuda())
        est_src = truncate(est_src, length)

        return est_src, est_spk