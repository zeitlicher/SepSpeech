import pytorch_lightning as pl
from typing import Tuple
import torch
import torch.nn.functional as F

'''
    拡散モデル
'''
class DiffusionSolver(pl.LightningModule):
    def __init__(self, config:dict, model) -> None:
        super().__init__()
        self.model = model
        # forward process
        self.timesteps = config['diffusion']['timesteps'] #200
        self.beta_start = config['diffusion']['beta_start'] # 0.001
        self.beta_end = config['diffusion']['beta_end'] # 0.02
        self.betas = linear_beta_schedule(timesteps=self.timesteps)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

        # reverse process
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1,0), value=1.0)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1./self.alphas)
        self.ce = nn.CrossEntropyLoss()
        
        self.lambda1 = config['diffusion']['lambda1']
        self.lambda2 = config['diffusion']['lambda2']

    def p_losses(self, noise:Tensor, est_noise:Tensor):
        loss = F.l2_loss(noise, est_noise)
        return loss 

    def extract(self, a, t, x_shape):
        batch_size = t.shape[0]
        out = a.gather(-1, t.cpu())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

    def linear_beta_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.timesteps)

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    @torch.no_grad()
    def p_sample(self, x, s, t, t_index):
        betas_t = extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t, x.shape)
        model_mean = sqrt_recip_alphas_t * (x - betas_t * self.model(x,s,t)[0] / sqrt_one_minus_cumprod_t)
        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = extract(self.posterior_variance, t, x.shape)
        noise = torch.randn_like(x)

        return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def p_sample_loop(self, shape):
        device = next(self.model.parameters()).device
        b=shape[0]
        img = torch.randn(shape, device=device)
        imgs=[]

        for i in tqdm(reversed(range(0, self.timesteps)), total=self.timesteps):
            img = p_sample(img, torch.full((b,), i, device=device, dtype=torch.long), i)
            imgs.append(img.cupu().numpy())
        return imgs

    @torch.no_grad()
    def sample(self, image_size, batch_size=16, channels=3):
        return p_sample_loop(shape=(batch_size, channels, image_size, image_size))

    def forward(self, mix:Tensor, enr:Tensor, noise:Tensor, t:int) -> Tuple[Tensor, Tensor]:
        if noise is None:
            noise = torch.randn_like(mix)
        x_noisy = q_sample(x_start=x_start, t=t, noise=noise)
        predicted_noise, predicted_spk = self.model(x_noisy, enr)

        return noise, predicted_noise, predicted_spk

    def training_step(self, batch, batch_idx:int) -> Tensor:
        mix, src, enr, _, spk = batch
        batch_size = len(mix)
        t = torch.randint(0, self.timesteps, (batch_size,), device=device).long()
        noise, est_noise, est_spk = self.forward(mix, enr, t)

        loss1 = p_losses(noise, est_noise)
        loss2 = self.ce(est_spk, spk)
        loss = self.lambda1 * loss1 + self.lambda2 * loss2
        values = {'loss': loss, 'p_loss': loss1, 'ce': loss2}
        #self.log_dict(values)

        return values

    def training_epoch_end(self, outputs:Tensor):
        agv_loss = torch.stack([x['loss'] for x in outputs]).mean()
        tensorboard_logs={'loss': agv_loss}
        return {'avg_loss': avg_loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx, dataloader_idx):
        mix, src, enr, _, spk = batch

        t = torch.randint(0, self.timesteps, (batch_size,), device=device).long()
        noise, est_noise, est_spk = self.forward(mix, enr, t)
        loss1 = p_loss(noise, est_noise)
        loss2 = self.ce(est_spk, spk)
        loss = self.lambda1 * loss1 + self.lambda2 * loss2
        values = {'val_loss': loss, 'val_p_loss': loss1, 'val_ce': loss2}
        #self.log_dict(values)
        return values

    def validation_epoch_end(self, outputs:Tensor):
        agv_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs={'val_loss': agv_loss}
        return {'avg_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
