import pytorch_lightning as pt
from models.sepformer import Separator
from losses import NegativeSISDR

class LitSepSpeaker(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = Separator(self.config)
        self.ce = nn.CrossEntropyLoss()
        self.sdr = NegativeSISDR()

    def padding(self, x):
        self.org_len = x.shape[-1]
        pads=self.config['model']['stride'] - x.shape[-1] % self.config['model']['stride'] - 1
        x = F.pad(x, (0, pads))

    def truncate(self, x):
        return x[:, :self.org_len]

    def forward(self, mix, enr):
        return self.model(mix, enr)

    def training_step(self, batch, batch_idx):
        mix, src, enr, _, spk = batch
        mix = padding(mix)
        est_src, est_spk = self.model(mix, enr)
        # TODO solve tensor-size mismatching
        est_src = truncate(est_src)
        sdr_loss = self.sdr(est_src, src)
        ce_loss = self.ce(est_spk, spk)
        loss = self.lambda1 * sdr_loss + self.lambda2 * ce_loss
        values = {'loss': loss, 'sdr': sdr_loss, 'ce': ce_loss}
        self.log_dict(values)

        return loss

    def training_epoch_end(outputs):
        loss = torch.mean(outputs)

    def validation_step(self, batch, batch_idx, dataloader_idx):
        mix, src, enr, len, spk = batch
        est_src, est_spk = self.model(mix, enr)
        sdr_loss = self.sdr(est_src, src, len)
        ce_loss = self.ce(est_spk, spk)
        loss = self.lambda1 * sdr_loss + self.lambda2 * ce_loss
        values = {'val_loss': loss, 'val_sdr': sdr_loss, 'val_ce': ce_loss}
        self.log_dict(values)

    def validation_epoch_end(outputs):
        loss = torch.mean(outputs)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
