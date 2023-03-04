import torch
from torch.nn.modules.loss import _Loss

class NegativeSISDR(_Loss):
    def __init__(self, zero_mean=True, take_log=True, reduction='none', eps=1.e-8):
        super().__init__(reduction=reduction)
        self.zero_mean = zero_mean
        self.take_log = take_log
        self.eps = eps

    def forward(self, est_target:Tensor, target:Tensor) -> Tensor:
        assert target.size() == est_target.size()
        if self.zero_mean :
            mean_source = torch.mean(target, dim=1, keepdim=True)
            mean_estimate = torch.mean(est_target, dim=1, keepdim=True)
            target = target - mean_source
            est_target = est_target - mean_estimate
                        
        dot = torch.sum(est_target * target, dim=1, keepdim=True)
        s_target_energy = torch.sum(target**2, dim=1, keepdim=True) + self.eps
        scaled_target = dot * target / s_target_energy

        e_noise = est_target - target
        losses = torch.sum(scaled_target**2, dim=1) / (torch.sum(e_noise**2, dim=1) + self.EPS)
        if self.take_log:
            losses = 10 * torch.log10(losses + self.eps)
        losses = losses.mean() if self.reduction == "mean" else losses

        return -losses
