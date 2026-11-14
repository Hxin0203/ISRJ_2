"""Identical observed-only residual restorer and nested training objectives."""
import torch
from torch import nn
from .vendor.legacy_model import FixedDechirp, RGSSRConfig, ReliabilityGuidedUNet1D


class CorrectionNet(nn.Module):
    def __init__(self):
        super().__init__()
        cfg=RGSSRConfig(restoration_channels=(16,32,64),reliability_channels=(16,32,64),
                        use_reliability_input=False,use_reliability_features=False,gate_mode='none')
        self.rotate=FixedDechirp(cfg)
        self.core=ReliabilityGuidedUNet1D(cfg)

    def forward(self,y):
        scale=y.square().sum(1,keepdim=True).mean(-1,keepdim=True).sqrt().clamp_min(1e-8)
        z=self.rotate(y/scale)
        r=torch.ones_like(z[:,:1])
        delta,_=self.core(z,r,(r,r,r))
        return self.rotate.inverse(delta)*scale


def matched_filter(x):
    z=torch.complex(x[...,0,:],x[...,1,:])
    t=torch.arange(600,device=x.device,dtype=x.dtype)/60e6-5e-6
    reference=torch.exp(1j*torch.pi*(30e6/10e-6)*t.square())
    n=1 << (x.shape[-1]+599).bit_length()
    # Circular indexing of zero-padded linear correlation; all lags retained.
    return torch.fft.ifft(torch.fft.fft(z,n=n)*torch.fft.fft(reference,n=n).conj())


def training_loss(c,s,j,e,method):
    label=torch.stack((0*j,0*j,j,j),1)
    desired=torch.stack((e,s+e,e,s+e),1)
    err=c-label
    denom=desired.square().sum((-2,-1)).clamp_min(1e-12)
    endpoint=(err.square().sum((-2,-1))/denom).mean()
    pc=endpoint*0
    damage=endpoint*0
    if method in ('B2','B4'):
        pc=(matched_filter(err).abs().square().sum(-1)/matched_filter(desired).abs().square().sum(-1).clamp_min(1e-12)).mean()
    if method=='B4':
        damage=((c[:,3]-c[:,2]).square().sum((-2,-1))/s.square().sum((-2,-1)).clamp_min(1e-12)).mean()
    if method not in ('B1','B2','B4'):
        raise ValueError('unregistered training method')
    return dict(total=endpoint+.1*pc+damage,endpoint=endpoint,pc=pc,damage=damage)
