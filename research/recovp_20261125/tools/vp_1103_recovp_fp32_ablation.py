"""Matched-channel NoEquiv control: keep magnitude/MF channels fixed in place."""
import torch
import torch.nn.functional as F
from vp_1103_recovp_fp32 import ReCoVPNet,real32,N,FFT

class NoEquivControl(ReCoVPNet):
    def __init__(self,config):super().__init__(config,share_mode="none",equivariant=False)
    def invariant_encode(self,x):
        matched=torch.fft.ifft(torch.fft.fft(x,n=FFT)*self.matched_spectrum.conj())[...,:N].abs()
        glob=torch.stack((x.abs(),matched,x.real,x.imag),1)
        f0=real32(self.enc0,glob)
        f1=real32(self.enc1,F.avg_pool1d(f0,2));f2=real32(self.enc2,F.avg_pool1d(f1,2))
        d1=real32(self.dec1,torch.cat((F.interpolate(f2,size=f1.shape[-1],mode="linear",align_corners=False),f1),1))
        features=real32(self.dec0,torch.cat((F.interpolate(d1,size=N,mode="linear",align_corners=False),f0),1))
        return f2,features,matched,glob

def make_recovp(config,arm):
    return NoEquivControl(config) if arm=="ReCoVP-NoEquiv" else ReCoVPNet(config,share_mode="soft_subspace" if arm=="ReCoVP-SoftShare" else "none")
