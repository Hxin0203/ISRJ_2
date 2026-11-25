"""RAW_IQ shared physical heads with frozen TRAIN-selected peak candidates."""
import math
import torch
import torch.nn.functional as F
from vp_1103_p2_smoothvp import N, MIN_PRECISION
from vp_1103_p2_smoothvp_v2 import SmoothVPNetV2
from vp_1103_v21_extractors import extract_torch

class SmoothVPNetV21(SmoothVPNetV2):
    def __init__(self, extractor_config, fixed_lambda=False):
        super().__init__("RAW_IQ")
        self.extractor_config=dict(extractor_config)
        self.k=int(extractor_config["k"])
        self.fixed_lambda=fixed_lambda
        if self.k not in (8,10,12): raise ValueError("Unsupported candidate count")
        if fixed_lambda: del self.shared_precision_head

    def factor(self, iq, anchors_override=None):
        y,scale,f2,features,matched=self.encode(iq)
        anchors=extract_torch(y,self.extractor_config,self.pulse_spectrum)
        if anchors_override is not None:
            if anchors_override.shape!=anchors.shape: raise ValueError("Candidate shape changed")
            anchors=anchors_override
        local=self.sample_features(features,anchors)
        peak=self.sample_features(matched[:,None],anchors)
        shared=f2.mean(-1)[:,None].expand(-1,self.k,-1)
        position=(anchors/N).unsqueeze(-1)
        delta=2*torch.tanh(self.refine_head(torch.cat((local,shared,position,torch.log1p(peak)),-1))[...,0])
        delay=anchors+delta
        phase=-2j*math.pi*self.frequency[None,None]*delay.double()[...,None]
        template=torch.fft.ifft(self.pulse_spectrum[None,None]*torch.exp(phase))[...,:N]
        z=y[:,None].to(torch.complex128)*template.conj()
        channels=[]
        for length in (5,11):
            smooth=torch.complex(F.avg_pool1d(z.real,length,1,length//2,count_include_pad=False),
                                 F.avg_pool1d(z.imag,length,1,length//2,count_include_pad=False))
            channels.extend((smooth.real,smooth.imag,smooth.abs()))
        channels.extend((template.abs(),y.abs()[:,None].expand(-1,self.k,-1)))
        physical=torch.stack(channels,2).float()
        inputs=torch.cat((features[:,None].expand(-1,self.k,-1,-1),physical),2).flatten(0,1)
        masks=torch.sigmoid(self.window_head(inputs)).reshape(-1,self.k,N)
        masks=F.avg_pool1d(masks,3,1,1,count_include_pad=False)
        b=masks.double()*template
        b=(b/torch.linalg.vector_norm(b,dim=-1,keepdim=True).clamp_min(1e-12)).transpose(1,2)
        yd=y.to(torch.complex128)
        rho=(b.mH@yd[...,None])[...,0].abs().square()
        noise=(y.abs().square().median(-1).values/math.log(2)).double().clamp_min(1e-12)
        rho=rho/noise[:,None]
        if self.fixed_lambda:
            log_precision=torch.zeros_like(rho)
            precision=torch.full_like(rho,MIN_PRECISION)
        else:
            inputs=torch.cat((shared,local,torch.log1p(rho).float()[...,None],masks.mean(-1)[...,None],position,torch.log1p(peak)),-1)
            log_precision=self.shared_precision_head(inputs)[...,0].double()
            exponent=log_precision+2*(math.log(9)-torch.log1p(rho))
            precision=MIN_PRECISION+torch.exp(30*torch.tanh(exponent/30))
        gram=b.mH@b
        normal=.5*(gram+gram.mH)+torch.diag_embed(precision).to(torch.complex128)
        right=torch.linalg.solve(normal,b.mH)
        correction=(b@((right@yd[...,None])[...,0])[...,None])[...,0]*scale[:,0]
        return {"correction":correction,"b":b,"right":right,"delay":delay,"mask":masks,
                "log_precision":log_precision,"precision":precision,"rho":rho,"scale":scale,"gram":gram,
                "anchors":anchors,"refinement":delta,"anchor_fallback_count":torch.zeros(len(iq),device=iq.device)}
