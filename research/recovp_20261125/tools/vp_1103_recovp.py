"""RAW_IQ ReCoVP: invariant physical dictionary and optional soft subspace."""
import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.func import functional_call
from vp_1103_p2_smoothvp_v2 import SmoothVPNetV2
from vp_1103_recovp_extractors import extract_torch

N, FFT = 1800, 4096

def complex_input(iq):
    if iq.is_complex(): return iq.to(torch.complex128)
    return torch.complex(iq[:,0].double(),iq[:,1].double())

def normalize(iq):
    y=complex_input(iq)
    scale=y.abs().square().mean(-1,keepdim=True).sqrt().clamp_min(1e-12)
    return y,y/scale,scale

def lag_product(x):
    return torch.cat((torch.zeros_like(x[...,:1]),x[...,1:]*x[...,:-1].conj()),-1)

def real64(module,x):
    # FP32 stored parameters/Adam, FP64 real arithmetic for the strict internal path.
    parameters={name:p.double() for name,p in module.named_parameters()}
    parameters.update({name:b.double() for name,b in module.named_buffers()})
    return functional_call(module,parameters,(x,))

def radial_mad_variance(x):
    # Rotation-equivariant geometric center, then radial median absolute deviation.
    center=x.mean(-1,keepdim=True)
    for _ in range(3):
        weight=(x-center).abs().clamp_min(1e-12).reciprocal()
        center=(weight*x).sum(-1,keepdim=True)/weight.sum(-1,keepdim=True)
    return ((x-center).abs().median(-1).values.square()/math.log(2)).clamp_min(1e-12)

def soft_subspace(activity,ell,eta,epsilon=1e-8):
    weighted=activity*ell
    denominator=(ell.conj()*weighted).real.sum(-1)+epsilon
    gamma=torch.diag_embed(activity).to(torch.complex128)-weighted[...,None]*weighted.conj()[...,None,:]/denominator[...,None,None]
    gamma=eta[...,None,None]*gamma
    return .5*(gamma+gamma.mH)

class ReCoVPNet(SmoothVPNetV2):
    def __init__(self,extractor_config,share_mode="none",equivariant=True):
        super().__init__("RAW_IQ")
        if share_mode not in ("none","soft_subspace"):raise ValueError(share_mode)
        self.extractor_config=dict(extractor_config);self.k=int(extractor_config["k"])
        self.share_mode=share_mode;self.equivariant=equivariant
        self.enc0[0]=nn.Conv1d(4,18,3,padding=1)
        self.window_head=nn.Sequential(nn.Conv1d(27,12,3,padding=1),nn.SiLU(),nn.Conv1d(12,1,3,padding=1))
        self.register_buffer("unit_spectrum",self.pulse_spectrum/math.sqrt(600))
        window={"rect":torch.ones(600,dtype=torch.float64),"hann":torch.hann_window(600,periodic=False,dtype=torch.float64),
                "hamming":torch.hamming_window(600,periodic=False,dtype=torch.float64)}[extractor_config["window"]]
        pulse=torch.fft.ifft(self.unit_spectrum)[:600]
        self.register_buffer("matched_spectrum",torch.fft.fft(pulse*window,n=FFT))
        if share_mode=="soft_subspace":
            self.node_context_head=nn.Sequential(nn.Linear(90,12),nn.SiLU(),nn.Linear(12,2))
            self.eta_head=nn.Sequential(nn.Linear(90,8),nn.SiLU(),nn.Linear(8,1))
            nn.init.constant_(self.eta_head[-1].bias,-4.)

    def invariant_encode(self,x):
        matched=torch.fft.ifft(torch.fft.fft(x,n=FFT)*self.matched_spectrum.conj())[...,:N].abs()
        lag=lag_product(x)
        glob=torch.stack((x.abs(),matched,lag.real,lag.imag),1) if self.equivariant else torch.stack((x.real,x.imag,matched,x.abs()),1)
        f0=real64(self.enc0,glob)
        f1=real64(self.enc1,F.avg_pool1d(f0,2));f2=real64(self.enc2,F.avg_pool1d(f1,2))
        d1=real64(self.dec1,torch.cat((F.interpolate(f2,size=f1.shape[-1],mode="linear",align_corners=False),f1),1))
        features=real64(self.dec0,torch.cat((F.interpolate(d1,size=N,mode="linear",align_corners=False),f0),1))
        return f2,features,matched,glob

    def factor(self,iq,anchors_override=None):
        y,x,scale=normalize(iq)
        f2,features,matched,glob=self.invariant_encode(x)
        anchors=extract_torch(x,self.extractor_config,self.pulse_spectrum)
        if anchors_override is not None:
            if anchors_override.shape!=anchors.shape:raise ValueError("Wrong candidate shape")
            anchors=anchors_override.to(x.real)
        local=self.sample_features(features,anchors);peak=self.sample_features(matched[:,None],anchors)
        shared=f2.mean(-1)[:,None].expand(-1,self.k,-1);position=(anchors/N)[...,None]
        delta=2*torch.tanh(real64(self.refine_head,torch.cat((local,shared,position,torch.log1p(peak)),-1))[...,0])
        delay=anchors+delta
        phase=-2j*math.pi*self.frequency[None,None]*delay[...,None]
        template=torch.fft.ifft(self.unit_spectrum[None,None]*torch.exp(phase))[...,:N]
        z=x[:,None]*template.conj();channels=[]
        for length in (5,15,31):
            smooth=torch.complex(F.avg_pool1d(z.real,length,1,length//2,count_include_pad=False),
                                 F.avg_pool1d(z.imag,length,1,length//2,count_include_pad=False))
            lag=lag_product(smooth)
            channels.extend((smooth.abs(),lag.real,lag.imag) if self.equivariant else (smooth.abs(),smooth.real,smooth.imag))
        inputs=torch.cat((features[:,None].expand(-1,self.k,-1,-1),torch.stack(channels,2)),2).flatten(0,1)
        raw_mask=torch.sigmoid(real64(self.window_head,inputs)).reshape(-1,self.k,N)
        raw_mask=F.avg_pool1d(raw_mask,3,1,1,count_include_pad=False)
        mask_peak=raw_mask.amax(-1,keepdim=True)
        masks=raw_mask/mask_peak.clamp_min(.05)
        atom=masks*template;d=torch.linalg.vector_norm(atom,dim=-1)+1e-12
        b=(atom/d[...,None]).transpose(1,2)
        sigma=radial_mad_variance(x)
        rho=(b.mH@x[...,None])[...,0].abs().square()/sigma[:,None]
        context=torch.cat((shared,local,torch.log1p(rho)[...,None],masks.mean(-1)[...,None],position,torch.log1p(peak)),-1)
        log_precision=real64(self.shared_precision_head,context)[...,0]
        exponent=log_precision+2*(math.log(9)-torch.log1p(rho))
        precision=1e-4+torch.exp(30*torch.tanh(exponent/30))
        activity=1/(1+precision)
        if self.share_mode=="soft_subspace":
            node=real64(self.node_context_head,context)
            q=torch.exp(2*torch.tanh(node[...,1])+1j*math.pi*torch.tanh(node[...,0]))
            ell=d*q
            pooled=(context*activity[...,None]).sum(1)/activity.sum(1).clamp_min(1e-12)[...,None]
            eta=F.softplus(real64(self.eta_head,pooled)[...,0])
            gamma=soft_subspace(activity,ell,eta)
        else:
            q=torch.ones_like(d,dtype=torch.complex128);ell=d*q
            eta=torch.zeros(len(y),dtype=torch.float64,device=y.device)
            gamma=torch.zeros((len(y),self.k,self.k),dtype=torch.complex128,device=y.device)
        gram=b.mH@b;omega=torch.diag_embed(precision).to(torch.complex128)+gamma
        normal=.5*(gram+gram.mH)+omega
        right=torch.linalg.solve(normal,b.mH)
        correction=(b@(right@y[...,None]))[...,0]
        return {"correction":correction,"restored":y-correction,"b":b,"right":right,"gram":gram,
                "precision":precision,"log_precision":log_precision,"rho":rho,"activity":activity,"omega":omega,
                "gamma":gamma,"eta":eta,"ell":ell,"q":q,"atom_norm":d,"raw_mask":raw_mask,"mask_peak":mask_peak,
                "mask":masks,"anchors":anchors,"delay":delay,"scale":scale,"global_features":glob,
                "sigma_normalized":sigma,"anchor_fallback_count":torch.zeros(len(y),device=y.device)}

    def forward(self,iq):
        value=self.factor(iq)["correction"].to(torch.complex64)
        return torch.stack((value.real,value.imag),1)
