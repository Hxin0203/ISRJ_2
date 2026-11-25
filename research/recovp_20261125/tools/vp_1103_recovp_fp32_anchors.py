"""Observation-only peak candidates. No trainable parameters or truth inputs."""
import numpy as np
from scipy.ndimage import maximum_filter1d, uniform_filter1d
from scipy.fft import fft, ifft
import torch
import torch.nn.functional as F

N, FFT = 1800, 4096

def pulse_numpy():
    t = np.arange(600)/60e6-5e-6
    return np.exp(1j*np.pi*3e12*t*t)

def spectra(window):
    weights = {"rect": np.ones(600), "hann": np.hanning(600), "hamming": np.hamming(600)}[window]
    return fft(pulse_numpy()*weights, n=FFT)

def select_numpy(envelope, count, selected=None):
    available = np.ones(envelope.shape, bool)
    available[:, :2] = available[:, -2:] = False
    position = np.arange(N)[None]
    if selected is not None:
        available &= (abs(position[:,:,None]-selected[:,None,:]) >= 6).all(-1)
    local = envelope >= maximum_filter1d(envelope, 13, axis=-1, mode="constant", cval=-np.inf)
    result = []
    for _ in range(count):
        score = np.where(available & local, envelope, -np.inf)
        fallback = ~np.isfinite(score).any(-1)
        score[fallback] = np.where(available, envelope, -np.inf)[fallback]
        index = score.argmax(-1)
        result.append(index)
        available &= abs(position-index[:,None]) >= 6
    return np.stack(result,-1)

def extract_numpy(wave, config):
    wave=wave.astype(np.complex128)
    scale=np.maximum(np.sqrt(np.mean(abs(wave)**2,axis=-1,keepdims=True)),1e-12)
    wave=wave/scale
    residual = wave.copy()
    k, rounds = config["k"], config["rounds"]
    counts = [k] if rounds == 1 else [4,k-4] if rounds == 2 else [3,3,k-6]
    selected = None
    spectrum = spectra(config["window"])
    pos = np.arange(N)[None]
    pulse = pulse_numpy()
    denominator = uniform_filter1d(np.ones(N),11,mode="constant")
    for count in counts:
        stopped=np.linalg.norm(residual,axis=-1)<=1e-3*np.linalg.norm(wave,axis=-1)
        residual[stopped]=wave[stopped]  # Fixed-K fillers after relative CLEAN stopping.
        envelope = abs(ifft(fft(residual,n=FFT,axis=-1)*spectrum.conj(),axis=-1)[...,:N])
        anchors = select_numpy(envelope,count,selected)
        selected = anchors if selected is None else np.concatenate((selected,anchors),-1)
        if selected.shape[-1] < k:
            for column in anchors.T:
                local = pos-column[:,None]
                template = np.where((local>=0)&(local<600), pulse[np.clip(local,0,599)],0)
                z = residual*template.conj()
                smooth = (uniform_filter1d(z.real,11,axis=-1,mode="constant")+
                          1j*uniform_filter1d(z.imag,11,axis=-1,mode="constant"))/denominator
                residual -= .8*template*smooth
    return selected.astype(float)

def extract_torch(wave, config, pulse_spectrum):
    # Candidate extraction is fixed and detached. The learned dictionary still uses original wave.
    with torch.no_grad():
        wave=wave.detach().to(torch.complex64)
        wave=wave/torch.sqrt(wave.abs().square().mean(-1,keepdim=True)).clamp_min(1e-12)
        residual = wave.clone()
        k, rounds = config["k"], config["rounds"]
        counts = [k] if rounds == 1 else [4,k-4] if rounds == 2 else [3,3,k-6]
        pulse = torch.fft.ifft(pulse_spectrum)[:600]
        weights = {"rect": lambda: torch.ones(600,device=wave.device,dtype=torch.float32),
                   "hann": lambda: torch.hann_window(600,periodic=False,device=wave.device,dtype=torch.float32),
                   "hamming": lambda: torch.hamming_window(600,periodic=False,device=wave.device,dtype=torch.float32)}
        spec = torch.fft.fft(pulse*weights[config["window"]](),n=FFT)
        selected = None
        position = torch.arange(N,device=wave.device)[None]
        for count in counts:
            stopped=torch.linalg.vector_norm(residual,dim=-1)<=1e-3*torch.linalg.vector_norm(wave,dim=-1)
            residual=torch.where(stopped[:,None],wave,residual)
            envelope = torch.fft.ifft(torch.fft.fft(residual,n=FFT)*spec.conj())[...,:N].abs()
            localmax = envelope >= F.max_pool1d(envelope[:,None],13,1,6)[:,0]
            available = ((position>=2)&(position<N-2)).expand_as(envelope).clone()
            if selected is not None:
                available &= (abs(position[:,:,None]-selected[:,None,:])>=6).all(-1)
            anchors = []
            for _ in range(count):
                scores = envelope.masked_fill(~(available&localmax),-torch.inf)
                missing = ~torch.isfinite(scores).any(-1)
                scores = torch.where(missing[:,None],envelope.masked_fill(~available,-torch.inf),scores)
                index = scores.argmax(-1)
                anchors.append(index)
                available &= abs(position-index[:,None])>=6
            anchors = torch.stack(anchors,-1)
            selected = anchors if selected is None else torch.cat((selected,anchors),-1)
            if selected.shape[-1] < k:
                for column in anchors.T:
                    offset = position-column[:,None]
                    template = torch.where((offset>=0)&(offset<600),pulse[offset.clamp(0,599)],0)
                    z = residual*template.conj()
                    smooth = torch.complex(F.avg_pool1d(z.real[:,None],11,1,5,count_include_pad=False)[:,0],
                                           F.avg_pool1d(z.imag[:,None],11,1,5,count_include_pad=False)[:,0])
                    residual -= .8*template*smooth
        return selected.to(wave.real.dtype)

def configurations():
    return {f"{name}_{window}_k{k}": {"method":name,"window":window,"rounds":rounds,"k":k}
            for k in (8,10,12) for name,rounds,windows in
            (("E0",1,("rect",)),("E1",1,("hann","hamming")),
             ("E2",2,("hann","hamming")),("E3",3,("hann","hamming"))) for window in windows}
