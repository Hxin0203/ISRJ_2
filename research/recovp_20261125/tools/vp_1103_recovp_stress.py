"""DEV-derived piecewise amplitude/phase stress; geometry is fixture-only."""
import numpy as np
import torch
from torch import nn
from vp_1103_common import WORK,read,sha,atomic_json
from vp_1103_recovp import ReCoVPNet
from vp_1103_replay_truth import replay_components
from vp_1103_p0 import evaluate
from vp_1103_snr_jsr import frame_axes,summarize

class HardSharing(nn.Module):
    def __init__(self,model):super().__init__();self.model=model
    def forward(self,iq):
        from vp_1103_recovp import complex_input
        p=self.model.factor(iq);ell=p["ell"];wave=(p["b"]@ell[...,None])[...,0]
        denominator=wave.abs().square().sum(-1)+(ell.abs().square()*p["precision"]).sum(-1)
        gain=(wave.conj()*complex_input(iq)).sum(-1)/denominator
        correction=(wave*gain[:,None]).to(torch.complex64)
        return torch.stack((correction.real,correction.imag),1)

def run(config,dev,rows):
    geometry_path=WORK/"outputs/recovp_formal/dev_geometry.npz"
    binding=read(geometry_path.with_suffix(".json"))
    if sha(geometry_path)!=binding["sha256"]:raise ValueError("DEV fixture geometry changed")
    with np.load(geometry_path,allow_pickle=False) as z:
        if not np.array_equal(z["parent_id"],dev["parent_id"]):raise ValueError("Stress roster changed")
        geometry=z["geometry"]
    results={}
    for amplitude,degrees in ((0.,0.),(.05,5.),(.15,15.)):
        stressed={key:value for key,value in dev.items()};jam=dev["j"].astype(np.complex128).copy()
        for parent in range(256):
            for family in range(3):
                for pulse in range(8):
                    parts=replay_components(float(dev["target_delay_samples"][parent]),float(dev["doppler_hz"][parent]),family,geometry[parent,family,pulse])
                    if not parts:continue
                    waves=np.stack([p["wave"] for p in parts]);summed=waves.sum(0)
                    alpha=np.vdot(summed,jam[parent,family,pulse])/np.vdot(summed,summed)
                    signs=np.random.default_rng(20261124+parent*24+family*8+pulse).choice([-1.,1.],(2,len(parts)))
                    multiplier=(1+amplitude*signs[0])*np.exp(1j*np.deg2rad(degrees)*signs[1])
                    jam[parent,family,pulse]=alpha*(waves*multiplier[:,None]).sum(0)
        stressed["j"]=jam
        for arm in ("ReCoVP-Base","ReCoVP-SoftShare"):
            for seed,row in rows[arm].items():
                cp=WORK/"outputs/recovp_formal"/arm/seed/"epoch_256.pt"
                model=ReCoVPNet(config["extractor"],share_mode="soft_subspace" if arm.endswith("SoftShare") else "none")
                if sha(cp)!=row["checkpoint_sha256"]:raise ValueError("Stress model checkpoint mismatch")
                model.load_state_dict(torch.load(cp,map_location="cpu",weights_only=False)["model"]);model=model.cpu().eval()
                variants={arm:model}
                if arm.endswith("SoftShare"):variants["HardShare-counterfactual"]=HardSharing(model).eval()
                for label,variant in variants.items():
                    metrics=evaluate(variant,stressed)
                    results[f"{label}_seed{seed}_amp{amplitude:g}_phase{degrees:g}"]={"seed":int(seed),"amplitude":amplitude,"phase_degrees":degrees,
                        "metrics":summarize(metrics,frame_axes(dev)),"checkpoint_sha256":sha(cp)}
    result={"methods":results,"scope":"Same 256 DEV parents, native baseline metadata retained for strata",
            "hard_sharing":"Exact rank-one hard limit of the same trained SoftShare heads; no separate hard-sharing training",
            "geometry":"Only to synthesize stress waveforms; never model input","independent_confirmation":False}
    atomic_json(WORK/"evidence/outputs/recovp_formal/PIECEWISE_STRESS.json",result)
    return result
