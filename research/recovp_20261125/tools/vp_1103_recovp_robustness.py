"""DEV-derived paired complex-scaling audit; no additional data partitions."""
import numpy as np
import torch
from vp_1103_common import WORK,atomic_json,sha
from vp_1103_p2_smoothvp import real_parameters
from vp_1103_recovp import ReCoVPNet
from vp_1103_recovp_ablation import make_recovp
from vp_1103_p0 import evaluate
from vp_1103_snr_jsr import frame_axes,summarize
from tools.unet_ablation.models import make_model

def scaling_audit(config,data):
    rng=np.random.default_rng(20261124)
    amplitude=rng.uniform(.2,5.,len(data["parent_id"]));phase=rng.uniform(0,2*np.pi,len(amplitude))
    c=amplitude*np.exp(1j*phase)
    transformed={key:value for key,value in data.items()}
    for key in ("s","j","epsilon"):
        transformed[key]=data[key].astype(np.complex128)*c[:,None,None,None]
    rows={}
    for name in ("ReCoVP-Base","ReCoVP-NoEquiv","P0-NoDechirp-Tail"):
        if name.startswith("ReCoVP"):
            model=make_recovp(config["extractor"],name)
            cp=WORK/"outputs/recovp_formal"/name/"17/epoch_256.pt"
        else:
            model=make_model("no_dechirp")
            cp=WORK/"outputs/p0_dechirp_tail_2x2/P0-NoDechirp-Tail/17/epoch_256.pt"
        model.load_state_dict(torch.load(cp,map_location="cpu",weights_only=False)["model"])
        model=model.cpu().eval()
        original=evaluate(model,data);scaled=evaluate(model,transformed)
        rows[name]={"standard":summarize(original,frame_axes(data)),"complex_scaled":summarize(scaled,frame_axes(data)),
                    "paired_scaled_minus_standard":summarize({k:scaled[k]-original[k] for k in original},frame_axes(data)),
                    "checkpoint_sha256":sha(cp),"effective_real_parameters":real_parameters(model)}
    result={"seed":20261124,"models":rows,"scope":"Same 256 DEV parents; deterministic derived scaling, native metadata unchanged",
            "amplitude_range":[.2,5.],"phase_range":[0,2*np.pi],"independent_confirmation":False}
    atomic_json(WORK/"evidence/outputs/recovp_formal/COMPLEX_SCALING.json",result)
    return result
