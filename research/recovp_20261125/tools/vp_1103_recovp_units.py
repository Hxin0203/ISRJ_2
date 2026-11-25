"""Independent sixteen-input random mathematical checks, with no dataset reads."""
import numpy as np
import torch
from vp_1103_recovp import ReCoVPNet
from vp_1103_recovp_math import math_checks
from vp_1103_common import WORK,atomic_json,provenance,now,sha

def run(config,checkpoint=None):
    torch.set_num_threads(2)
    rng=np.random.default_rng(20261124)
    waves=(rng.normal(size=(16,1800))+1j*rng.normal(size=(16,1800)))/np.sqrt(2)
    torch.manual_seed(17)
    model=ReCoVPNet(config,share_mode="none")
    if checkpoint is not None:model.load_state_dict(torch.load(checkpoint,map_location="cpu",weights_only=False)["model"])
    base=math_checks(model,waves)
    torch.manual_seed(17)
    soft=math_checks(ReCoVPNet(config,share_mode="soft_subspace"),waves)
    result={**provenance(__file__,WORK/"tools/vp_1103_recovp_math.py",WORK/"tools/vp_1103_recovp.py"),
            "utc":now(),"base":base,"softshare_initial":soft,"dataset_reads":0,"training_updates":0,
            "checkpoint_sha256":sha(checkpoint) if checkpoint else None,
            "status":"PASS" if base["status"]==soft["status"]=="PASS" else "FAIL"}
    atomic_json(WORK/"evidence/outputs/recovp_gate_summary"/("RANDOM_MATH_FINAL.json" if checkpoint else "RANDOM_MATH_INITIAL.json"),result)
    return result
