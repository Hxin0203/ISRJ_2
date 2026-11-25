"""Private failed-parent heads, including activity and subspace calibration."""
import numpy as np
import torch
from vp_1103_common import WORK,sha
from vp_1103_p2_evaluate import iq_tensor
from tools.durable_io import atomic_write
from vp_1103_recovp import complex_input

def failure_heads(model,data,cases,folder):
    ids=data["parent_id"]
    if not np.all(np.diff(ids)>0):raise ValueError("Failure export needs sorted parent IDs")
    selected=sorted({row["parent_id"] for rows in cases.values() for row in rows})
    payload={"parent_id":np.array(selected,dtype=np.int64)}
    keys=("b","delay","precision","rho","mask","raw_mask","mask_peak","q","eta","activity","gamma","omega","atom_norm","sigma_normalized")
    with torch.inference_mode():
        for parent in selected:
            index=int(np.searchsorted(ids,parent))
            s,j,e=(data[key][index].astype(np.complex128) for key in ("s","j","epsilon"))
            for state,wave in (("background",j+e),("target_present",s+j+e)):
                iq=iq_tensor(wave.reshape(-1,1800),next(model.parameters()).device)
                parts=model.factor(iq)
                for key in keys:payload[f"{parent}_{state}_{key}"]=parts[key].cpu().numpy()
                payload[f"{parent}_{state}_estimated_beta"]=(parts["right"]@complex_input(iq)[...,None])[...,0].cpu().numpy()
    path=folder/"failure_recovp_heads.npz"
    atomic_write(path,lambda f:np.savez_compressed(f,**payload))
    return {"path":path.relative_to(WORK).as_posix(),"sha256":sha(path),"parent_ids":selected,
            "keys":(*keys,"estimated_beta"),"scope":"Private local estimated heads; no arrays or model weights in Git"}
