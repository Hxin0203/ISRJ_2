"""Reproduce dataset-convention and isolated mathematical checks on CPU."""
import math
import numpy as np
import torch
from vp_1103_common import WORK,atomic_json,now,provenance,read
from vp_1103_p2_preflight import load
from vp_1103_snr_jsr import crosscheck
from vp_1103_recovp import ReCoVPNet,soft_subspace,normalize
from vp_1103_recovp_ablation import NoEquivControl
from vp_1103_recovp_cvar import self_test,record_risk,parent_batches
from vp_1103_recovp_units import run as random_math
from vp_1103_recovp_drift import three_layers
from vp_1103_p2_smoothvp import real_parameters
from paper2.network import training_loss

OUT=WORK/"evidence/outputs/recovp_gate_summary"
def run():
    torch.set_num_threads(2)
    config=read(WORK/"evidence/outputs/exp0_anchor_audit/SUMMARY.json")["locked_config"]
    data,binding=load("train");s=data["s"][:,0].astype(np.complex128);e=data["epsilon"][:,0].astype(np.complex128)
    active=np.count_nonzero(s,axis=-1)
    active_snr=10*np.log10(np.sum(abs(s)**2,-1)/active/np.mean(abs(e)**2,-1))
    whole=10*np.log10(np.sum(abs(s)**2,-1)/np.sum(abs(e)**2,-1));offset=whole-active_snr
    result={**provenance(__file__,WORK/"tools/vp_1103_snr_jsr.py"),"utc":now(),"train":binding,"metadata_check":crosscheck(data),
            "active_target_samples_min_max":[int(active.min()),int(active.max())],
            "whole_window_minus_active_snr_db_min_max":[float(offset.min()),float(offset.max())],
            "analytic_offset_db":10*math.log10(600/1800),"bins_use_only_native_metadata":True}
    atomic_json(OUT/"NATIVE_POWER_CONVENTION.json",result)
    rng=np.random.default_rng(17);ell=torch.from_numpy(rng.normal(size=(4,12))+1j*rng.normal(size=(4,12)))
    activity=torch.from_numpy(rng.uniform(.1,.9,(4,12)));eta=torch.ones(4,dtype=torch.float64);rows=[]
    for value in (1e-14,0.):
        activity[:,-1]=value;g=soft_subspace(activity,ell,eta);reduced=soft_subspace(activity[:,:-1],ell[:,:-1],eta)
        rows.append({"activity":value,"row_l2":float(torch.linalg.vector_norm(g[:,-1],dim=-1).max()),
                     "column_l2":float(torch.linalg.vector_norm(g[:,:,-1],dim=-1).max()),
                     "active_block_frobenius":float(torch.linalg.matrix_norm(g[:,:-1,:-1]-reduced).max())})
    assert all(v[k]<1e-12 for v in rows for k in ("row_l2","column_l2","active_block_frobenius"))
    atomic_json(OUT/"BOUNDARY_NORM_TEST.json",{**provenance(__file__),"utc":now(),"status":"PASS","curves":rows})
    torch.manual_seed(17);r=4;c=torch.randn(r*8,4,2,1800);b={k:torch.randn(r*8,2,1800) for k in ("s","j","e")}
    a=record_risk(c,b,r);base=training_loss(c,b["s"],b["j"],b["e"],"B4")
    error=float(abs(a.mean()-base["endpoint"]-base["damage"]))
    parents=np.concatenate(list(parent_batches(1,17)))
    assert len(parents)==len(np.unique(parents))==512 and error<1e-6
    derivative=self_test();assert derivative["status"]=="PASS"
    atomic_json(OUT/"CVAR_UNIT_TEST.json",{**provenance(__file__,WORK/"tools/vp_1103_recovp_cvar.py"),"utc":now(),"status":"PASS",
        "derivative_test":derivative,"parent_loss_matches_frozen_endpoint_damage_error":error,
        "epoch_independent_parents":512,"epoch_records":1536,"optimizer_updates":48})
    torch.manual_seed(17);base=ReCoVPNet(config);torch.manual_seed(17);control=NoEquivControl(config)
    assert all(torch.equal(v,control.state_dict()[k]) for k,v in base.state_dict().items())
    y=torch.randn(2,1800,dtype=torch.complex128);_,x,_=normalize(y)
    with torch.no_grad():a=base.invariant_encode(x)[-1];b=control.invariant_encode(x)[-1]
    assert torch.equal(a[:,:2],b[:,:2])
    atomic_json(OUT/"NOEQUIV_CONTROL_TEST.json",{**provenance(__file__,WORK/"tools/vp_1103_recovp_ablation.py"),"utc":now(),"status":"PASS",
        "same_initial_tensors":True,"same_magnitude_and_matched_filter_channel_positions":True,
        "base_real_parameters":real_parameters(base),"control_real_parameters":real_parameters(control),"training_updates":0})
    toy={key:(rng.normal(size=(1,3,8,1800))+1j*rng.normal(size=(1,3,8,1800))).astype(np.complex64) for key in ("s","j","epsilon")}
    toy["parent_id"]=np.array([0]);layers,errors=three_layers(base,toy)
    assert max(errors.values())<1e-10 and layers["frozen_B_omega"]["drift_over_target_energy"].max()==0
    atomic_json(OUT/"THREE_LAYER_ALGEBRA_TEST.json",{**provenance(__file__,WORK/"tools/vp_1103_recovp_drift.py"),"utc":now(),"status":"PASS",
        "identity_errors":errors,"frozen_operator_drift":0.,"scope":"Synthetic algebra only; not DEV","training_updates":0})
    assert random_math(config)["status"]=="PASS"
    print("Native convention, CVaR, matched-channel control, boundary and operator algebra PASS")

if __name__=="__main__":run()
