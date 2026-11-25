"""Parent-level smoothed Rockafellar-Uryasev CVaR and exact two-pass weights."""
import numpy as np
import torch
import torch.nn.functional as F

ALPHA,TEMPERATURE,RISK_WEIGHT=.9,.05,.05

def parent_batches(epoch,seed):
    # Same 48 optimizer updates/epoch and 1536 records; all four Exp3 cells use this collation.
    rng=np.random.default_rng(seed+100_000+epoch)
    yield from np.array_split(rng.permutation(512),48)

def record_risk(correction,batch,records):
    c=correction.reshape(records,8,4,2,1800)
    s,j,e=(batch[key].reshape(records,8,2,1800) for key in ("s","j","e"))
    label=torch.stack((0*j,0*j,j,j),2)
    desired=torch.stack((e,s+e,e,s+e),2)
    endpoint=((c-label).square().sum((-2,-1))/desired.square().sum((-2,-1)).clamp_min(1e-12)).mean((1,2))
    damage=((c[:,:,3]-c[:,:,2]).square().sum((-2,-1))/s.square().sum((-2,-1)).clamp_min(1e-12)).mean(1)
    return endpoint+damage

def parent_losses(record_losses):
    if record_losses.numel()%3:raise ValueError("Each independent parent needs its three ordered families")
    return record_losses.reshape(-1,3).mean(1)

def ru_cvar(losses,threshold,alpha=ALPHA,temperature=TEMPERATURE):
    return threshold+temperature*F.softplus((losses-threshold)/temperature).mean()/(1-alpha)

def two_pass_weights(losses,threshold):
    weights=torch.sigmoid((losses-threshold)/TEMPERATURE)/(1-ALPHA)
    return weights.detach(),(1-weights.mean()).detach()

def self_test():
    rng=torch.Generator().manual_seed(20261124)
    values=torch.randn(4,3,generator=rng,dtype=torch.float64,requires_grad=True)
    threshold=torch.tensor(.2,dtype=torch.float64,requires_grad=True)
    direct=ru_cvar(values.mean(1),threshold)
    grad,tgrad=torch.autograd.grad(direct,(values,threshold))
    weights,manual_tgrad=two_pass_weights(values.detach().mean(1),threshold.detach())
    manual=weights[:,None].expand(-1,3)/(4*3)
    network_error=float((grad-manual).abs().max());threshold_error=float(abs(tgrad-manual_tgrad))
    return {"status":"PASS" if network_error<1e-12 and threshold_error<1e-12 else "FAIL",
            "network_gradient_error":network_error,"threshold_gradient_error":threshold_error,
            "independent_parents":4,"families_per_parent":3,"risk_weight":RISK_WEIGHT,
            "alpha":ALPHA,"temperature":TEMPERATURE,
            "scope":"Exact gradients of smoothed parent RU objective, not a top-k surrogate"}
