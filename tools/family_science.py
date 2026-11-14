"""Eight-state objective and separate-target gain constraints."""
import numpy as np
import torch
from paper2.network import matched_filter
from paper2.science import energy,metrics,parent_mean


def eight_states(a,b,j,e):
    y=np.stack([e+ia*a+ib*b+ih*j for ia in (0,1) for ib in (0,1) for ih in (0,1)],1)
    label=np.stack([ih*j for ia in (0,1) for ib in (0,1) for ih in (0,1)],1)
    return y,label


def family_loss(c,a,b,j,e,method):
    desired=torch.stack([e+ia*a+ib*b for ia in (0,1) for ib in (0,1) for ih in (0,1)],1)
    label=torch.stack([ih*j for ia in (0,1) for ib in (0,1) for ih in (0,1)],1)
    err=c-label
    endpoint=(err.square().sum((-2,-1))/desired.square().sum((-2,-1)).clamp_min(1e-12)).mean()
    pc=endpoint*0;response=endpoint*0
    if method in ('B2','B4'):
        pc=(matched_filter(err).abs().square().sum(-1)/matched_filter(desired).abs().square().sum(-1).clamp_min(1e-12)).mean()
    if method=='B4':
        terms=[]
        for left,right,target in [(5,1,a),(7,3,a),(3,1,b),(7,5,b)]:
            terms.append((c[:,left]-c[:,right]).square().sum((-2,-1))/target.square().sum((-2,-1)).clamp_min(1e-12))
        response=torch.stack(terms).mean()
    if method not in ('B1','B2','B4'):
        raise ValueError(method)
    return dict(total=endpoint+.1*pc+response,endpoint=endpoint,pc=pc,response=response)


def family_gain(a,b,j,e,c,ids,budget,tail_limit=None,complex_control=True):
    a,b,j,e,c=[np.asarray(x,np.complex128) for x in (a,b,j,e,c)]
    den=energy(a+b+e).clip(1e-24)
    aa=np.mean(energy(c[:,7])/den)
    q=np.mean((c[:,7].conj()*j).sum(-1)/den)
    radius=2.
    for left,right,target in [(7,3,a),(7,5,b)]:
        d=energy(c[:,left]-c[:,right])/energy(target).clip(1e-24)
        if d.mean()>0:
            radius=min(radius,float(np.sqrt(budget/d.mean())))
        if tail_limit is not None:
            worst=np.array([np.max(d[ids==p]) for p in np.unique(ids)])
            allowed=int(np.floor(tail_limit*len(worst)))
            threshold=np.sort(worst)[::-1][allowed]
            if threshold>0:
                radius=min(radius,float(np.sqrt(.25/threshold))*(1-1e-12))
    if aa<=1e-24:
        return 0.
    optimum=q/aa if complex_control else max(0.,float(q.real/aa))
    return optimum*radius/abs(optimum) if abs(optimum)>radius else optimum


def family_metrics(a,b,j,e,c,gain,ids):
    results={}
    for name,target,other,index in [('strong',a,b,3),('weak',b,a,5)]:
        # Metrics expects target/no-target pairs in positions 3 and 2.
        no_j_other=2 if name=='strong' else 4
        mapped=c[:,[no_j_other,6,index,7]]
        m=metrics(target,j,other+e,mapped,gain)
        parents={k:parent_mean(v,ids) for k,v in m.items()}
        result={k:float(v.mean()) for k,v in parents.items()}
        result['damage_p95']=float(np.quantile(parents['damage'],.95))
        result['severe_parent_rate']=float(np.mean([np.any(m['damage'][ids==p]>.25) for p in np.unique(ids)]))
        result['circular_phase_mean_deg']=float(np.rad2deg(np.angle(np.exp(1j*np.deg2rad(m['phase_deg'])).mean())))
        result['median_abs_phase_deg']=float(np.median(np.abs(m['phase_deg'])))
        results[name]=result
    return dict(u_nmse=results['strong']['u_nmse'],strong=results['strong'],weak=results['weak']),parents['u_nmse']
