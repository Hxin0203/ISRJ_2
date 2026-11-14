"""Complex128 physical metrics and closed-form global-gain controls."""
import numpy as np


def four_states(s, j, e):
    if not (s.shape == j.shape == e.shape):
        raise ValueError('component shapes differ')
    return np.stack((e, s+e, j+e, s+j+e), 1), np.stack((0*j, 0*j, j, j), 1)


def energy(x):
    return np.sum(np.abs(x)**2, axis=-1)


def metrics(s, j, e, correction, gain=1):
    s,j,e,c = [np.asarray(x, np.complex128) for x in (s,j,e,correction)]
    if c.shape != (len(s),4,s.shape[-1]):
        raise ValueError('expected [records,4,length] correction')
    es, eu, ee = [energy(x).clip(1e-24) for x in (s,s+e,e)]
    r0 = j-gain*c[:,2]
    rs = -gain*(c[:,3]-c[:,2])
    response = s+rs
    g = (s.conj()*response).sum(-1)/es
    return dict(u_nmse=energy(j-gain*c[:,3])/eu,
                damage=energy(rs)/es, r0_u=energy(r0)/eu,
                rs_u=energy(rs)/eu, cross_u=2*(r0.conj()*rs).sum(-1).real/eu,
                background_nmse=energy(r0)/ee,
                h0_nmse=energy(gain*c[:,1])/eu,
                noise_nmse=energy(gain*c[:,0])/ee,
                amplitude=np.abs(g), phase_deg=np.angle(g,deg=True))


def parent_mean(values, parent_ids):
    values, parent_ids = np.asarray(values),np.asarray(parent_ids)
    return np.array([values[parent_ids==p].mean(axis=0) for p in np.unique(parent_ids)])


def optimal_gain(s,j,e,c,budget,*,complex_control=False):
    s,j,e,c = [np.asarray(x,np.complex128) for x in (s,j,e,c)]
    eu = energy(s+e).clip(1e-24)
    a = np.mean(energy(c[:,3])/eu)
    q = np.mean(np.sum(c[:,3].conj()*j,-1)/eu)
    damage = np.mean(energy(c[:,3]-c[:,2])/energy(s).clip(1e-24))
    radius = min(2., np.sqrt(budget/damage)) if damage>0 else 2.
    if a<=1e-24:
        return 0.
    unconstrained = q/a if complex_control else max(0.,float(q.real/a))
    if abs(unconstrained)>radius:
        unconstrained = unconstrained*radius/abs(unconstrained)
    return unconstrained


def bootstrap_difference(a,b,seed=20261020,reps=3000):
    difference=np.asarray(a)-np.asarray(b)
    rng=np.random.default_rng(seed)
    means=difference[rng.integers(len(difference),size=(reps,len(difference)))].mean(1)
    lo,hi=np.quantile(means,[.025,.975])
    return dict(mean=float(difference.mean()),ci95=[float(lo),float(hi)],parents=len(difference))
