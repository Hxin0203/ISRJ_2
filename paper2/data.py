"""New-role simulator calls; legacy data directories are never opened."""
from dataclasses import asdict
import numpy as np
import torch
from .vendor.generate_v2 import GeneratorConfig, generate_dataset
from .science import four_states
from .io import ROOT,read,write,sha,now

ROLES=dict(engineering=(16,2026102001,220010000),train=(512,2026102002,220020000),dev=(192,2026102003,220030000))


def generate(role):
    n,seed,start=ROLES[role]
    path=ROOT/'data'/f'{role}.npz'
    receipt=ROOT/'data'/f'{role}.receipt.json'
    if path.exists():
        saved=read(receipt)
        if saved['sha256']!=sha(path):
            raise RuntimeError('role identity drift')
        return path
    cfg=GeneratorConfig(num_scenes=n,seed=seed,scene_id_start=start,
                        fixed_partition='train' if role in ('engineering','train') else 'validation',
                        snr_min_db=-10,snr_max_db=5,jsr_min_db=0,jsr_max_db=25)
    generate_dataset(path,cfg)
    with np.load(path,allow_pickle=False) as raw:
        s,j,e=[raw[k].astype(np.complex128) for k in ('s','j','epsilon')]
        y,c=four_states(s,j,e)
        ids=raw['scene_id']
        assert np.unique(ids).size==n
        assert all(np.sum(ids==p)==3 for p in np.unique(ids))
        for p in np.unique(ids):
            members=ids==p
            assert np.array_equal(s[members],np.broadcast_to(s[members][0],s[members].shape))
            assert np.array_equal(e[members],np.broadcast_to(e[members][0],e[members].shape))
        error=float(np.max(np.abs(y[:,3]-y[:,2]-s)))
        assert error<1e-10
        assert all(np.isfinite(a).all() for a in (s,j,e))
    write(receipt,dict(role=role,created_utc=now(),parents=n,variants=3,states=4,
                       independent_unit='physical parent',sha256=sha(path),config=asdict(cfg),
                       four_state_identity_max=error,shared_parent_s_e=True))
    return path


def iq(x):
    return torch.from_numpy(np.stack((x.real,x.imag),axis=-2).astype(np.float32))


def load(role):
    path=generate(role)
    with np.load(path,allow_pickle=False) as raw:
        a={k:raw[k] for k in ('s','j','epsilon','scene_id','jam_type','snr_db','jsr_db')}
    return a


def tensors(a):
    # States are assembled in complex128 first, then rounded once to float32.
    s,j,e=[a[k].astype(np.complex128) for k in ('s','j','epsilon')]
    y,_=four_states(s,j,e)
    return dict(y=iq(y),s=iq(s),j=iq(j),e=iq(e))
