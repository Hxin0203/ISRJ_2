"""Prospective role generation for the single-jammer/two-target family."""
from dataclasses import asdict
import numpy as np
from paper2.vendor.generate_v2 import GeneratorConfig,generate_dataset,_draw_target
from paper2.data import iq
from paper2.io import ROOT,read,write,sha,now
from .family_science import eight_states

ROLES={'engineering':(8,2026102020,220100000),'train':(512,2026102021,220070000),
       'dev':(192,2026102022,220080000),'confirm':(192,2026102023,220090000)}


def load_family(role):
    n,seed,start=ROLES[role]
    directory=ROOT/'data'/'family';directory.mkdir(parents=True,exist_ok=True)
    raw_path=directory/f'{role}_simulator.npz'
    path=directory/f'{role}.npz';receipt=directory/f'{role}.receipt.json'
    cfg=GeneratorConfig(num_scenes=n,seed=seed,scene_id_start=start,fixed_partition='train' if role in ('train','engineering') else 'validation',
                        snr_min_db=-10,snr_max_db=5,jsr_min_db=0,jsr_max_db=25)
    if not path.exists():
        generate_dataset(raw_path,cfg)
        with np.load(raw_path,allow_pickle=False) as raw:
            a={k:raw[k] for k in ('s','j','epsilon','scene_id','target_amplitude','jam_type')}
        rng=np.random.default_rng(seed+1000)
        weak=np.zeros_like(a['s']);ratios=[]
        for parent in np.unique(a['scene_id']):
            members=np.flatnonzero(a['scene_id']==parent)
            target,_,meta=_draw_target(cfg,rng)
            ratio=float(rng.uniform(.15,.4));ratios.append(ratio)
            target*=ratio*float(a['target_amplitude'][members[0]])/meta['amplitude']
            weak[members]=target.astype(np.complex64)
        np.savez_compressed(path,**a,weak=weak)
        write(receipt,dict(created_utc=now(),role=role,parents=n,config=asdict(cfg),sha256=sha(path),
                           simulator_sha256=sha(raw_path),secondary_seed=seed+1000,amplitude_ratio_range=[.15,.4],
                           amplitude_ratio_min=float(min(ratios)),amplitude_ratio_max=float(max(ratios))))
    else:
        assert sha(path)==read(receipt)['sha256']
    with np.load(path,allow_pickle=False) as raw:
        a={k:raw[k] for k in raw.files}
    assert np.unique(a['scene_id']).size==n
    assert all(np.sum(a['scene_id']==p)==3 for p in np.unique(a['scene_id']))
    return a


def components(a):
    return [a[k].astype(np.complex128) for k in ('s','weak','j','epsilon')]


def training_tensors(a):
    strong,weak,j,e=components(a)
    y,_=eight_states(strong,weak,j,e)
    assert np.max(np.abs(y[:,7]-y[:,3]-strong))<1e-10
    assert np.max(np.abs(y[:,7]-y[:,5]-weak))<1e-10
    return dict(y=iq(y),a=iq(strong),b=iq(weak),j=iq(j),e=iq(e))
