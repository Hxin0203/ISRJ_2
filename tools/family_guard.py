"""Pre-DEV validation without modifying the frozen family implementation."""
import numpy as np
import torch
from paper2.io import ROOT,read,write,sha,now
from .family_study import bindings,freeze,evaluate


def audit_training():
    cfg=freeze();frozen_sha=sha(ROOT/'FAMILY_STUDY_FREEZE.json')
    train_path=ROOT/'data'/'family'/'train.npz';train_sha=sha(train_path)
    assert read(ROOT/'data'/'family'/'train.receipt.json')['sha256']==train_sha
    for seed in (17,29):
        identical=[]
        for method in ('B1','B2','B4'):
            folder=ROOT/'outputs'/'family'/f'{method}_seed{seed}'
            start=read(folder/'STARTED.json');end=read(folder/'COMPLETE.json')
            assert start['source']==end['source']==bindings()
            assert start['dataset_sha256']==end['dataset_sha256']==train_sha
            assert start['freeze_sha256']==end['freeze_sha256']==frozen_sha
            assert end['steps']==1200 and end['exposure']==153600
            assert sha(folder/'model.pt')==end['checkpoint_sha256']
            payload=torch.load(folder/'model.pt',map_location='cpu',weights_only=False)
            assert payload['source']==bindings() and payload['dataset_sha256']==train_sha
            assert payload['freeze_sha256']==frozen_sha
            identical.append(tuple(end[k] for k in ('initial_parameters_sha256','batch_sequence_sha256','exposure')))
        assert len(set(identical))==1
    all_roles={}
    for p in (ROOT/'data').glob('*.npz'):
        if p.name=='dev_confirmation_weak_targets.npz':continue
        with np.load(p,allow_pickle=False) as raw:
            if 'scene_id' in raw:
                all_roles[p.name]=set(raw['scene_id'].tolist())
    with np.load(train_path,allow_pickle=False) as raw:
        ids=raw['scene_id'];current=set(ids.tolist())
        cached={k:raw[k] for k in ('s','weak','epsilon')}
        assert not any(220040000<=x<220050000 for x in current)
        for old in all_roles.values():assert current.isdisjoint(old)
        for p in np.unique(ids):
            members=ids==p
            for k in ('s','weak','epsilon'):
                value=cached[k][members]
                assert np.array_equal(value,np.broadcast_to(value[0],value.shape))
    return cfg


def main():
    cfg=audit_training()
    assert not list((ROOT/'data'/'family').glob('confirm*')),'confirmation assets already exist'
    write(ROOT/'evidence'/'family'/'BEFORE_DEV_AUDIT.json',dict(checked_utc=now(),all_checks_pass=True,
          source=bindings(),train_sha256=sha(ROOT/'data'/'family'/'train.npz'),
          per_seed_initialization_batches_exposure_equal=True,confirmation_assets_absent=True,
          original_final_test_namespace_unused=True,guard_sha256=sha(__file__)))
    evaluate(cfg)
    audit_training()
    gain=read(ROOT/'evidence'/'family'/'DEV_GAIN_FREEZE.json')
    confirm=read(ROOT/'data'/'family'/'confirm.receipt.json')
    assert gain['created_utc']<confirm['created_utc']
    assert gain['source']==bindings()
    # Confirm role grouping and disjoint namespaces after generation.
    seen=[]
    for role in ('engineering','train','dev','confirm'):
        path=ROOT/'data'/'family'/f'{role}.npz'
        assert sha(path)==read(path.with_suffix('.receipt.json'))['sha256']
        with np.load(path,allow_pickle=False) as raw:
            ids=raw['scene_id'];current=set(ids.tolist())
            cached={k:raw[k] for k in ('s','weak','epsilon')}
            assert all(current.isdisjoint(old) for old in seen)
            assert all(current.isdisjoint(old) for old in _old_namespaces())
            assert not any(220040000<=x<220050000 for x in current)
            for p in np.unique(ids):
                members=ids==p
                for k in ('s','weak','epsilon'):
                    v=cached[k][members]
                    assert np.array_equal(v,np.broadcast_to(v[0],v.shape))
            seen.append(current)
    write(ROOT/'evidence'/'family'/'FINAL_AUDIT.json',dict(completed_utc=now(),all_checks_pass=True,
          source=bindings(),gain_freeze_sha256=sha(ROOT/'evidence'/'family'/'DEV_GAIN_FREEZE.json'),
          results_sha256=sha(ROOT/'evidence'/'family'/'RESULTS.json'),
          role_namespaces_disjoint=True,parent_components_shared=True,confirmation_generated_after_gain_freeze=True,
          scientific_runs=6,final_test_opened=False))
    print('Family pre-DEV and final provenance audits passed.',flush=True)


def _old_namespaces():
    for p in (ROOT/'data').glob('*.npz'):
        with np.load(p,allow_pickle=False) as raw:
            if 'scene_id' in raw:yield set(raw['scene_id'].tolist())


if __name__=='__main__':main()
