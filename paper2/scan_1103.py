"""Resumable parent-paired 1103 scans; local arrays never enter evidence/."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import platform
import time
import traceback

import numpy as np
from scipy.stats import beta
import torch

from .baselines_1103 import run_bcs_oracle, run_gmm_omp, run_max_tf
from .evaluation_1103_v4 import (coherent_metrics, group_metrics, radar_metrics,
                                restore_model, states_from_components, summarize)
from .io import ROOT, now, read, sha, source_bindings, write
from .network import CorrectionNet
from .scenes_1103 import conditions, scene

TRAIN = ROOT/'outputs/paper2_1103_retry_20261114'
FREEZE = ROOT/'configs/paper2_1103_scan_freeze.json'
METHODS = {'main': ('model','keep','reference','max_tf','gmm_omp','bcs_oracle'),
           'pd': ('model','keep','reference'),
           'retention': ('model','keep','reference'),
           'coherent': ('model','keep','reference'),
           'sliceA': ('model','keep','reference','max_tf'),
           'sliceB': ('model','keep','reference','max_tf')}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def verify_checkpoints():
    complete = read(TRAIN/'TRAINING_COMPLETE.json')
    rows = []
    for run in complete['runs']:
        for epoch in (32,64,96,128):
            path = TRAIN/f"seed_{run['seed']}"/f'epoch_{epoch:03d}.pt'
            receipt = read(path.with_suffix('.json'))
            actual = sha(path)
            assert actual == run['checkpoints'][str(epoch)] == receipt['checkpoint_sha256']
            assert receipt['dataset_sha256'] == complete['data_sha256']
            for name, expected in receipt['source_sha256'].items():
                assert sha(ROOT/name) == expected, name
            dev = read(path.parent/f'dev_epoch_{epoch:03d}_v4.json')
            assert dev['checkpoint_sha256'] == actual
            assert dev['data_sha256'] == sha(ROOT/'data/paper2_1103/dev.npz')
            for name, expected in dev['evaluator_source_sha256'].items():
                assert sha(ROOT/name) == expected, name
            rows.append(dict(seed=run['seed'], epoch=epoch,
                             checkpoint=str(path.relative_to(ROOT)).replace('\\','/'),
                             sha256=actual, dev=dev['metrics']['model']))
    return complete, rows


def freeze():
    if FREEZE.exists():
        raise RuntimeError('freeze exists; never overwrite the pre-TEST identity')
    complete, rows = verify_checkpoints()
    aggregate = {str(ep): {k: float(np.mean([r['dev'][k]['mean'] for r in rows if r['epoch']==ep]))
                          for k in rows[0]['dev']} for ep in (32,64,96,128)}
    epoch = min((32,64,96,128), key=lambda ep: aggregate[str(ep)]['u_nmse'])
    selected = next(r for r in rows if r['seed']==29 and r['epoch']==epoch)
    value = dict(created_utc=now(), stage='FROZEN_BEFORE_INDEPENDENT_TEST',
                 selection_timing='Rule finalized AFTER all DEV v4 results, BEFORE TEST generation; not preregistered before DEV.',
                 selection_rule='Minimum three-seed mean DEV group u_nmse selects epoch; fixed middle listed seed 29, not best seed.',
                 selected_checkpoint=selected['checkpoint'], checkpoint_sha256=selected['sha256'],
                 seed=29, epoch=epoch, all_checkpoints=rows, dev_epoch_mean=aggregate,
                 train_sha256=complete['data_sha256'],
                 sources=source_bindings(),
                 grids={suite: conditions(suite) for suite in METHODS}, methods=METHODS,
                 bootstrap_reps=4000, bootstrap_seed=20261114,
                 inference_batch_pulses=192, parent_block_size=8,
                 typical_parent=0, coherent_fft_points=4096,
                 limitations=['48 updates/epoch, versus 512 in 1103; no claim of identical training budget.',
                              'Baselines are local formula reproductions, no TEST tuning; BCS uses oracle support.',
                              'Nominal CFAR Pfa=1e-6 is not an equal-empirical-Pfa comparison.',
                              'TEST scenes are independent of TRAIN/DEV; grid cells reuse physical parents.',
                              'All seeds/checkpoints and negative conditions are retained; no ablations.'])
    write(FREEZE, value)
    write(ROOT/'evidence/outputs/paper2_1103_retry_20261114/CHECKPOINT_VERIFICATION.json',
          dict(verified_utc=now(), checkpoints=[{k:v for k,v in r.items() if k!='dev'} for r in rows],
               training_complete_sha256=sha(TRAIN/'TRAINING_COMPLETE.json')))
    return value


def binomial_interval(k, n):
    """Exact pointwise Clopper-Pearson interval for independent parent events."""
    return [0. if k==0 else float(beta.ppf(.025,k,n-k+1)),
            1. if k==n else float(beta.ppf(.975,k+1,n-k))]


def metric_summary(name, values):
    result = summarize(values)
    n = len(values)
    if name in ('pd','pd_no_jammer','frame_pfa_noise','frame_pfa_jammer'):
        k = int(np.sum(values))
        assert np.isin(values, [0.,1.]).all()
        result.update(events=k, trials=n, binomial_cp_ci95=binomial_interval(k,n))
    if name.startswith('cell_pfa_'):
        k = int(np.rint(np.sum(values)*9032))
        result.update(false_cells=k, valid_cuts=n*9032,
                      interval_note='Parent-cluster bootstrap; CUTs are correlated, not independent Bernoulli trials.')
        if k==0:
            result['zero_event_conservative_upper95'] = binomial_interval(0,n)[1]
            result['zero_event_note'] = 'Zero bootstrap interval is not a population guarantee. Cell probability <= frame probability; upper bound uses independent parents.'
    return result


def build_batch(role, suite, condition, ids):
    args = {k:v for k,v in condition.items() if k!='count'}
    scenes = [scene(role, suite, p, **args) for p in ids]
    s,j,e = [np.stack([v[k] for v in scenes]) for k in range(3)]
    if suite=='retention':
        j *= 0
    return s,j,e,[v[3] for v in scenes]


def measure(suite, s,j,e,obs,u,meta):
    if not np.isfinite(u).all() or u.shape != obs.shape:
        raise ValueError('nonfinite or incorrectly shaped restored waveform')
    result = group_metrics(s,j,e,u)
    result.update(radar_metrics(s,obs,u,np.asarray([v['doppler_hz'] for v in meta])))
    if suite=='coherent':
        result.update(coherent_metrics(s,u[:,3],fft_points=4096))
    return result


def process_batch(role, suite, condition, ids, method, model=None, device='cpu', controls=True):
    s,j,e,meta = build_batch(role,suite,condition,ids)
    obs = states_from_components(s,j,e)
    reference = states_from_components(s,np.zeros_like(j),e)
    diagnostics = {}
    if method=='model':
        u = restore_model(model,obs,device)
    elif method=='keep':
        u = obs
    elif method=='reference':
        u = reference
    elif method in ('max_tf','gmm_omp'):
        adapter = run_max_tf if method=='max_tf' else run_gmm_omp
        # Each parent is isolated: EM stopping in one parent cannot depend on
        # another parent's convergence or the scheduling/batch size.
        u = np.stack([adapter(x.reshape(32,1800)).reshape(4,8,1800) for x in obs])
    elif method=='bcs_oracle':
        # States E and S+E have ZERO jammer; only J+E and S+J+E
        # receive the actual jammer support.
        support = states_from_components(np.zeros_like(s),j,np.zeros_like(e))
        pairs = [run_bcs_oracle(x.reshape(32,1800), jm.reshape(32,1800))
                 for x,jm in zip(obs,support)]
        u = np.stack([x[0].reshape(4,8,1800) for x in pairs])
        diagnostics = dict(bcs_converged_pulses=sum(x[1]['converged_pulses'] for x in pairs),
                           bcs_total_pulses=sum(x[1]['pulse_count'] for x in pairs))
    else:
        raise ValueError(method)
    values = measure(suite,s,j,e,obs,u,meta)
    if not controls:
        return values,meta,diagnostics
    keep = values if method=='keep' else measure(suite,s,j,e,obs,obs,meta)
    ref = values if method=='reference' else measure(suite,s,j,e,obs,reference,meta)
    keep_delta = {f'paired_keep__{k}': values[k]-v for k,v in keep.items()}
    ref_delta = {f'paired_reference__{k}': values[k]-v for k,v in ref.items()}
    values.update(keep_delta)
    values.update(ref_delta)
    return values,meta,diagnostics


def worker_init():
    torch.set_num_threads(1)


def cpu_parent(args):
    role,suite,condition,parent,method = args
    return process_batch(role,suite,condition,[parent],method,controls=False)


def combine(parts):
    values = {k: np.concatenate([p[0][k] for p in parts]) for k in parts[0][0]}
    meta = [m for p in parts for m in p[1]]
    keys = set().union(*(p[2] for p in parts))
    diagnostics = {k:sum(p[2].get(k,0) for p in parts) for k in keys}
    return values,meta,diagnostics


def condition_key(suite, index):
    return f'{suite}_{index:03d}'


def evaluate_condition(root,evidence,binding,role,suite,index,condition,method,model,device,pool):
    started = time.perf_counter()
    folder = root/condition_key(suite,index)/method
    out = folder/'result.json'
    identity = dict(binding=binding,role=role,suite=suite,index=index,condition=condition,method=method)
    identity_sha = digest(identity)
    if out.exists():
        saved = read(out)
        if saved['identity_sha256']!=identity_sha:
            raise RuntimeError(f'existing condition identity mismatch: {out}')
        for block in saved['blocks']:
            if sha(root/block['file'])!=block['sha256']:
                raise RuntimeError('saved parent block SHA mismatch')
        return saved
    blocks, collected, allmeta, diagnostic = [], [], [], {}
    for lo in range(0,condition['count'],8):
        ids = list(range(lo,min(lo+8,condition['count'])))
        path = folder/f'parents_{lo:04d}.npz'
        receipt = path.with_suffix('.json')
        block_identity = digest(dict(identity_sha256=identity_sha,parents=ids))
        if receipt.exists():
            br = read(receipt)
            if br['identity_sha256']!=block_identity or sha(path)!=br['sha256']:
                raise RuntimeError('existing parent block identity or hash mismatch')
            with np.load(path,allow_pickle=False) as z:
                values = {k:z[k].copy() for k in z.files}
            meta,diag = br['metadata'],br['diagnostics']
        else:
            if path.exists():
                # Interrupted write with no receipt: preserve, never silently overwrite.
                path.rename(path.with_suffix(f'.unreceipted_{time.time_ns()}.npz'))
            if pool is not None and method in ('max_tf','gmm_omp','bcs_oracle'):
                values,meta,diag = combine(list(pool.map(cpu_parent,
                    [(role,suite,condition,p,method) for p in ids])))
            else:
                values,meta,diag = process_batch(role,suite,condition,ids,method,model,device,controls=False)
            # Controls are deterministic metrics on the same parent IDs. Store
            # them once per block and reuse across methods, without IQ caching.
            controls = {}
            for control in ('keep','reference'):
                control_path = root/condition_key(suite,index)/'_controls'/control/f'parents_{lo:04d}.npz'
                control_receipt = control_path.with_suffix('.json')
                control_identity = digest(dict(binding=binding,role=role,suite=suite,
                    condition=condition,parents=ids,method=control))
                if control_receipt.exists():
                    cr = read(control_receipt)
                    if cr['identity_sha256']!=control_identity or sha(control_path)!=cr['sha256']:
                        raise RuntimeError('control cache identity mismatch')
                    with np.load(control_path,allow_pickle=False) as z:
                        controls[control]={k:z[k].copy() for k in z.files}
                else:
                    if method==control:
                        cv=values
                    else:
                        cv,_,_=process_batch(role,suite,condition,ids,control,controls=False)
                    control_path.parent.mkdir(parents=True,exist_ok=True)
                    np.savez_compressed(control_path,**cv)
                    write(control_receipt,dict(identity_sha256=control_identity,sha256=sha(control_path)))
                    controls[control]=cv
            deltas={f'paired_{control}__{k}':values[k]-v
                    for control,cv in controls.items() for k,v in cv.items()}
            values.update(deltas)
            path.parent.mkdir(parents=True,exist_ok=True)
            with path.with_suffix('.tmp').open('wb') as f:
                np.savez_compressed(f,**values)
            path.with_suffix('.tmp').replace(path)
            write(receipt,dict(identity_sha256=block_identity,sha256=sha(path),
                               metadata=meta,diagnostics=diag,created_utc=now()))
        blocks.append(dict(file=str(path.relative_to(root)).replace('\\','/'),sha256=sha(path)))
        collected.append(values)
        allmeta.extend(meta)
        for k,v in diag.items():
            diagnostic[k]=diagnostic.get(k,0)+v
        write(root/'STATUS.json',dict(stage='RUNNING',suite=suite,index=index,method=method,
              parents_done=ids[-1]+1,parents=condition['count'],pid=os.getpid(),updated_utc=now()))
    values = {k:np.concatenate([p[k] for p in collected]) for k in collected[0]}
    metrics = {k:metric_summary(k,v) for k,v in values.items() if not k.startswith('paired_')}
    paired = {k:summarize(v) for k,v in values.items() if k.startswith('paired_')}
    qa = {k:summarize([m[k] for m in allmeta]) for k in
          ('total_jsr_db','target_coverage_all','target_coverage_selected')}
    result = dict(created_utc=now(),status='COMPLETE',identity=identity,identity_sha256=identity_sha,
                  metrics=metrics,paired_differences=paired,scene_qa=qa,diagnostics=diagnostic,
                  parent_metadata_sha256=digest(allmeta),blocks=blocks,
                  elapsed_seconds_this_invocation=time.perf_counter()-started,
                  scope='INDEPENDENT_TEST' if role=='test' else 'ENGINEERING_ONLY',
                  interval_note='Pointwise intervals; physical parents reused across conditions. No independence across grid cells assumed.')
    write(out,result)
    light = {k:v for k,v in result.items() if k not in ('blocks','identity')}
    light['identity']={k:v for k,v in identity.items() if k!='binding'}
    light['frozen_binding_sha256']=digest(binding)
    write(evidence/condition_key(suite,index)/f'{method}.json',light)
    print(f'{role} {suite} {index+1}/{len(conditions(suite))} {method} complete in '
          f'{result["elapsed_seconds_this_invocation"]:.1f}s',flush=True)
    return result


def run(args):
    torch.set_num_threads(2)
    current_sources=source_bindings()
    if args.role=='test':
        frozen=read(FREEZE)
        if frozen['sources']!=current_sources:
            raise RuntimeError('scientific sources changed after freeze; do not reuse TEST identity')
        if args.count is not None:
            raise ValueError('TEST sample size cannot be overridden')
        checkpoint=ROOT/frozen['selected_checkpoint']
        if sha(checkpoint)!=frozen['checkpoint_sha256']:
            raise RuntimeError('frozen model SHA mismatch')
        binding=dict(freeze_sha256=sha(FREEZE),sources_sha256=digest(current_sources),
                     checkpoint_sha256=frozen['checkpoint_sha256'])
    else:
        checkpoint=TRAIN/'seed_29/epoch_128.pt'
        binding=dict(sources_sha256=digest(current_sources),checkpoint_sha256=sha(checkpoint))
    root=ROOT/'outputs'/args.run
    evidence=ROOT/'evidence/outputs'/args.run
    model=None
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.methods is None or 'model' in args.methods:
        model=CorrectionNet().to(device).eval()
        model.load_state_dict(torch.load(checkpoint,map_location=device,weights_only=False)['state'])
    pool=ProcessPoolExecutor(max_workers=args.workers,initializer=worker_init) if args.workers>1 else None
    write(root/'INVOCATION.json',dict(args=vars(args),binding=binding,started_utc=now(),
          python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,pid=os.getpid()))
    try:
        for suite in args.suites:
            grid=conditions(suite)
            indexes=range(len(grid)) if args.limit is None else range(min(args.limit,len(grid)))
            for index in indexes:
                condition=dict(grid[index])
                if args.count is not None:
                    condition['count']=args.count
                for method in METHODS[suite]:
                    if args.methods is None or method in args.methods:
                        evaluate_condition(root,evidence,binding,args.role,suite,index,condition,
                                           method,model,device,pool)
        write(root/'STATUS.json',dict(stage='INVOCATION_COMPLETE',suites=args.suites,
              methods=args.methods,limit=args.limit,role=args.role,updated_utc=now(),pid=os.getpid()))
    except BaseException:
        write(root/f'FAILURE_{time.time_ns()}.json',dict(created_utc=now(),args=vars(args),traceback=traceback.format_exc()))
        raise
    finally:
        if pool is not None:
            pool.shutdown()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=('freeze','run'))
    parser.add_argument('--role',choices=('engineering','test'),default='engineering')
    parser.add_argument('--run',default='paper2_1103_engineering_scan_20261114')
    parser.add_argument('--suites',nargs='+',choices=tuple(METHODS),default=list(METHODS))
    parser.add_argument('--methods',nargs='+',choices=METHODS['main'])
    parser.add_argument('--workers',type=int,default=1)
    parser.add_argument('--limit',type=int)
    parser.add_argument('--count',type=int)
    args=parser.parse_args()
    if args.action=='freeze':
        result=freeze()
        print(f"Frozen seed {result['seed']}, epoch {result['epoch']}; {result['checkpoint_sha256']}",flush=True)
    else:
        run(args)


if __name__=='__main__':
    main()
