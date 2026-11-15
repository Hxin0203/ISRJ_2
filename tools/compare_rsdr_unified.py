"""Frozen, resumable, parent-paired RSDR-Net comparison on second-paper TEST.

Only this workspace stores new files. Original weights/source are read-only.
No source under paper2/ is changed and no old Gate arrays are accessed.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np

from paper2.io import ROOT, now, read, sha
from paper2.scan_1103 import build_batch, digest, measure
from paper2.evaluation_1103_v4 import states_from_components
from tools.durable_io import ProcessLock, atomic_json, atomic_write
from tools.resume_scan_1103 import verify_frozen_sources
from tools.rsdr_adapter import EXPECTED
from tools.rsdr_stream import make_backend, server_source

DEST = ROOT / "outputs/rsdr_unified_20261115"
LIGHT = ROOT / "evidence/outputs/RSDR_UNIFIED_20261115"
FREEZE = ROOT / "configs/rsdr_unified_20261115.json"
OLD_FREEZE = ROOT / "configs/paper2_1103_scan_freeze.json"
BACKENDS = ("local", "rsdr-b/cuda:0", "rsdr-b/cuda:1")
BLOCK = 16


def bindings():
    names = ["tools/rsdr_adapter.py", "tools/rsdr_stream.py", "tools/compare_rsdr_unified.py",
             "tools/durable_io.py"]
    names += [str(p.relative_to(ROOT)).replace('\\','/') for p in (ROOT / "tools/rsdr_vendor").glob('*') if p.is_file()]
    return {p: sha(ROOT / p) for p in sorted(names)}


def make_freeze():
    if FREEZE.exists():
        raise FileExistsError("Comparison freeze is immutable")
    old = read(OLD_FREEZE)
    verify_frozen_sources(old)
    parity = read(LIGHT / "ADAPTER_PARITY.json")
    extended = read(LIGHT / "EXTENDED_PARITY.json")
    assert parity['status'] == extended['status'] == 'PASS'
    plan = []
    # Ordering is fixed before new RSDR TEST inference, with no stopping by gain.
    for suite in ('main', 'coherent', 'retention', 'sliceA', 'sliceB', 'pd'):
        indexes = list(range(len(old['grids'][suite])))
        if suite == 'main':
            indexes.sort(key=lambda i:(old['grids'][suite][i]['jam_type'] != 'repeat',i))
        if suite == 'pd':
            indexes.sort(key=lambda i:(old['grids'][suite][i]['jsr_db'] != 20,i))
        for i in indexes:
            plan.append(dict(suite=suite,index=i,condition=old['grids'][suite][i],
                             backend=BACKENDS[len(plan) % len(BACKENDS)]))
    frozen = dict(created_utc=now(), stage="FROZEN_BEFORE_RSDR_TEST_INFERENCE",
        selection_timing="Second-paper TEST results already exist. No new model selection or tuning; additive benchmark, not prospective independent confirmation.",
        rsdr_identity=dict(name="first-paper complete F-R7 RSDR-Net", seed=40312102,
                           epoch=32, steps_per_trained_component=16384, assets=EXPECTED),
        paper2_identity={k:old[k] for k in ('seed','epoch','selected_checkpoint','checkpoint_sha256','selection_rule','selection_timing')},
        original_freeze_sha256=sha(OLD_FREEZE), sources=bindings(),
        engineering_parity_sha256=sha(LIGHT/'ADAPTER_PARITY.json'),
        extended_parity_sha256=sha(LIGHT/'EXTENDED_PARITY.json'),
        remote_stream_source_sha256={d:__import__('hashlib').sha256(server_source(d).encode()).hexdigest() for d in ('cuda:0','cuda:1')},
        plan=plan, parent_block=BLOCK, pulse_batch=256, bootstrap_reps=4000, bootstrap_seed=20261114,
        paired_statistics="Equal-cell means within physical parent, then paired parent bootstrap; separate retention jam groups are resampled within strata. No cell/pulse pseudoreplication.",
        primary_scope="56 repeat-jammer main-grid cells, 256 shared physical parents; u_nmse, target-difference damage, SJRIF and Pd, with actual H0 false alarms.",
        secondary_scope="Remaining full main, coherent, retention, width and dense Pd grids. Pointwise/descriptive intervals; no multiplicity-adjusted superiority claim.",
        endpoint_direction=dict(u_nmse='lower',damage='lower',sjrif_db='higher',pd='higher',no_jammer_pooled_nmse='lower',cell_pfa_noise='lower',cell_pfa_jammer='lower'),
        limitations=["One frozen training seed per model; parent uncertainty does not estimate training-seed uncertainty.",
                     "New model: 128*48=6144 updates; old:32*512=16384 updates per trained component plus multistage training. Not equal training budget.",
                     "Both models tested on the second-paper simulator distribution; old-model training distribution differs. Not a universal ranking.",
                     "Same nominal CA-CFAR Pfa=1e-6, not matched empirical Pfa. Pd must be read with observed H0 false alarms.",
                     "SJRIF is a target-bin/strongest-other-bin proxy after pulse compression, not an oracle physical jammer-energy ratio.",
                     "No TEST-dependent selection of epoch256 models, parameters, grid cells or example parents."])
    atomic_json(FREEZE, frozen)
    return frozen


def verify_freeze():
    f = read(FREEZE)
    assert f['sources'] == bindings(), "Comparison sources drifted after freeze"
    assert f['original_freeze_sha256'] == sha(OLD_FREEZE)
    verify_frozen_sources(read(OLD_FREEZE))
    assert sha(ROOT / f['paper2_identity']['selected_checkpoint']) == f['paper2_identity']['checkpoint_sha256']
    return f


def load_block(path, identity):
    receipt = read(path.with_suffix('.json'))
    assert receipt['identity_sha256'] == digest(identity)
    assert receipt['sha256'] == sha(path)
    with np.load(path, allow_pickle=False) as z:
        values = {k:z[k].copy() for k in z.files}
    return values, receipt


def run_condition(job, backend, frozen):
    suite,index,c = job['suite'],job['index'],job['condition']
    root = DEST / f'{suite}_{index:03d}' / 'rsdr'
    identity = dict(freeze_sha256=sha(FREEZE), **job)
    result_path = root / 'result.json'
    if result_path.exists():
        r = read(result_path)
        assert r['identity'] == identity and r['status'] == 'COMPLETE'
        for block in r['blocks']:
            assert sha(DEST / block['file']) == block['sha256']
        return r
    started = time.monotonic()
    parts, blocks = [], []
    for lo in range(0,c['count'],BLOCK):
        ids = list(range(lo,min(c['count'],lo+BLOCK)))
        block_identity = dict(**identity,parent_indices=ids)
        path = root / f'parents_{lo:04d}.npz'
        if path.exists() and path.with_suffix('.json').exists():
            values, receipt = load_block(path,block_identity)
        else:
            s,j,e,meta = build_batch('test',suite,c,ids)
            obs = states_from_components(s,j,e)
            # No-jammer observations are repeated exactly across the JSR and
            # jammer-type cells (except separately drawn retention groups).
            key = dict(role='test',suite=suite,snr_db=c['snr_db'],ids=ids,backend=job['backend'],
                       retention_jam=c['jam_type'] if suite=='retention' else None,
                       freeze_sha256=identity['freeze_sha256'])
            cache = DEST / 'no_jammer_cache' / (digest(key)+'.npy')
            if cache.exists() and cache.with_suffix('.json').exists():
                cr = read(cache.with_suffix('.json'))
                assert cr['identity'] == key and cr['sha256'] == sha(cache)
                h0 = np.load(cache,allow_pickle=False)
            else:
                h0 = backend.restore(obs,state_indexes=[0,1])
                atomic_write(cache,lambda stream:np.save(stream,h0,allow_pickle=False))
                atomic_json(cache.with_suffix('.json'),dict(identity=key,sha256=sha(cache)))
            h1 = h0 if suite=='retention' else backend.restore(obs,state_indexes=[2,3])
            u = np.concatenate((h0,h1),axis=1)
            values = measure(suite,s,j,e,obs,u,meta)
            assert all(len(v)==len(ids) and np.isfinite(v).all() for v in values.values())
            atomic_write(path,lambda stream:np.savez_compressed(stream,**values))
            receipt = dict(created_utc=now(),identity_sha256=digest(block_identity),sha256=sha(path),
                           metadata=meta,backend=job['backend'])
            atomic_json(path.with_suffix('.json'),receipt)
            atomic_json(DEST / f"PROGRESS_{job['backend'].replace('/','_').replace(':','_')}.json",
                        dict(updated_utc=now(),pid=os.getpid(),suite=suite,index=index,parents_finished=ids[-1]+1,
                             parents_total=c['count'],backend=job['backend']))
        parts.append(values)
        blocks.append(dict(file=str(path.relative_to(DEST)).replace('\\','/'),sha256=receipt['sha256']))
    values = {k:np.concatenate([p[k] for p in parts]) for k in parts[0]}
    result = dict(status='COMPLETE',scope='INDEPENDENT_TEST_ADDITIVE_COMPARISON',created_utc=now(),
                  identity=identity,blocks=blocks,elapsed_this_invocation_seconds=time.monotonic()-started,
                  metric_means={k:float(v.mean()) for k,v in values.items()})
    atomic_json(result_path,result)
    print(json.dumps(dict(completed=f'{suite}_{index:03d}',backend=job['backend'],seconds=result['elapsed_this_invocation_seconds'],
                          u_nmse=result['metric_means']['u_nmse'],sjrif_db=result['metric_means']['sjrif_db'],pd=result['metric_means']['pd'])),flush=True)
    return result


def worker(name, frozen):
    backend = make_backend(name)
    try:
        for job in frozen['plan']:
            if job['backend'] == name:
                run_condition(job,backend,frozen)
    finally:
        if hasattr(backend,'close'):backend.close()


def run():
    frozen = verify_freeze()
    with ProcessLock(DEST/'PROCESS.lock'):
        atomic_json(DEST/'INVOCATION.json',dict(started_utc=now(),pid=os.getpid(),freeze_sha256=sha(FREEZE)))
        try:
            with ThreadPoolExecutor(max_workers=len(BACKENDS)) as pool:
                futures = [pool.submit(worker,name,frozen) for name in BACKENDS]
                for future in futures:future.result()
            results = [read(DEST/f"{j['suite']}_{j['index']:03d}/rsdr/result.json") for j in frozen['plan']]
            assert len(results) == len(frozen['plan'])
            atomic_json(DEST/'COMPLETE.json',dict(status='COMPLETE',completed_utc=now(),freeze_sha256=sha(FREEZE),
                        conditions=len(results),parent_evaluations=sum(j['condition']['count'] for j in frozen['plan'])))
        except BaseException:
            atomic_json(DEST/f'FAILURE_{time.time_ns()}.json',dict(failed_utc=now(),traceback=traceback.format_exc()))
            raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','run']);args=p.parse_args()
    make_freeze() if args.action=='freeze' else run()
