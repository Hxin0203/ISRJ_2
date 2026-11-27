"""Read-only fixed-DEV review: five frozen Hybrid seeds and frozen first-paper RSDR.

Run on server B only. No fitting, checkpoint selection, or confirmation-set access.
Raw parent/variant arrays remain on the server. Each worker has GPU admission.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path("D:/code/Xin/ISRJ_222")
OUT = ROOT / "outputs/hybrid_five_seed_review_20261127"
SEEDS = (17, 29, 41, 53, 67)
DEV = ROOT / "data/paper2_1103/dev.npz"
DEV_SHA = "0e71bf9ddb43a0a89f62ce3e09e741d9ea20c19596ab75aebc787cd1e7796dc1"


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)


def worker(label):
    from tools.server_gpu_admission import require_server_identity, require_visible_gpu, NoGpuAvailable
    require_server_identity()
    try:
        gpu = require_visible_gpu()
    except NoGpuAvailable:
        return 75
    import numpy as np
    from tools.paper2_baseline_suite import equal_pfa_suite as eq
    from paper2.evaluation_1103_v4 import (states_from_components, group_metrics,
                                          radar_metrics, coherent_metrics)
    eq.configure_precision()
    if hashlib.sha256(DEV.read_bytes()).hexdigest() != DEV_SHA:
        raise RuntimeError('DEV SHA mismatch')
    if label == 'rsdr':
        from tools.paper2_baseline_suite import rsdr_suite as rs
        net, binding = rs.load()
        restore = lambda x: {'main': rs.restore(net, x)}
    else:
        seed = int(label)
        if seed not in SEEDS:
            raise ValueError(seed)
        net, binding = eq.load('ReCoVP-Hybrid', seed)
        restore = lambda x: eq.restore(net, 'hybrid', x)
    with np.load(DEV, allow_pickle=False) as z:
        s,j,e = (z[k] for k in ('s','j','epsilon'))
        pid,fd = z['parent_id'].copy(),z['doppler_hz'].copy()
    parts = {}
    t0 = time.perf_counter()
    for lo in range(0,len(s),8):
        hi=min(lo+8,len(s)); n=hi-lo
        ss,jj,ee=(x[lo:hi].reshape(n*3,8,1800).astype(np.complex128) for x in (s,j,e))
        original=states_from_components(ss,jj,ee)
        for output,u in restore(states_from_components(s[lo:hi],j[lo:hi],e[lo:hi])).items():
            uu=np.moveaxis(u,2,1).reshape(n*3,4,8,1800).astype(np.complex128)
            metrics=group_metrics(ss,jj,ee,uu)
            metrics.update(radar_metrics(ss,original,uu,np.repeat(fd[lo:hi],3)))
            for kind,wave in [('operational',uu[:,3]),('increment',uu[:,3]-uu[:,2])]:
                metrics.update({kind+'_'+k:v for k,v in coherent_metrics(ss,wave).items()})
            for k,v in metrics.items():
                parts.setdefault(output+'__'+k,[]).append(np.asarray(v).reshape(n,3))
        print(label,lo,round(time.perf_counter()-t0,1),flush=True)
    arrays={k:np.concatenate(v) for k,v in parts.items()}
    folder=OUT/label;folder.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(folder/'PARENT_VARIANT_VALUES.npz',parent_id=pid,**arrays)
    write(folder/'SUMMARY.json',{
        'label':label,'seed':None if label=='rsdr' else int(label),'epoch':32 if label=='rsdr' else 256,
        'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'dev_sha256':DEV_SHA,
        'binding':binding,'gpu_admission':gpu,'parents':len(pid),'jammer_variants':['direct','repeat','cyclic'],
        'means':{k:float(v.mean()) for k,v in arrays.items()},
        'variant_means':{k:dict(zip(['direct','repeat','cyclic'],v.mean(axis=0).tolist())) for k,v in arrays.items()},
        'seconds':time.perf_counter()-t0,'fft_points':4096,
        'protocol':'fixed DEV; operational=u(S+J+E), increment=u(S+J+E)-u(J+E); coherent metrics relative to target-only pulse compression; FFT zero padding is interpolation, not improved physical resolution',
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    return 0


def queue():
    from tools.server_gpu_admission import require_server_identity,choose_gpu,NoGpuAvailable
    require_server_identity();OUT.mkdir(parents=True,exist_ok=True)
    labels=[str(s) for s in SEEDS]+['rsdr']
    for label in labels:
        if (OUT/label/'SUMMARY.json').exists():continue
        while True:
            try: gpu=choose_gpu()
            except NoGpuAvailable:
                write(OUT/'STATUS.json',{'state':'WAIT_GPU','label':label,'updated_utc':time.time()})
                time.sleep(15);continue
            write(OUT/'STATUS.json',{'state':'INFERENCE','label':label,'gpu':gpu,'updated_utc':time.time()})
            env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu['physical_gpu_index'])}
            with (OUT/(label+'.log')).open('a',encoding='utf-8') as log:
                rc=subprocess.call([sys.executable,'-m','tools.hybrid_five_seed_review_inference','--worker',label],
                                   cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            if rc==75:time.sleep(5);continue
            if rc:raise RuntimeError(f'{label}: worker exited {rc}')
            break
    write(OUT/'STATUS.json',{'state':'COMPLETE','labels':labels,'updated_utc':time.time()})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worker');args=p.parse_args()
    if args.worker:sys.exit(worker(args.worker))
    queue()
