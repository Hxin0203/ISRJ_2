"""Engineering timing and bounded six-run training, with completion receipts."""
import argparse
import hashlib
import os
import platform
import time
import traceback
import numpy as np
import torch
from .data import load,tensors
from .network import CorrectionNet,training_loss
from .io import ROOT,now,read,write,sha,source_bindings

OUT=ROOT/'outputs'


def setup(seed):
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def fingerprint(model):
    h=hashlib.sha256()
    for p in model.parameters():
        h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def batches(n,steps,seed):
    rng=np.random.default_rng(seed+10000)
    step=0
    while step<steps:
        order=rng.permutation(n)
        for start in range(0,n,32):
            if step>=steps:
                return
            yield order[start:start+32]
            step+=1


def update(model,optimizer,batch,method):
    optimizer.zero_grad(set_to_none=True)
    y=batch['y']
    correction=model(y.flatten(0,1)).reshape_as(y)
    losses=training_loss(correction,batch['s'],batch['j'],batch['e'],method)
    if not torch.isfinite(losses['total']):
        raise RuntimeError('nonfinite training loss')
    losses['total'].backward()
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True)
    optimizer.step()
    return {k:float(v.detach()) for k,v in losses.items()} | {'gradient_norm':float(norm)}


def engineering():
    receipt=OUT/'engineering.json'
    if receipt.exists():
        print('Engineering already complete; preserved.',flush=True)
        return
    device=setup(991)
    a=tensors(load('engineering'))
    model=CorrectionNet().to(device)
    opt=torch.optim.Adam(model.parameters(),lr=.001)
    batch={k:v[:16].to(device) for k,v in a.items()}
    torch.cuda.reset_peak_memory_stats() if device.type=='cuda' else None
    update(model,opt,batch,'B4')
    torch.cuda.synchronize() if device.type=='cuda' else None
    start=time.perf_counter()
    for _ in range(20):
        losses=update(model,opt,batch,'B4')
    torch.cuda.synchronize() if device.type=='cuda' else None
    sec=(time.perf_counter()-start)/20
    evidence=dict(completed_utc=now(),device=str(device),gpu=torch.cuda.get_device_name() if device.type=='cuda' else None,
                  torch=torch.__version__,python=platform.python_version(),parameters=sum(p.numel() for p in model.parameters()),
                  seconds_per_16_groups=sec,conservative_seconds_per_32_groups=sec*2.5,
                  peak_gpu_bytes=torch.cuda.max_memory_allocated() if device.type=='cuda' else 0,
                  updates=21,last_loss=losses,source=source_bindings())
    write(receipt,evidence)
    # Freeze before any TRAIN or DEV forward. Shorter training on slower hosts.
    steps=min(1200,max(100,int(600/(sec*2.5))))
    cfg=dict(created_utc=now(),steps=steps,seeds=[17,29],methods=['B1','B2','B4'],batch_groups=32,
             lr=.001,pc_weight=.1,response_weight=1.,budget=[.01,.05,.1],
             alpha_grid=list(np.linspace(0,2,81)),max_seconds_per_run=900,
             engineering_sha256=sha(receipt),selection='fixed final checkpoint, no DEV checkpoint selection')
    write(ROOT/'PILOT_FROZEN_CONFIG.json',cfg)
    print(evidence,flush=True)
    print('FROZEN',cfg,flush=True)


def train():
    cfg=read(ROOT/'PILOT_FROZEN_CONFIG.json')
    data=tensors(load('train'))
    # Materialize DEV for identity, but no model output is inspected here.
    load('dev')
    n=len(data['s'])
    source=source_bindings()
    config_sha=sha(ROOT/'PILOT_FROZEN_CONFIG.json')
    for seed in cfg['seeds']:
        for method in cfg['methods']:
            tag=f'{method}_seed{seed}'
            folder=OUT/tag
            complete=folder/'COMPLETE.json'
            if complete.exists():
                old=read(complete)
                assert old['source']==source and old['config_sha256']==config_sha
                assert old['checkpoint_sha256']==sha(folder/'model.pt')
                print('PRESERVED',tag,flush=True)
                continue
            if (folder/'STARTED.json').exists():
                raise RuntimeError(f'incomplete scientific run {tag}; inspect failure before any restart')
            device=setup(seed)
            model=CorrectionNet().to(device)
            opt=torch.optim.Adam(model.parameters(),lr=cfg['lr'])
            init=fingerprint(model)
            write(folder/'STARTED.json',dict(tag=tag,seed=seed,method=method,started_utc=now(),pid=os.getpid(),source=source,
                                           config_sha256=config_sha,initial_parameters_sha256=init))
            history=[]
            batch_hash=hashlib.sha256()
            start=time.perf_counter()
            exposure=0
            for step,idx in enumerate(batches(n,cfg['steps'],seed),1):
                batch_hash.update(idx.tobytes())
                batch={k:v[idx].to(device) for k,v in data.items()}
                losses=update(model,opt,batch,method)
                exposure+=len(idx)*4
                history.append(dict(step=step,**losses))
                if step==1 or step%50==0 or step==cfg['steps']:
                    status=dict(stage='TRAINING',tag=tag,step=step,steps=cfg['steps'],loss=losses,
                                elapsed_seconds=time.perf_counter()-start,updated_utc=now(),pid=os.getpid())
                    write(OUT/'STATUS.json',status)
                    print(status,flush=True)
                if time.perf_counter()-start>cfg['max_seconds_per_run']:
                    raise RuntimeError('predefined run time ceiling exceeded')
            torch.cuda.synchronize() if device.type=='cuda' else None
            elapsed=time.perf_counter()-start
            temporary=folder/'model.tmp.pt'
            torch.save(dict(state=model.state_dict(),method=method,seed=seed,config=cfg,source=source),temporary)
            temporary.replace(folder/'model.pt')
            write(folder/'history.json',history)
            result=dict(tag=tag,method=method,seed=seed,steps=cfg['steps'],completed_utc=now(),elapsed_seconds=elapsed,
                        exposure_state_observations=exposure,independent_train_parents=512,variant_records=n,
                        source=source,config_sha256=config_sha,initial_parameters_sha256=init,
                        batch_sequence_sha256=batch_hash.hexdigest(),checkpoint_sha256=sha(folder/'model.pt'),
                        final_loss=losses)
            write(complete,result)
    write(OUT/'TRAINING_COMPLETE.json',dict(completed_utc=now(),runs=6,config_sha256=config_sha,source=source))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['engineering','train'])
    args=parser.parse_args()
    OUT.mkdir(exist_ok=True)
    lock=OUT/'PROCESS.lock'
    try:
        with lock.open('x') as f:
            f.write(str(os.getpid()))
    except FileExistsError:
        raise RuntimeError('existing process lock: inspect before starting duplicate work')
    try:
        engineering() if args.stage=='engineering' else train()
    except BaseException as exc:
        write(OUT/'FAILURE.json',dict(time=now(),stage=args.stage,error=repr(exc),traceback=traceback.format_exc()))
        raise
    finally:
        lock.unlink()


if __name__=='__main__':
    main()
