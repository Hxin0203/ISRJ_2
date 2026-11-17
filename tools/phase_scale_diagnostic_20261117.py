"""Small server-CPU DEV diagnostic; no optimizer, checkpoint writes or tuning."""
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

from paper2.io import ROOT, now, read, sha
from tools.durable_io import atomic_json
from tools.unet_ablation.models import make_model
from tools.unet_ablation.protocol import CONFIG, DATA, ORIGINAL, BASE
from tools.unet_ablation.runtime import configure_precision

OUT = ROOT / 'evidence/outputs/phase_scale_diagnostic_20261117_v1'


def forward(net, z):
    v = np.stack((z.real, z.imag), axis=-2).astype(np.float32).reshape(-1, 2, 1800)
    with torch.inference_mode():
        chunks = [net(torch.from_numpy(v[lo:lo+32])).numpy() for lo in range(0, len(v), 32)]
    out = np.concatenate(chunks)
    return (out[:, 0] + 1j*out[:, 1]).reshape(z.shape)


def main():
    if platform.node().casefold() != 'desktop-rfferm3' or ROOT.resolve() != Path('D:/code/Xin/ISRJ_222').resolve():
        raise RuntimeError('Server B only, including diagnostic CPU inference')
    import psutil
    if psutil.cpu_percent(interval=1) >= 85 or psutil.virtual_memory().available < 8*2**30:
        raise RuntimeError('Insufficient CPU resources; preserve running GPU jobs')
    configure_precision()
    torch.set_num_threads(2)
    cfg = read(CONFIG)
    drift = [n for n, h in cfg['source_sha256'].items() if sha(ROOT/n) != h]
    if drift:
        raise RuntimeError(f'Baseline source drift: {drift}')
    data_sha = sha(DATA/'dev.npz')
    if read(DATA/'dev.receipt.json')['sha256'] != data_sha:
        raise RuntimeError('DEV receipt mismatch')
    if data_sha != cfg['data_sha256']['dev.npz']:
        raise RuntimeError('Frozen DEV changed')
    with np.load(DATA/'dev.npz', allow_pickle=False) as archive:
        # Fixed first 16 parents; no outcome-driven selection.
        s, j, e = [archive[k][:16].astype(np.complex64) for k in ('s','j','epsilon')]
        ids = archive['parent_id'][:16].tolist()
    y = np.stack((j+e, s+j+e), axis=2)
    reference = np.stack((e, s+e), axis=2)
    transforms = [('phase_pi4', np.exp(1j*np.pi/4)), ('phase_pi2', 1j),
                  ('phase_pi', -1), ('scale_0p1', .1), ('scale_10', 10)]
    binding = {'source_sha256': {n:sha(ROOT/n) for n in cfg['source_sha256']},
               'diagnostic_sha256': sha(Path(__file__)), 'dev_sha256': data_sha,
               'config_sha256': sha(CONFIG)}
    rows = []
    OUT.mkdir(parents=True, exist_ok=True)
    for variant in ('full','no_dechirp','unet'):
        for seed in (17,29,41):
            cp = (ORIGINAL if variant == 'full' else BASE/variant)/f'seed_{seed}'/'epoch_256.pt'
            digest = sha(cp)
            if read(cp.with_suffix('.json'))['checkpoint_sha256'] != digest:
                raise RuntimeError('Checkpoint receipt mismatch')
            payload = torch.load(cp, map_location='cpu', weights_only=False)
            if payload['epoch'] != 256 or payload['seed'] != seed:
                raise RuntimeError('Checkpoint seed/epoch mismatch')
            if variant == 'full':
                if digest != cfg['original_epoch256'][str(seed)]['checkpoint_sha256']:
                    raise RuntimeError('Frozen full checkpoint changed')
            elif (payload['identity']['config_sha256'] != sha(CONFIG)
                  or payload['identity']['variant'] != variant
                  or payload['identity']['data_sha256'] != cfg['data_sha256']['train.npz']):
                raise RuntimeError('Baseline checkpoint identity mismatch')
            path = OUT/f'{variant}_seed_{seed}.json'
            if path.exists():
                old=read(path)
                if old['binding'] != binding or old['checkpoint_sha256'] != digest:
                    raise RuntimeError('Preserve diagnostic with changed identity')
                rows.append(old)
                continue
            started=time.perf_counter()
            net = make_model(variant).eval()
            net.load_state_dict(payload['model'], strict=True)
            c = forward(net,y)
            # All transforms act on S,J,E jointly. Phase/positive scale lose no samples.
            noise_energy = np.sum(np.abs(e)**2,axis=(-2,-1))[:,:,None]
            c_energy = np.sum(np.abs(c)**2,axis=(-2,-1))
            denominator = np.maximum(c_energy,noise_energy*1e-12)
            baseline_error = np.sum(np.abs(y-c-reference)**2,axis=(-2,-1)) / np.maximum(np.sum(np.abs(j)**2,axis=(-2,-1))[:,:,None],1e-30)
            result=[]
            for name, g in transforms:
                transformed=forward(net,y*g)/g
                mismatch=np.sum(np.abs(transformed-c)**2,axis=(-2,-1))/denominator
                residual=np.sum(np.abs(y-transformed-reference)**2,axis=(-2,-1))/np.maximum(np.sum(np.abs(j)**2,axis=(-2,-1))[:,:,None],1e-30)
                assert np.isfinite(mismatch).all() and np.isfinite(residual).all()
                result.append({'transform':name,
                    'mean_equivariance_relative_squared_error':float(mismatch.mean()),
                    'p95_equivariance_relative_squared_error':float(np.quantile(mismatch,.95)),
                    'mean_endpoint_error_over_jammer_energy':float(residual.mean()),
                    'mean_endpoint_change':float((residual-baseline_error).mean())})
            row={'binding':binding,'variant':variant,'seed':seed,'epoch':256,
                 'checkpoint_sha256':digest,'parents':16,'states':['J+E','S+J+E'],
                 'baseline_endpoint_error_over_jammer_energy':float(baseline_error.mean()),
                 'transforms':result,'elapsed_seconds':time.perf_counter()-started,'updated_utc':now()}
            atomic_json(path,row)
            rows.append(row)
            atomic_json(OUT/'STATUS.json',{'stage':'DIAGNOSING','completed_model_seed_pairs':len(rows),'updated_utc':now()})
    summary={'stage':'DIAGNOSTIC_COMPLETE','rows':rows,'parents':16,'parent_ids_sha256':__import__('hashlib').sha256(json.dumps(ids).encode()).hexdigest(),
             'device':'server CPU; 2 threads','created_utc':now(),
             'limitations':'Fixed DEV subset; descriptive mapping consistency and endpoint changes only. No detection/correlation/causality claim. Zero correction can be equivariant without suppressing jamming. Delay/boundary and RSDR teacher not evaluated here.'}
    atomic_json(OUT/'SUMMARY.json',summary)
    print(json.dumps({'stage':summary['stage'],'model_seed_pairs':len(rows)},ensure_ascii=True))


if __name__ == '__main__':
    main()
