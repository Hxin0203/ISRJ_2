"""Evaluate the 12 frozen checkpoints on DEV only, preserving v3 history."""
from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np
import torch

from .data_1103 import JAM_TYPES
from .evaluation_1103_v4 import (group_metrics, radar_metrics, restore_model,
                                 states_from_components, summarize)
from .io import ROOT, now, read, sha, write
from .network import CorrectionNet

BASE = ROOT / 'outputs/paper2_1103_retry_20261114'
EVIDENCE = ROOT / 'evidence/outputs/paper2_1103_retry_20261114'


def sources():
    names = ('dev_1103_v4.py', 'evaluation_1103_v4.py', 'radar_eval.py', 'network.py')
    return {f'paper2/{n}': sha(ROOT/'paper2'/n) for n in names}


def evaluate(seed, epoch):
    started = time.perf_counter()
    checkpoint = BASE/f'seed_{seed}'/f'epoch_{epoch:03d}.pt'
    cp_sha = sha(checkpoint)
    receipt = read(checkpoint.with_suffix('.json'))
    if receipt['checkpoint_sha256'] != cp_sha:
        raise RuntimeError('checkpoint SHA mismatch')
    data_path = ROOT/'data/paper2_1103/dev.npz'
    data_sha = sha(data_path)
    out = BASE/f'seed_{seed}'/f'dev_epoch_{epoch:03d}_v4.json'
    binding = dict(checkpoint_sha256=cp_sha, data_sha256=data_sha, evaluator_source_sha256=sources())
    if out.exists():
        saved = read(out)
        if any(saved[k] != v for k, v in binding.items()):
            raise RuntimeError(f'existing v4 identity changed; preserve old run: {out}')
        return saved
    torch.set_num_threads(4)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CorrectionNet().to(device).eval()
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    if payload['seed'] != seed or payload['epoch'] != epoch:
        raise RuntimeError('checkpoint identity mismatch')
    model.load_state_dict(payload['state'])
    with np.load(data_path, allow_pickle=False) as z:
        components = [z[k] for k in ('s', 'j', 'epsilon')]
        ids, fd = z['parent_id'].copy(), z['doppler_hz'].copy()
    n = len(ids)
    values = {method: {} for method in ('model', 'keep', 'reference')}
    for lo in range(0, n, 8):
        hi = min(lo+8, n)
        s, j, e = [x[lo:hi].reshape(-1, 8, 1800).astype(np.complex128) for x in components]
        obs = states_from_components(s, j, e)
        restored = restore_model(model, obs, device)
        reference = states_from_components(s, np.zeros_like(j), e)
        for method, u in (('model', restored), ('keep', obs), ('reference', reference)):
            metrics = group_metrics(s, j, e, u)
            metrics.update(radar_metrics(s, obs, u, np.repeat(fd[lo:hi], len(JAM_TYPES))))
            for name, v in metrics.items():
                values[method].setdefault(name, []).append(v.reshape(hi-lo, len(JAM_TYPES)))
        write(BASE/'DEV_V4_STATUS.json', dict(stage='DEV_V4', seed=seed, epoch=epoch,
              parents_done=hi, parents=n, updated_utc=now()))
    parent_values = {method: {k: np.concatenate(v).mean(axis=1) for k, v in d.items()}
                     for method, d in values.items()}
    raw = out.with_suffix('.npz')
    np.savez_compressed(raw, parent_id=ids, **{f'{method}__{k}': v
                        for method, d in parent_values.items() for k, v in d.items()})
    summaries = {method: {k: summarize(v) for k, v in d.items()} for method, d in parent_values.items()}
    paired = {k: summarize(parent_values['model'][k]-parent_values['keep'][k])
              for k in parent_values['model']}
    result = dict(stage='DEV_ONLY_V4', seed=seed, epoch=epoch, parents=n,
                  created_utc=now(), elapsed_seconds=time.perf_counter()-started,
                  device=str(device), **binding, metrics=summaries,
                  model_minus_keep=paired, local_parent_values_sha256=sha(raw),
                  metric_note='Energy ratios use full eight-pulse group; three jammer variants averaged within parent. '
                  'Doppler scoring is periodic. Pfa includes separate E and J+E states. '
                  'Intervals resample physical parents, not pulses or variants. DEV only.')
    write(out, result)
    write(EVIDENCE/f'seed_{seed}'/out.name, result)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seeds', nargs='+', type=int, choices=(17,29,41), default=[17,29,41])
    p.add_argument('--epochs', nargs='+', type=int, choices=(32,64,96,128), default=[32,64,96,128])
    args = p.parse_args()
    rows = []
    for seed in args.seeds:
        for epoch in args.epochs:
            r = evaluate(seed, epoch)
            rows.append(dict(seed=seed, epoch=epoch, checkpoint_sha256=r['checkpoint_sha256'],
                             metrics=r['metrics']['model']))
            print(f'completed DEV v4 seed={seed} epoch={epoch}', flush=True)
            write(EVIDENCE/'DEV_SUMMARY_v4.json', dict(created_utc=now(), rows=rows,
                  status='partial' if len(rows)<12 else 'complete', scope='DEV_ONLY'))
    write(BASE/'DEV_V4_STATUS.json', dict(stage='DEV_V4_COMPLETE', evaluated=len(rows), updated_utc=now()))


if __name__ == '__main__':
    main()
