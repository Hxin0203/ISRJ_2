"""Read-only snapshots and server CPU engineering checks; never dispatch training."""
from __future__ import annotations
import argparse
import json
import os
import platform
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('engineering', 'snapshot'))
    parser.add_argument('--arm', choices=('original', 'tail'), default='original')
    args = parser.parse_args()
    os.environ['ISRJ2_CORRECTION_ARM'] = args.arm
    from paper2.io import ROOT, now, read, sha
    from tools.durable_io import atomic_json, process_alive
    from .settings import CONTROL_RUN, TAIL_RUN, DILATION_RUN, PAUSE_OWNER
    if platform.node().casefold() != 'desktop-rfferm3' or ROOT.resolve() != Path('D:/code/Xin/ISRJ_222').resolve():
        raise RuntimeError('Server B only, including engineering checks')
    if args.action == 'engineering':
        from .preflight import geometry_check
        from .train import local_sources, DATA
        geometry = geometry_check()
        data = {}
        for role in ('train', 'dev'):
            digest = sha(DATA / f'{role}.npz')
            assert digest == read(DATA / f'{role}.receipt.json')['sha256']
            data[role] = digest
        references = None
        if args.arm == 'original':
            from .reference import reference_binding
            references = {str(seed): reference_binding(seed, data['train']) for seed in (17, 29, 41)}
        result = {'stage': 'CPU_ENGINEERING_PASSED', 'arm': args.arm,
                  'geometry': geometry, 'source_sha256': local_sources(), 'data_sha256': data,
                  'reference_bindings': references, 'gpu_recovery_preflight': 'PENDING_DEVICE_RELEASE',
                  'created_utc': now()}
        atomic_json(ROOT / f'evidence/outputs/CORRECTIONNET_CPU_CHECK_{args.arm.upper()}_20261117.json', result)
    else:
        import statistics
        from .train import local_sources
        jobs = []
        for run in (CONTROL_RUN, TAIL_RUN):
            base = ROOT / 'outputs' / run
            binding = read(base/'QUEUE_BINDING.json') if (base/'QUEUE_BINDING.json').exists() else None
            pid = None
            if (base/'CONTROLLER.lock').exists():
                lock = json.loads((base/'CONTROLLER.lock').read_bytes()[1:])
                pid = lock['pid']
            seeds = []
            for seed in (17, 29, 41):
                folder = base/'arm'/f'seed_{seed}'
                receipts = sorted((folder/'recovery').glob('epoch_*.json'))
                latest = read(receipts[-1]) if receipts else None
                times = [p.with_suffix('.pt').stat().st_mtime for p in receipts[-21:]]
                intervals = [b-a for a,b in zip(times,times[1:]) if b>a]
                seeds.append({'seed': seed, 'saved_epoch': latest['epoch'] if latest else None,
                    'recovery_receipts': len(receipts),
                    'latest_sha_matches': sha(receipts[-1].with_suffix('.pt')) == latest['checkpoint_sha256'] if latest else None,
                    'complete': (folder/'TRAINING_COMPLETE.json').exists(),
                    'recent_wall_seconds_per_epoch': statistics.median(intervals) if intervals else None})
            jobs.append({'run': run, 'physical_gpu': 1, 'controller_pid': pid,
                'controller_alive': process_alive(pid),
                'status': read(base/'STATUS.json') if (base/'STATUS.json').exists() else {'stage': 'NOT_STARTED'},
                'queue_binding_sha256': sha(base/'QUEUE_BINDING.json') if binding else None,
                'source_drift': [p for p,h in binding['source_sha256'].items() if sha(ROOT/p)!=h] if binding else None,
                'seeds': seeds,
                'numerical_preflight_passed': (base/'NUMERICAL_PREFLIGHT.json').exists()})
        pause = ROOT/'outputs'/DILATION_RUN/'PAUSE'
        result = {'snapshot_utc': now(), 'jobs': jobs,
                  'dilation_pause_owned': pause.exists() and pause.read_text(encoding='utf8').strip() == PAUSE_OWNER,
                  'dilation_control_status': read(pause.parent/'CONTROL_STATUS.json') if (pause.parent/'CONTROL_STATUS.json').exists() else None,
                  'scope': 'Lightweight queue snapshot; no new efficacy or full recovery archive audit'}
        atomic_json(ROOT/'evidence/outputs/CORRECTIONNET_PRIORITY_STATUS_20261117.json', result)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
