"""Read-only saved-epoch/source audit, without evaluating or changing models."""
import json
import platform
import subprocess

from paper2.io import ROOT, now, read, sha
from tools.durable_io import atomic_json
from tools.idea_probe_20261118.run import require_server

RUNS = {
    'unet_mf_tail_screen_20261117_v2': 'tail',
    'unet_noise_norm_screen_20261117_v1': 'norm',
    'unet_frequency_screen_20261117_v1': 'factor',
    'unet_dilation_screen_20261117_v1': 'factor',
    'correction_original_control_20261117_v1': 'arm',
    'correction_mf_tail_20261117_v1': 'arm',
}


def run():
    admission = require_server()
    rows = []
    data_hashes = {role: sha(ROOT / f'data/paper2_1103/{role}.npz') for role in ('train', 'dev')}
    for name, arm in RUNS.items():
        base = ROOT / 'outputs' / name
        config = read(base / 'CONFIG.json')
        binding = read(base / 'QUEUE_BINDING.json') if (base / 'QUEUE_BINDING.json').exists() else config
        sources = binding.get('source_sha256', config.get('source_sha256', {}))
        seeds = []
        for seed in (17, 29, 41):
            folder = base / arm / f'seed_{seed}'
            receipts = [read(path) for path in (folder / 'recovery').glob('epoch_*.json')]
            latest = max(receipts, key=lambda item: item['epoch'])
            path = folder / 'recovery' / f"epoch_{latest['epoch']:03d}.pt"
            if not path.exists():
                candidates = [item for item in (folder / 'recovery').glob('epoch_*.json')
                              if read(item)['epoch'] == latest['epoch']]
                path = candidates[0].with_suffix('.pt')
            milestones = []
            for receipt in sorted(folder.glob('epoch_*.json')):
                value = read(receipt)
                milestones.append(dict(epoch=value['epoch'],
                    sha_matches=sha(receipt.with_suffix('.pt')) == value['checkpoint_sha256']))
            seeds.append(dict(seed=seed, saved_epoch=latest['epoch'], recovery_receipts=len(receipts),
                missing_recovery_epochs=sorted(set(range(latest['epoch'] + 1)) -
                                              {item['epoch'] for item in receipts}),
                latest_sha256=sha(path), latest_sha_matches=sha(path) == latest['checkpoint_sha256'],
                missing_milestones=sorted(set(range(32, latest['epoch'] + 1, 32)) -
                                          {item['epoch'] for item in milestones}),
                milestones=milestones, complete=read(folder / 'TRAINING_COMPLETE.json')))
        rows.append(dict(run=name, source_files=len(sources),
            source_drift=[path for path, digest in sources.items() if sha(ROOT / path) != digest],
            config_sha256=sha(base / 'CONFIG.json'), seeds=seeds,
            data_sha_matches={role: data_hashes[role] == config.get(
                'training_data_sha256' if role == 'train' else 'dev_data_sha256') for role in data_hashes},
            pipeline=read(base / 'PIPELINE_STATUS.json') if (base / 'PIPELINE_STATUS.json').exists() else None))
    snapshot = dict(checked_utc=now(), host=platform.node(), admission=admission,
        experiments=rows, data_sha256=data_hashes, source_sha256=sha(__file__),
        gpu_csv=subprocess.check_output(['nvidia-smi',
            '--query-gpu=index,uuid,utilization.gpu,memory.used,memory.total',
            '--format=csv,noheader,nounits'], text=True).strip(),
        scope='Latest and milestone checkpoint hashes; complete recovery receipt coverage, not every recovery binary')
    target = ROOT / 'evidence/outputs/idea_probe_20261118_v1/SERVER_STATUS.json'
    if target.exists():
        raise FileExistsError(target)
    atomic_json(target, snapshot)
    print(json.dumps(dict(checked_utc=snapshot['checked_utc'],
        runs=[dict(run=row['run'], epochs=[item['saved_epoch'] for item in row['seeds']],
                   source_drift=row['source_drift'], data_sha_matches=row['data_sha_matches']) for row in rows])))


if __name__ == '__main__':
    run()
