"""Read-only CPU checkpoint audit and server-side backup after user pause."""
import os
import shutil
import socket
import subprocess
from pathlib import Path

from tools.pause_recovp_hybrid_remaining import ROOT, OUT, now, processes, read, sha, write


def main():
    assert socket.gethostname().upper() == 'DESKTOP-RFFERM3'
    os.chdir(ROOT)
    pause = read(OUT / 'PAUSE_STATUS.json')
    assert pause['status'] == 'USER_PAUSED'
    assert not processes()
    import torch
    audit = {'started_utc': now(), 'device': 'cpu', 'seeds': {}, 'data': {}}
    backup = ROOT / 'backups/recovp_hybrid_user_pause_20261127'
    backup.mkdir(parents=True, exist_ok=True)
    for seed in (79, 97):
        base = ROOT / f'outputs/recovp_hybrid_seed{seed}'
        folder = base / f'seed_{seed}'
        config = read(base / 'CONFIG.json')
        source_checks = {rel: sha(ROOT / rel) == digest for rel, digest in config['source_sha256'].items()}
        assert all(source_checks.values()), source_checks
        for rel, digest in config['data_sha256'].items():
            if rel not in audit['data']:
                audit['data'][rel] = sha(ROOT / 'data/paper2_1103' / rel)
            assert audit['data'][rel] == digest, rel
        assert sha(base / 'PREFLIGHT.json') == config['preflight_sha256']
        payload = torch.load(folder / 'latest.pt', map_location='cpu', weights_only=False)
        epoch = pause['seeds'][str(seed)]['saved_epoch']
        identity = payload['identity']
        assert payload['epoch'] == epoch and payload['global_step'] == epoch * 48
        assert payload['format'] == f'recovp-hybrid-seed{seed}-epoch-v1'
        assert identity['seed'] == seed
        assert identity['config_sha256'] == sha(base / 'CONFIG.json')
        assert identity['preflight_sha256'] == config['preflight_sha256']
        assert identity['source_sha256'] == config['source_sha256']
        assert identity['train_sha256'] == config['data_sha256']['train.npz']
        assert payload['model'] and payload['optimizer']['state'] and payload['rng']
        checkpoints = []
        for path in sorted(folder.rglob('*.pt')):
            sidecar = read(path.with_suffix('.json'))
            digest = sha(path)
            assert digest == sidecar['checkpoint_sha256'], str(path)
            assert sidecar['identity'] == identity
            checkpoints.append({'path': str(path.relative_to(base)), 'sha256': digest, 'epoch': sidecar['epoch'], 'bytes': path.stat().st_size})
        saved = sorted(int(p.stem.split('_')[-1]) for p in (folder / 'recovery').glob('epoch_*.pt'))
        assert saved == list(range(epoch + 1)), saved
        assert all((folder / f'epoch_{e:03d}.pt').exists() for e in range(32, epoch + 1, 32))
        for rel in config['source_sha256']:
            target = backup / 'sources' / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / rel, target)
        dest = backup / base.name
        assert not dest.exists(), 'Do not overwrite an existing pause backup'
        shutil.copytree(base, dest)
        manifest = []
        for path in sorted(base.rglob('*')):
            if path.is_file():
                copied = dest / path.relative_to(base)
                digest = sha(path)
                assert sha(copied) == digest
                manifest.append({'path': str(path.relative_to(base)), 'sha256': digest, 'bytes': path.stat().st_size})
        row = {'epoch': epoch, 'global_step': payload['global_step'], 'identity': identity,
               'execution': config['execution'], 'model_tensors': len(payload['model']),
               'optimizer_states': len(payload['optimizer']['state']), 'rng_keys': list(payload['rng']),
               'all_recovery_epochs': saved, 'checkpoint_count': len(checkpoints),
               'checkpoints': checkpoints, 'source_checks': source_checks,
               'server_backup': str(dest), 'backup_files': len(manifest),
               'backup_manifest': manifest, 'all_checks_passed': True}
        audit['seeds'][str(seed)] = row
        write(OUT / f'seed_{seed}_AUDIT.json', row)
        print(f'seed {seed}: epoch {epoch}; {len(checkpoints)} checkpoints verified; backup verified', flush=True)
    audit['related_processes_after_audit'] = processes()
    assert not audit['related_processes_after_audit']
    audit['gpu_snapshot'] = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,utilization.gpu,memory.used', '--format=csv,noheader,nounits'], text=True)
    audit['finished_utc'] = now()
    audit['all_checks_passed'] = True
    write(OUT / 'INTEGRITY_AUDIT.json', audit)
    write(backup / 'PAUSE_STATUS.json', pause)
    write(backup / 'INTEGRITY_AUDIT.json', audit)
    print(audit['gpu_snapshot'], flush=True)


if __name__ == '__main__':
    main()
