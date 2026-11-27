"""One-shot, user-authorized pause of seeds 79/97/109 on server B.

Does not change frozen training sources, resume training, or terminate process trees.
Waits for a complete epoch publication before stopping each selected worker.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import time
from datetime import datetime, timezone

ROOT = Path('D:/code/Xin/ISRJ_222')
OUT = ROOT / 'evidence/outputs/recovp_hybrid_user_pause_20261127'


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    tmp.replace(path)


def processes():
    command = "Get-CimInstance Win32_Process | Where-Object { $_.Name -match 'python' -and $_.CommandLine -match '-m tools\\.(run_recovp_hybrid_8seed_pipeline|run_recovp_hybrid_split_pipeline|recover_recovp_hybrid_queue|start_recovp_hybrid_extended|vp_1103_recovp_hybrid_extended)( |$)' } | Select-Object ProcessId,ParentProcessId,Name,CommandLine | ConvertTo-Json -Depth 3"
    result = subprocess.check_output(['powershell', '-NoProfile', '-Command', command], text=True)
    value = json.loads(result) if result.strip() else []
    return value if isinstance(value, list) else [value]


def stop(selected):
    stopped = []
    for row in selected:
        fresh = {p['ProcessId']: p for p in processes()}
        if row['ProcessId'] not in fresh:
            continue
        assert fresh[row['ProcessId']]['CommandLine'] == row['CommandLine'], 'PID identity changed'
        subprocess.run(['powershell', '-NoProfile', '-Command', f"Stop-Process -Id {int(row['ProcessId'])} -Force -ErrorAction Stop"], check=True)
        stopped.append(row)
    return stopped


def main():
    assert socket.gethostname().upper() == 'DESKTOP-RFFERM3', 'server B only'
    os.chdir(ROOT)
    OUT.mkdir(parents=True, exist_ok=True)
    receipt = {'status': 'PAUSING', 'requested_seeds': [79, 97, 109],
               'started_utc': now(), 'automatic_resume': False,
               'resume_requires': 'A new explicit user instruction; do not restart controllers.',
               'seeds': {}, 'stopped_controllers': []}
    write(OUT / 'PAUSE_REQUEST.json', receipt)
    controllers = [p for p in processes() if '-m tools.vp_1103_recovp_hybrid_extended' not in p['CommandLine']]
    receipt['stopped_controllers'] = stop(controllers)
    pending = {}
    for seed in [79, 97]:
        folder = ROOT / f'outputs/recovp_hybrid_seed{seed}/seed_{seed}'
        records = sorted(folder.glob('training_epoch_*.json'))
        pending[seed] = int(read(records[-1])['epoch']) if records else -1
    deadline = time.monotonic() + 420
    print(json.dumps({'baseline_epochs': pending}), flush=True)
    while pending:
        if time.monotonic() > deadline:
            raise RuntimeError('No newly committed epoch within 7 minutes; workers left intact for inspection')
        for seed, baseline in list(pending.items()):
            folder = ROOT / f'outputs/recovp_hybrid_seed{seed}/seed_{seed}'
            records = sorted(folder.glob('training_epoch_*.json'))
            if not records:
                continue
            record = read(records[-1]); epoch = record['epoch']
            if epoch <= baseline:
                continue
            latest = read(folder / 'latest.json')
            archive = folder / 'recovery' / f'epoch_{epoch:03d}.pt'
            digest = record['checkpoint_sha256']
            if latest['epoch'] != epoch or latest['checkpoint_sha256'] != digest:
                continue
            assert sha(archive) == sha(folder / 'latest.pt') == digest
            selected = [p for p in processes() if '-m tools.vp_1103_recovp_hybrid_extended' in p['CommandLine'] and re.search(r'--seed\s+' + str(seed) + r'(?:\s|$)', p['CommandLine'])]
            assert selected, f'seed {seed} worker unexpectedly absent'
            stopped = stop(list(reversed(selected)))
            receipt['seeds'][str(seed)] = {'status': 'USER_PAUSED', 'saved_epoch': epoch,
                'global_step': record['global_step'], 'checkpoint': str(archive),
                'checkpoint_sha256': digest, 'stopped_workers': stopped, 'stopped_utc': now(),
                'partial_next_epoch': 'Any updates after this committed checkpoint are discarded and recomputed on resume.'}
            del pending[seed]
            write(OUT / 'PAUSE_STATUS.json', receipt)
            print(json.dumps(receipt['seeds'][str(seed)]), flush=True)
        if pending:
            time.sleep(2)
    assert not (ROOT / 'outputs/recovp_hybrid_seed109').exists(), 'seed109 unexpectedly has outputs'
    receipt['seeds']['109'] = {'status': 'NOT_STARTED_USER_PAUSED', 'saved_epoch': None}
    time.sleep(5)
    receipt['remaining_related_processes'] = processes()
    assert not receipt['remaining_related_processes'], 'Related processes still present'
    receipt['status'] = 'USER_PAUSED'
    receipt['finished_utc'] = now()
    write(OUT / 'PAUSE_STATUS.json', receipt)
    print(json.dumps({'status': receipt['status'], 'finished_utc': receipt['finished_utc']}), flush=True)


if __name__ == '__main__':
    main()
