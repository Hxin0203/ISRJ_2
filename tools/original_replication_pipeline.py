"""Persistent supervision of executable replication stages and remaining work."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import psutil

from tools.durable_io import atomic_json
from tools.server_gpu_admission import query_resources, require_server_identity

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'outputs/original_paper_replication_256_v1'
OUT = BASE / 'pipeline'
STATUS = OUT / 'PIPELINE_STATUS.json'
SEEDS = (67, 97, 53, 41, 109)
TASKS = [
    {'id': 'P2_AUDIT', 'deps': [], 'command': ['tools.original_replication_p2']},
    {'id': 'P2_CALIBRATION', 'deps': ['P2_AUDIT'],
     'command': ['tools.paper2_baseline_suite.equal_pfa_suite', 'queue',
                 '--family', 'ReCoVP-Hybrid', *[value for seed in SEEDS for value in ('--seed', str(seed))]]},
    {'id': 'R3_MAIN', 'deps': ['P2_CALIBRATION'],
     'command': ['tools.original_replication_suite', 'queue'],
     'external_script': 'recover_original_replication_r3.ps1',
     'role': 'legacy three-mechanism averaged reference'},
    {'id': 'R3_CLASSICAL_MECHANISMS', 'deps': ['P2_CALIBRATION'],
     'command': ['tools.original_replication_classical_by_mechanism', 'queue']},
    {'id': 'R5_HYBRID', 'deps': ['P2_CALIBRATION'],
     'command': ['tools.original_replication_detection', 'queue']},
    {'id': 'R3_MECHANISM_SUMMARY', 'deps': ['P2_CALIBRATION'],
     'command': ['tools.original_replication_mechanism_summary', 'watch']},
    {'id': 'R3_SUMMARY', 'deps': ['R3_MAIN'],
     'command': ['tools.original_replication_suite', 'summarize'],
     'role': 'legacy averaged summary and incremental mechanism reconciliation'},
    {'id': 'R1_PREVIEW', 'deps': ['P2_CALIBRATION'],
     'command': ['tools.original_replication_r1'],
     'external_waiter': True, 'role': 'initial example with complete R1 acceptance pending'},
]
GAPS = {
    'P2_SCIENTIFIC_ACCEPTANCE': 'Four-method per-mechanism scientific acceptance pending',
    'R1_FULL': 'Reference spectra, all mechanisms, coherence examples and rendered QA pending',
    'R2': 'Full M2/M4 two-JSR-definition width execution pending',
    'R3_CVED_MECHANISMS': 'Existing sealed CVED-Net per-mechanism evaluation pending',
    'R4': 'Four-topic execution pending',
    'R5_COMPARATORS': 'Max-TF, GMM-OMP and sealed CVED-Net full-grid execution pending',
    'R6': 'Hybrid ablation and matched U-Net training/evaluation execution pending',
}


def now():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def complete(task_id):
    if task_id == 'P2_AUDIT':
        return (BASE / 'P2/P2_COMPLETE.json').exists()
    if task_id == 'P2_CALIBRATION':
        return all((ROOT / f'outputs/paper2_baseline_suite_20261126/E3/ReCoVP-Hybrid/seed_{seed}/ROW.json').exists()
                   for seed in SEEDS)
    if task_id == 'R3_MAIN':
        folders = [BASE / f'R3_main/results/ReCoVP-Hybrid/seed_{seed}' for seed in SEEDS]
        folders += [BASE / f'R3_main/results/{method}/single' for method in ('Max-TF', 'GMM-OMP', 'CVEDNet')]
        return ((BASE / 'R3_main/ALL_COMPLETE.json').exists()
                and all((folder / 'DONE.json').exists() and len(list(folder.glob('cell_*.npz'))) == 35
                        for folder in folders))
    if task_id == 'R3_CLASSICAL_MECHANISMS':
        receipt = BASE / 'R3_by_mechanism/CLASSICAL_COMPLETE.json'
        return receipt.exists() and all(read(BASE / f'R3_by_mechanism/{method}/DONE.json').get('completed_cells') == 35
                                        for method in ('Max-TF', 'GMM-OMP'))
    if task_id == 'R5_HYBRID':
        receipt = BASE / 'R5_detection/HYBRID_COMPLETE.json'
        return receipt.exists() and all(read(BASE / f'R5_detection/seed_{seed}/DONE.json').get('completed_cells') == 310
                                        for seed in SEEDS)
    if task_id == 'R3_MECHANISM_SUMMARY':
        receipt = BASE / 'R3_by_mechanism/postprocessing/STATUS.json'
        return receipt.exists() and read(receipt).get('state') == 'HYBRID_CLASSICAL_READY_CVED_PENDING'
    if task_id == 'R1_PREVIEW':
        return (BASE / 'R1_waveforms/R1_PRELIMINARY.json').exists()
    return (BASE / 'R3_main/SWEEP_SUMMARY.json').exists()


def matching_processes(task):
    matches = []
    module = task['command'][0]
    action = task['command'][1] if len(task['command']) > 1 else None
    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'create_time']):
        command = proc.info['cmdline'] or []
        name = (proc.info['name'] or '').lower()
        python_match = (name in ('python.exe', 'pythonw.exe') and module in command
                        and (action is None or action in command))
        shell_match = (name == 'powershell.exe' and
                       ((task.get('external_script') and task['external_script'] in ' '.join(command))
                        or (task.get('external_waiter') and '-m tools.original_replication_r1' in ' '.join(command))))
        if python_match or shell_match:
            item = {'pid': proc.pid, 'create_time': proc.info['create_time']}
            if shell_match and task.get('external_waiter'):
                item['waiting_for_external_dependency'] = True
            matches.append(item)
    return matches


def progress():
    result = {}
    for seed in SEEDS:
        path = BASE / f'R5_detection/seed_{seed}/WORKER_STATUS.json'
        if path.exists():
            row = read(path)
            result[f'R5_Hybrid_{seed}'] = {key: row.get(key) for key in
                                         ('state', 'completed_cells', 'completed_parent_chunks', 'last_update_utc')}
    for method in ('Max-TF', 'GMM-OMP'):
        path = BASE / f'R3_by_mechanism/{method}/STATUS.json'
        if path.exists():
            row = read(path)
            result[f'R3_{method}'] = {key: row.get(key) for key in
                                     ('state', 'completed_cells', 'completed_chunks', 'updated_utc')}
    return result


def main():
    require_server_identity()
    OUT.mkdir(parents=True, exist_ok=True)
    state = read(STATUS) if STATUS.exists() else {'created_utc': now(), 'tasks': {}}
    if state.get('controller_pid') != os.getpid():
        atomic_json(OUT / f'PRIOR_CONTROLLER_{int(time.time())}.json', state)
    state.update({'controller_pid': os.getpid(), 'supervisor_started_utc': now(),
                  'scope_complete': False, 'implementation_gaps': GAPS,
                  'cvednet_policy': 'reuse sealed epoch320 checkpoint; same-data retraining is supplementary'})
    children = {}
    while True:
        for task_id, proc in list(children.items()):
            code = proc.poll()
            if code is not None:
                children.pop(task_id)
                state['tasks'][task_id].update({'exit_code': code, 'finished_utc': now(),
                                               'retry_after_unix': time.time() + 60})
        try:
            for task in TASKS:
                task_id = task['id']
                record = state['tasks'].setdefault(task_id, {'state': 'PENDING', 'attempts': 0})
                if complete(task_id):
                    record.update({'state': 'DONE', 'receipt_verified_utc': now()})
                    continue
                live = matching_processes(task)
                if live:
                    state_name = 'RUNNING'
                    if all(item.get('waiting_for_external_dependency') for item in live):
                        state_name = 'WAITING_EXTERNAL_DEPENDENCIES'
                    record.update({'state': state_name, 'processes': live, 'observed_utc': now()})
                    continue
                if any(state['tasks'].get(dep, {}).get('state') != 'DONE' for dep in task['deps']):
                    record['state'] = 'WAITING_DEPENDENCIES'
                    continue
                if record.get('attempts', 0) >= 3:
                    record['state'] = 'FAILED_RECOVERY_EXHAUSTED'
                    continue
                if time.time() < record.get('retry_after_unix', 0):
                    record['state'] = 'WAITING_RETRY'
                    continue
                resources = query_resources()
                if resources['memory']['utilization_percent'] >= 90:
                    record['state'] = 'WAITING_MEMORY'
                    continue
                log_path = OUT / 'logs' / f'{task_id}.log'
                log_path.parent.mkdir(parents=True, exist_ok=True)
                env = {**os.environ, 'PYTHONUTF8': '1', 'PYTHONDONTWRITEBYTECODE': '1',
                       'CUDA_DEVICE_ORDER': 'PCI_BUS_ID', 'CUDA_VISIBLE_DEVICES': ''}
                with log_path.open('a', encoding='utf-8') as handle:
                    proc = subprocess.Popen([sys.executable, '-u', '-m', *task['command']],
                                            cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT,
                                            creationflags=getattr(subprocess, 'CREATE_NEW_PROCESS_GROUP', 0))
                children[task_id] = proc
                record.update({'state': 'STARTING', 'pid': proc.pid,
                               'create_time': psutil.Process(proc.pid).create_time(),
                               'attempts': record.get('attempts', 0) + 1,
                               'started_utc': now(), 'dispatch_resources': resources,
                               'command': task['command'], 'role': task.get('role')})
                atomic_json(STATUS, state)
            state['progress'] = progress()
            state['state'] = 'RUNNING_WITH_IMPLEMENTATION_GAPS'
            if all(state['tasks'][task['id']]['state'] == 'DONE' for task in TASKS):
                state['state'] = 'WAITING_IMPLEMENTATION'
            state['heartbeat_utc'] = now()
            atomic_json(STATUS, state)
        except Exception:
            atomic_json(OUT / 'SUPERVISOR_LAST_ERROR.json', {'at_utc': now(), 'error': traceback.format_exc()})
        time.sleep(20)


if __name__ == '__main__':
    main()
