"""Supervise this finite frozen benchmark, then produce its final report.

Only an explicit MemoryError is retried. Scientific assertions and other errors
stop the supervisor. The frozen evaluator, plan and existing blocks are intact.
"""
from __future__ import annotations

import argparse
import ctypes
from datetime import datetime
import os
import subprocess
import sys
import time
import traceback

from paper2.io import ROOT, now, read, sha
from tools.durable_io import ProcessLock, atomic_json, process_alive
from tools.status_rsdr_unified import DEST, FREEZE, LIGHT, snapshot


def memory():
    if os.name != 'nt':
        return {}
    class Status(ctypes.Structure):
        _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong)] + [
            (name, ctypes.c_ulonglong) for name in (
                'physical_total', 'physical_available', 'commit_limit',
                'commit_available', 'virtual_total', 'virtual_available', 'extended')]
    value = Status()
    value.length = ctypes.sizeof(value)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(value)):
        raise ctypes.WinError()
    return {k: getattr(value, k) for k in (
        'physical_total', 'physical_available', 'commit_limit', 'commit_available')}


def is_memory_failure(path):
    return 'MemoryError' in read(path).get('traceback', '')


def wait_for_capacity(timeout=900):
    deadline = time.monotonic() + timeout
    while True:
        available = memory()
        if not available or (available['physical_available'] >= 4 * 2**30
                             and available['commit_available'] >= 10 * 2**30):
            return available
        if time.monotonic() >= deadline:
            raise RuntimeError('Memory capacity did not recover; saved blocks retained.')
        time.sleep(30)


def run(max_restarts=2):
    with ProcessLock(DEST / 'SUPERVISOR.lock'):
        events = []
        attempts = 0
        while not (DEST / 'COMPLETE.json').exists():
            invocation = read(DEST / 'INVOCATION.json')
            pid = invocation['pid']
            if process_alive(pid):
                status = snapshot(save=True)
                atomic_json(DEST / 'SUPERVISOR_STATUS.json', dict(
                    updated_utc=now(), pid=os.getpid(), stage='WAITING_FOR_FROZEN_EVALUATOR',
                    evaluator_pid=pid, saved=status['complete_conditions'],
                    expected=status['expected_conditions'], memory=memory(), restarts=attempts))
                time.sleep(30)
                continue
            failures = sorted(DEST.glob('FAILURE_*.json'))
            if (not failures or not is_memory_failure(failures[-1]) or
                datetime.fromisoformat(read(failures[-1])['failed_utc']) <
                    datetime.fromisoformat(invocation['started_utc'])):
                raise RuntimeError('Evaluator stopped without an explicit MemoryError; inspect failure before resuming.')
            if attempts >= max_restarts:
                raise RuntimeError('Bounded memory-recovery retries exhausted; saved blocks retained.')
            available = wait_for_capacity()
            attempts += 1
            stamp = time.time_ns()
            with (DEST / f'supervised_{stamp}.stdout.log').open('wb') as out, \
                 (DEST / f'supervised_{stamp}.stderr.log').open('wb') as err:
                child = subprocess.Popen(
                    [sys.executable, '-u', '-m', 'tools.compare_rsdr_unified', 'run'],
                    stdout=out, stderr=err, cwd=FREEZE.parents[1],
                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            event = dict(restarted_utc=now(), previous_failure=failures[-1].name,
                         pid=child.pid, attempt=attempts, memory=available)
            events.append(event)
            atomic_json(DEST / 'SUPERVISOR_EVENTS.json', dict(events=events))
            # Wait until the child records its invocation, or fail explicitly.
            deadline = time.monotonic() + 120
            while read(DEST / 'INVOCATION.json')['pid'] != child.pid:
                if child.poll() is not None or time.monotonic() >= deadline:
                    raise RuntimeError('Recovery evaluator did not start; inspect its saved logs.')
                time.sleep(1)

        wait_for_capacity()
        atomic_json(DEST / 'SUPERVISOR_STATUS.json', dict(
            updated_utc=now(), pid=os.getpid(), stage='FINAL_AUDIT_AND_REPORT', restarts=attempts))
        stamp = time.time_ns()
        with (DEST / f'final_{stamp}.stdout.log').open('wb') as out, \
             (DEST / f'final_{stamp}.stderr.log').open('wb') as err:
            completed = subprocess.run(
                [sys.executable, '-u', '-m', 'tools.finalize_rsdr_unified', '--once', 'all'],
                stdout=out, stderr=err, cwd=FREEZE.parents[1],
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        if completed.returncode:
            raise RuntimeError('Final postprocessing failed; inspect preserved logs and failure receipt.')
        receipt = read(LIGHT / 'POSTPROCESS_ALL_COMPLETE.json')
        assert receipt['conditions'] == 564 and receipt['status'] == 'COMPLETE'
        atomic_json(LIGHT / 'SUPERVISOR_COMPLETE.json', dict(
            completed_utc=now(), status='COMPLETE', conditions=564, restarts=attempts,
            freeze_sha256=sha(FREEZE), postprocess_receipt_sha256=sha(LIGHT / 'POSTPROCESS_ALL_COMPLETE.json'),
            frozen_sources_modified=False, failure_history_preserved=True))
        atomic_json(DEST / 'SUPERVISOR_STATUS.json', dict(
            updated_utc=now(), pid=os.getpid(), stage='COMPLETE', restarts=attempts))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-restarts', type=int, default=2)
    args = parser.parse_args()
    if not 0 <= args.max_restarts <= 2:
        parser.error('max-restarts must be between 0 and 2')
    try:
        run(args.max_restarts)
    except BaseException:
        atomic_json(DEST / f'SUPERVISOR_FAILURE_{time.time_ns()}.json', dict(
            failed_utc=now(), traceback=traceback.format_exc()))
        raise
