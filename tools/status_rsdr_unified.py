"""Read-only benchmark status from immutable condition receipts and heartbeats."""
from __future__ import annotations
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import time

from paper2.io import ROOT, now, read as read_json, sha
from tools.durable_io import atomic_json, process_alive
from tools.compare_rsdr_unified import DEST, LIGHT, FREEZE


def read(path):
    # Windows can briefly deny opens while a heartbeat is atomically replaced.
    # Retry access only; invalid JSON or scientific assertions still fail.
    for attempt in range(12):
        try:return read_json(path)
        except (PermissionError,FileNotFoundError):
            if attempt==11:raise
            time.sleep(min(.04*2**attempt,.5))


def snapshot(save=False):
    f=read(FREEZE); completed=[]; coverage={}; remaining={}; durations={}
    for j in f['plan']:
        s=j['suite']; c=coverage.setdefault(s,dict(complete=0,expected=0))
        c['expected']+=1
        p=DEST/f"{s}_{j['index']:03d}/rsdr/result.json"
        if p.exists():
            r=read(p)
            assert r['status']=='COMPLETE' and r['identity']==dict(freeze_sha256=sha(FREEZE),**j)
            c['complete']+=1; completed.append(j)
            durations.setdefault(j['backend'],[]).append(r['elapsed_this_invocation_seconds']/j['condition']['count'])
        else:
            remaining[j['backend']]=remaining.get(j['backend'],0)+j['condition']['count']
    invocation=read(DEST/'INVOCATION.json')
    heartbeats={}
    for p in DEST.glob('PROGRESS_*.json'):
        h=read(p); h['age_seconds']=(datetime.now(timezone.utc)-datetime.fromisoformat(h['updated_utc'])).total_seconds()
        heartbeats[h['backend']]=h
    # Engineering estimate only. Cache fill, transfer and contention can change throughput.
    eta={k:v*statistics.median(durations[k][-8:])/3600 for k,v in remaining.items() if k in durations}
    result=dict(checked_utc=now(),status='COMPLETE' if (DEST/'COMPLETE.json').exists() else 'RUNNING' if process_alive(invocation['pid']) else 'STOPPED_WITHOUT_COMPLETION',
        pid=invocation['pid'],complete_conditions=len(completed),expected_conditions=len(f['plan']),
        complete_parent_evaluations=sum(j['condition']['count'] for j in completed),
        expected_parent_evaluations=sum(j['condition']['count'] for j in f['plan']),coverage=coverage,
        primary_complete=all(j in completed for j in f['plan'] if j['suite']=='main' and j['condition']['jam_type']=='repeat'),
        remaining_hours_by_backend_estimate=eta,eta_note='Recent eight completed cells per backend; engineering estimate, not a deadline.',
        heartbeats=heartbeats,failures=[str(p.relative_to(ROOT)) for p in DEST.glob('FAILURE_*.json')])
    if save:atomic_json(LIGHT/'RUN_STATUS.json',result)
    return result


if __name__=='__main__':
    r=snapshot(save=True)
    print(json.dumps({k:v for k,v in r.items() if k!='heartbeats'},ensure_ascii=False,indent=2))
