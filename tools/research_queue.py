"""Evidence-bound research management; never launches research or training."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / 'reports/ISRJ2_RESEARCH_QUEUE.json'


def utc():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


@contextmanager
def lock(path):
    name = Path(str(path) + '.lock')
    fd = os.open(name, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        os.write(fd, json.dumps({'pid': os.getpid(), 'utc': utc()}).encode())
        os.close(fd)
        fd = None
        yield
    finally:
        if fd is not None:
            os.close(fd)
        name.unlink()


def atomic(path, value):
    path = Path(path)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as f:
        temp = Path(f.name)
        f.write((json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode())
        f.flush()
        os.fsync(f.fileno())
    try:
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def eligible(state, task):
    by_id = {t['id']: t for t in state['tasks']}
    return task['authorized'] and all(by_id.get(d, {}).get('status') == 'COMPLETED' for d in task['dependencies'])


def select(state):
    candidates = [t for t in state['tasks'] if t['status'] == 'READY' and eligible(state, t)]
    return min(candidates, key=lambda t: (t['priority'], t['id']))['id'] if candidates else None


def evidence(root, path, digest):
    root = Path(root).resolve()
    p = (root / path).resolve()
    if not p.is_relative_to(root) or p == root or not p.is_file():
        raise ValueError('evidence must be an existing workspace file')
    if len(digest) != 64 or hashlib.sha256(p.read_bytes()).hexdigest() != digest.lower():
        raise ValueError('evidence SHA-256 mismatch')
    return {'path': p.relative_to(root).as_posix(), 'sha256': digest.lower()}


def mutate(path, task_id, action, *, root=ROOT, evidence_path=None, sha256=None, verification=None):
    with lock(path):
        state = read(path)
        task = next(t for t in state['tasks'] if t['id'] == task_id)
        if not eligible(state, task):
            raise ValueError('authorization or dependencies not satisfied')
        if action in ('complete', 'ready'):
            if not verification or not verification.strip():
                raise ValueError('explicit verification description required')
            proof = evidence(root, evidence_path, sha256)
            if task.get('expected_report') and proof['path'] != task['expected_report']:
                raise ValueError('evidence does not match the task acceptance report')
            if action == 'complete' and task['status'] == 'COMPLETED':
                if task['evidence'] != proof:
                    raise ValueError('completed evidence cannot be replaced')
                return {'idempotent': True, 'next': select(state)}
            if action == 'complete' and task['status'] not in ('READY', 'EXECUTE', 'VERIFY'):
                raise ValueError('task is not executable')
            if action == 'ready' and task['status'] != 'WAIT_DEPENDENCY':
                raise ValueError('only dependency waiting tasks can be released')
            task['evidence'] = proof
            task['verification'] = verification
            task['status'] = 'COMPLETED' if action == 'complete' else 'READY'
            stages = ['VERIFY', 'ARCHIVE', 'NEXT'] if action == 'complete' else ['CHECK_EVIDENCE', 'READY']
        elif action == 'fail':
            if task['status'] not in ('READY', 'EXECUTE', 'VERIFY') or not verification:
                raise ValueError('failure requires executable task and reason')
            task['status'] = 'BLOCKED'
            task['failure'] = verification
            stages = ['DIAGNOSE', 'BLOCKED', 'NEXT']
        else:
            raise ValueError('unsupported action')
        state['events'].append({'utc': utc(), 'task': task_id, 'stages': stages,
                                'verification': verification, 'evidence': task.get('evidence')})
        state['updated_utc'] = utc()
        state['next_ready'] = select(state)
        atomic(path, state)
        return {'status': task['status'], 'next': state['next_ready'], 'idempotent': False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['status', 'select', 'complete', 'ready', 'fail'])
    p.add_argument('--state', type=Path, default=DEFAULT)
    p.add_argument('--task')
    p.add_argument('--evidence')
    p.add_argument('--sha256')
    p.add_argument('--verification')
    a = p.parse_args()
    if a.action in ('status', 'select'):
        s = read(a.state)
        result = {'next': select(s), 'execution': 'selection_only_no_research_executed'}
        if a.action == 'status':
            result['tasks'] = s['tasks']
    else:
        result = mutate(a.state, a.task, a.action, evidence_path=a.evidence,
                        sha256=a.sha256, verification=a.verification)
    print(json.dumps(result, ensure_ascii=True))


if __name__ == '__main__':
    main()
