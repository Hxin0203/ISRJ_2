"""Exercise failure handling in isolation; no model or evaluator is launched."""
from types import SimpleNamespace

import pytest

from tools import supervise_rsdr_unified as supervisor
from tools.durable_io import atomic_json
from paper2.io import read, sha


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    dest = tmp_path / 'run'
    light = tmp_path / 'evidence'
    freeze = tmp_path / 'configs' / 'freeze.json'
    dest.mkdir()
    light.mkdir()
    atomic_json(freeze, {'test_fixture': True})
    atomic_json(dest / 'INVOCATION.json', dict(pid=101, started_utc='2026-11-15T12:00:00+00:00'))
    monkeypatch.setattr(supervisor, 'DEST', dest)
    monkeypatch.setattr(supervisor, 'LIGHT', light)
    monkeypatch.setattr(supervisor, 'FREEZE', freeze)
    monkeypatch.setattr(supervisor, 'process_alive', lambda pid: False)
    return dest, light, freeze


@pytest.mark.parametrize('traceback,failed_utc', [
    ('AssertionError: identity differs', '2026-11-15T12:01:00+00:00'),
    ('MemoryError: allocation failed', '2026-11-15T11:59:00+00:00'),
])
def test_rejects_scientific_failure_and_stale_memory_error(isolated, monkeypatch, traceback, failed_utc):
    dest, light, _ = isolated
    failure = dest / 'FAILURE_001.json'
    atomic_json(failure, dict(traceback=traceback, failed_utc=failed_utc))
    original = sha(failure)
    def unexpected(*args, **kwargs):
        pytest.fail('An unapproved failure must not launch another process')
    monkeypatch.setattr(supervisor.subprocess, 'Popen', unexpected)
    monkeypatch.setattr(supervisor.subprocess, 'run', unexpected)
    with pytest.raises(RuntimeError, match='without an explicit MemoryError'):
        supervisor.run()
    assert sha(failure) == original
    assert not (light / 'SUPERVISOR_COMPLETE.json').exists()


def test_memory_recovery_is_bounded_and_preserves_failures(isolated, monkeypatch):
    dest, light, _ = isolated
    atomic_json(dest / 'FAILURE_000.json', dict(
        traceback='MemoryError: allocation failed', failed_utc='2026-11-15T12:01:00+00:00'))
    original = sha(dest / 'FAILURE_000.json')
    launches = []
    monkeypatch.setattr(supervisor, 'wait_for_capacity', lambda: {})
    def fake_launch(*args, **kwargs):
        number = len(launches) + 1
        launches.append(args)
        pid = 101 + number
        atomic_json(dest / 'INVOCATION.json', dict(pid=pid, started_utc='2026-11-15T12:00:00+00:00'))
        atomic_json(dest / f'FAILURE_{number:03d}.json', dict(
            traceback='MemoryError: allocation failed', failed_utc='2026-11-15T12:01:00+00:00'))
        return SimpleNamespace(pid=pid)
    monkeypatch.setattr(supervisor.subprocess, 'Popen', fake_launch)
    with pytest.raises(RuntimeError, match='retries exhausted'):
        supervisor.run(max_restarts=2)
    assert len(launches) == 2
    assert len(list(dest.glob('FAILURE_*.json'))) == 3
    assert sha(dest / 'FAILURE_000.json') == original
    assert not (light / 'SUPERVISOR_COMPLETE.json').exists()


def test_live_evaluator_is_not_duplicated_and_success_binds_audit(isolated, monkeypatch):
    dest, light, freeze = isolated
    monkeypatch.setattr(supervisor, 'process_alive', lambda pid: True)
    monkeypatch.setattr(supervisor, 'snapshot', lambda **kwargs: dict(complete_conditions=563, expected_conditions=564))
    monkeypatch.setattr(supervisor, 'memory', lambda: {})
    monkeypatch.setattr(supervisor, 'wait_for_capacity', lambda: {})
    monkeypatch.setattr(supervisor.time, 'sleep', lambda seconds: atomic_json(dest / 'COMPLETE.json', {}))
    def unexpected(*args, **kwargs):
        pytest.fail('A live evaluator must not be duplicated')
    monkeypatch.setattr(supervisor.subprocess, 'Popen', unexpected)
    calls = []
    def fake_finalize(*args, **kwargs):
        calls.append(args[0])
        atomic_json(light / 'POSTPROCESS_ALL_COMPLETE.json', dict(status='COMPLETE', conditions=564))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(supervisor.subprocess, 'run', fake_finalize)
    supervisor.run()
    assert len(calls) == 1 and calls[0][-2:] == ['--once', 'all']
    receipt = read(light / 'SUPERVISOR_COMPLETE.json')
    assert receipt['freeze_sha256'] == sha(freeze)
    assert receipt['postprocess_receipt_sha256'] == sha(light / 'POSTPROCESS_ALL_COMPLETE.json')
    assert receipt['restarts'] == 0


def test_failed_final_audit_cannot_publish_success(isolated, monkeypatch):
    dest, light, _ = isolated
    atomic_json(dest / 'COMPLETE.json', {})
    monkeypatch.setattr(supervisor, 'wait_for_capacity', lambda: {})
    monkeypatch.setattr(supervisor.subprocess, 'run', lambda *args, **kwargs: SimpleNamespace(returncode=1))
    with pytest.raises(RuntimeError, match='postprocessing failed'):
        supervisor.run()
    assert not (light / 'SUPERVISOR_COMPLETE.json').exists()


def test_insufficient_memory_stops_instead_of_retrying(isolated, monkeypatch):
    monkeypatch.setattr(supervisor, 'memory', lambda: dict(physical_available=0, commit_available=0))
    with pytest.raises(RuntimeError, match='did not recover'):
        supervisor.wait_for_capacity(timeout=0)
