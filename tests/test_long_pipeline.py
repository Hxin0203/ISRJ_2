"""Failure-focused engineering checks; synthetic data are not research results."""
import os
import subprocess
import sys
import time

import numpy as np
import pytest
import torch

from paper2 import train_256_8seed as train
from paper2.io import read, sha
from tools import durable_io
from tools.resume_scan_1103 import verify_frozen_sources


def payload(epoch, segment=0):
    return dict(format=train.CHECKPOINT_FORMAT, identity=train.checkpoint_identity(53, "d", "c", {}),
                seed=53, epoch=epoch, segment=segment, model={"w": torch.tensor([float(epoch)])},
                epoch_record={"epoch": epoch})


def test_replace_retries_transient_permission_but_not_other_io(tmp_path, monkeypatch):
    actual = os.replace
    calls = []
    def transient(source, target):
        calls.append(1)
        if len(calls) < 3:
            raise PermissionError("simulated transient Windows sharing failure")
        actual(source, target)
    monkeypatch.setattr(durable_io.os, "replace", transient)
    monkeypatch.setattr(durable_io.time, "sleep", lambda _: None)
    durable_io.atomic_json(tmp_path / "s.json", {"saved": True})
    assert read(tmp_path / "s.json") == {"saved": True} and len(calls) == 3
    def permanent(*_):
        raise OSError("simulated disk failure")
    monkeypatch.setattr(durable_io.os, "replace", permanent)
    with pytest.raises(OSError, match="disk failure"):
        durable_io.atomic_json(tmp_path / "s.json", {"saved": False})
    assert read(tmp_path / "s.json")["saved"] is True


def test_newer_archive_survives_latest_receipt_crash_window(tmp_path):
    first, second = payload(44), payload(45)
    train.write_recovery(tmp_path, first)
    old_receipt = read(tmp_path / "latest.json")
    train.write_recovery(tmp_path, second)
    durable_io.atomic_json(tmp_path / "latest.json", old_receipt)
    selected = train.choose_resume(tmp_path, second["identity"])
    assert selected[1]["epoch"] == 45 and selected[0].parent.name == "recovery"
    (tmp_path / "recovery/epoch_045_segment_000.pt").write_bytes(b"damaged")
    (tmp_path / "latest.pt").write_bytes(b"damaged")
    assert train.choose_resume(tmp_path, first["identity"])[1]["epoch"] == 44


def test_immutable_milestone_and_segment_order(tmp_path):
    value = payload(32)
    train.write_recovery(tmp_path, value, True)
    digest = sha(tmp_path / "epoch_032.pt")
    train.write_recovery(tmp_path, value, True)
    assert sha(tmp_path / "epoch_032.pt") == digest
    with pytest.raises(RuntimeError, match="immutable"):
        train.write_recovery(tmp_path, {**value, "model": {"w": torch.tensor([99.])}}, True)
    train.write_recovery(tmp_path, payload(33, 1))
    train.write_recovery(tmp_path, payload(33, 0))  # stale latest must not hide segment 1
    assert train.choose_resume(tmp_path, value["identity"])[1]["segment"] == 1


def test_freeze_rejects_modified_original_and_unapproved_addition():
    frozen = {"sources": {"paper2/network.py": "a"}}
    assert verify_frozen_sources(frozen, {"paper2/network.py": "a", "paper2/train_256_8seed.py": "b"})
    for values in ({"paper2/network.py": "changed"}, {},
                   {"paper2/network.py": "a", "paper2/unregistered.py": "x"}):
        with pytest.raises(RuntimeError, match="source mismatch"):
            verify_frozen_sources(frozen, values)


def test_os_lock_excludes_second_process_and_releases_after_termination(tmp_path):
    path = tmp_path / "exclusive.lock"
    ready = tmp_path / "ready"
    code = ("from tools.durable_io import ProcessLock; from pathlib import Path; import time; "
            f"lock=ProcessLock({str(path)!r}).acquire(); Path({str(ready)!r}).touch(); time.sleep(30)")
    child = subprocess.Popen([sys.executable, "-c", code])
    try:
        deadline = time.monotonic() + 10
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(.05)
        assert ready.exists()
        assert durable_io.process_alive(child.pid)
        with pytest.raises(RuntimeError, match="another process"):
            durable_io.ProcessLock(path).acquire()
    finally:
        child.terminate()
        child.wait(timeout=10)
    with durable_io.ProcessLock(path):
        pass
    assert not durable_io.process_alive(child.pid)


def test_epoch_256_interruption_finalizes_without_retraining(tmp_path, monkeypatch):
    import tools.training_progress_256
    monkeypatch.setattr(tools.training_progress_256, "refresh", lambda: None)
    monkeypatch.setattr(train, "BASE", tmp_path)
    monkeypatch.setattr(train, "MILESTONES", (256,))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    folder = tmp_path / "seed_53"
    identity = train.checkpoint_identity(53, "d", "c", {})
    net = train.CorrectionNet()
    optimizer = train.optimizer_for(net)
    record = dict(seed=53, epoch=256, segment=0, updates_this_epoch=48,
                  total=1., endpoint=1., pc=0., damage=0.)
    value = train.pack_checkpoint(53, 256, 0, net, optimizer, identity, record)
    train.write_recovery(folder, value, True)
    def dev(seed, epoch):
        durable_io.atomic_json(folder / f"dev_epoch_{epoch:03d}_v4.json", {"synthetic": True})
    monkeypatch.setattr(train, "evaluate_checkpoint", dev)
    result = train.train_one(53, {"s": np.empty((512, 0))}, "d", "c", {}, {})
    assert result["seed"] == 53 and (folder / "TRAINING_COMPLETE.json").is_file()
    assert read(folder / "training_epoch_256_segment_000.json")["total"] == 1.
    assert train.train_one(53, {}, "d", "c", {}, {}) == result
