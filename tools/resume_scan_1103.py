"""Resume the frozen TEST with a separately recorded mechanical JSON I/O repair.

Every originally frozen source must be unchanged. Only the two explicitly
unrelated continuation modules may be additive; frozen files are never edited.
"""
from __future__ import annotations

import argparse
import os
import time

from paper2.io import ROOT, now, read, sha, source_bindings
from tools.durable_io import ProcessLock, atomic_copy, atomic_json

ADDITIVE_SOURCES = {"paper2/train_256_8seed.py", "paper2/dev_256_8seed_v4.py"}


def verify_frozen_sources(frozen, current=None):
    current = source_bindings() if current is None else current
    expected = frozen["sources"]
    changed = [name for name, digest in expected.items() if current.get(name) != digest]
    extras = {name: digest for name, digest in current.items() if name not in expected}
    if changed or set(extras) - ADDITIVE_SOURCES:
        raise RuntimeError(f"frozen source mismatch: changed/missing={changed}; unexpected additions={set(extras)-ADDITIVE_SOURCES}")
    return extras


def run_resume(run_id="paper2_1103_test_20261114", classical=False):
    from paper2 import scan_1103 as scan
    frozen = read(scan.FREEZE)
    extras = verify_frozen_sources(frozen)
    root = ROOT / "outputs" / run_id
    with ProcessLock(root / "RESUME_PROCESS.lock"):
        stamp = str(time.time_ns())
        old_invocation = root / "INVOCATION.json"
        if old_invocation.exists():
            atomic_copy(old_invocation, root / "invocation_history" / f"{stamp}.json")
        repair = dict(created_utc=now(), pid=os.getpid(), freeze_sha256=sha(scan.FREEZE),
                      all_original_sources_unchanged=True, additive_continuation_sources=extras,
                      mechanical_sources={name: sha(ROOT / name) for name in
                          ("tools/durable_io.py", "tools/resume_scan_1103.py")},
                      change="JSON writes only: unique temporary file, fsync, bounded Windows sharing retry; original scientific code unchanged")
        atomic_json(root / "mechanical_repairs" / f"{stamp}.json", repair)
        atomic_json(ROOT / "evidence/outputs" / run_id / f"MECHANICAL_RESUME_{stamp}.json", repair)
        original_sources, original_write = scan.source_bindings, scan.write
        # This map is checked above against every original file, not re-frozen.
        scan.source_bindings = lambda: frozen["sources"]
        scan.write = atomic_json
        try:
            scan.run(argparse.Namespace(action="run", role="test", run=run_id,
                     suites=["main", "sliceA", "sliceB"] if classical else list(scan.METHODS),
                     methods=["max_tf", "gmm_omp", "bcs_oracle"] if classical else ["model", "keep", "reference"],
                     workers=1, limit=None, count=None))
        finally:
            scan.source_bindings, scan.write = original_sources, original_write


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--classical", action="store_true")
    args = parser.parse_args()
    run_resume("paper2_1103_test_classical_20261114" if args.classical else "paper2_1103_test_20261114", args.classical)
