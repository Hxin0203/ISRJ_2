"""Bounded, fail-closed SHA-256 reads for long-running controllers."""
from __future__ import annotations

import hashlib
from pathlib import Path


def sha(path: Path) -> str:
    target = Path(path)
    for attempt in range(3):
        size = target.stat().st_size
        digest = hashlib.sha256()
        with target.open("rb") as handle:
            remaining = size
            while remaining:
                chunk = handle.read(min(64 * 1024, remaining))
                if not chunk:
                    raise IOError(f"short read while hashing {target}")
                digest.update(chunk)
                remaining -= len(chunk)
            if handle.read(1):
                raise IOError(f"file grew while hashing {target}")
        if target.stat().st_size == size:
            return digest.hexdigest()
        if attempt == 2:
            raise IOError(f"file changed while hashing {target}")
    raise AssertionError("unreachable")
