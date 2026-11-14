"""Local durable writes and OS-owned locks for long Windows experiments.

These are engineering operations only; no model, scene or metric definitions.
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from pathlib import Path


def process_alive(pid):
    """Read-only liveness check; never signal a Windows process."""
    if not pid:
        return False
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        api = ctypes.WinDLL("kernel32", use_last_error=True)
        api.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        api.OpenProcess.restype = wintypes.HANDLE
        handle = api.OpenProcess(0x1000, False, int(pid))
        if not handle:
            return False
        try:
            code = wintypes.DWORD()
            api.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
            return bool(api.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == 259
        finally:
            api.CloseHandle.argtypes = [wintypes.HANDLE]
            api.CloseHandle(handle)
    return Path(f"/proc/{int(pid)}").exists()


def replace_retry(source, target, attempts=12):
    for attempt in range(attempts):
        try:
            os.replace(source, target)
            return
        except OSError as exc:
            if (getattr(exc, "winerror", None) not in (5, 32, 33)
                    and not isinstance(exc, PermissionError)) or attempt == attempts - 1:
                raise
            time.sleep(min(.05 * 2 ** attempt, 1.0))


def atomic_write(path, writer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        replace_retry(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def atomic_json(path, value):
    raw = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    atomic_write(path, lambda stream: stream.write(raw))


def atomic_copy(source, target):
    with Path(source).open("rb") as stream:
        atomic_write(target, lambda dest: shutil.copyfileobj(stream, dest))


class ProcessLock:
    """The OS releases the lock on process exit, even after forced termination.

    The lock file remains as metadata. Never check Windows liveness with os.kill.
    """
    def __init__(self, path):
        self.path = Path(path)
        self.stream = None

    def acquire(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        stream = self.path.open("a+b")
        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            stream.write(b"\0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            stream.close()
            raise RuntimeError(f"another process holds {self.path}") from exc
        self.stream = stream
        stream.seek(1)
        stream.truncate()
        stream.write(json.dumps({"pid": os.getpid(), "acquired_unix": time.time()}).encode())
        stream.flush()
        os.fsync(stream.fileno())
        return self

    def release(self):
        if self.stream is not None:
            self.stream.close()
            self.stream = None

    def __enter__(self):
        return self.acquire()

    def __exit__(self, *_):
        self.release()
