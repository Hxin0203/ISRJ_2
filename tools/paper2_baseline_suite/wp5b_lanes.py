"""Read-only: last lines of WP5b lane/helper logs, pending jobs, eval child processes."""
import subprocess
from pathlib import Path

W = Path("D:/code/Xin/ISRJ_222/outputs/paper2_baseline_suite_20261126/WP5b")
for log in sorted(W.glob("*.log")):
    lines = log.read_text(encoding="utf-8", errors="replace").strip().splitlines()
    if lines:
        print(f"{log.name}: {' | '.join(lines[-3:])[:300]}")
import sys
sys.path.insert(0, "D:/code/Xin/ISRJ_222")
from tools.paper2_baseline_suite import wp5b_suite as S
for m, s in S.jobs():
    tag = f"seed_{s}" if s is not None else "single"
    d = S.OUT / "results" / m / tag
    if not (d / "DONE.json").is_file():
        print("PENDING", m, s, "cells", len(list(d.glob("cell_*.npz"))) if d.is_dir() else 0)
ps = ("Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { $_.CommandLine -match 'wp5b_suite eval|wp5b_helper' } | "
      "ForEach-Object { '{0} {1} {2}' -f $_.ProcessId, $_.CreationDate.ToString('HH:mm:ss'), "
      "($_.CommandLine -replace '\\s+',' ').Substring([Math]::Max(0, $_.CommandLine.Length - 90)) }")
print(subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True).stdout)
