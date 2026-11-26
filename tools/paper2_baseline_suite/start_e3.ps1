$ErrorActionPreference = 'Stop'
# E3/E10 unified equal-Pfa queue. Server B only; GPU chosen per job at <85% utilization.
$root = 'D:\code\Xin\ISRJ_222'
if ($env:COMPUTERNAME -ne 'DESKTOP-RFFERM3' -or -not (Test-Path "$root\tools\paper2_baseline_suite\equal_pfa_suite.py")) {
    throw 'Server B and dedicated ISRJ_222 root required; no local execution.'
}
$out = Join-Path $root 'outputs\paper2_baseline_suite_20261126\E3'
New-Item -ItemType Directory -Force -Path $out | Out-Null
$env:PYTHONUTF8 = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$py = Join-Path $root '.venv\Scripts\python.exe'
$p = Start-Process -FilePath $py -ArgumentList '-m tools.paper2_baseline_suite.equal_pfa_suite queue' `
    -WorkingDirectory $root -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput (Join-Path $out "queue_$stamp.stdout.log") `
    -RedirectStandardError (Join-Path $out "queue_$stamp.stderr.log")
Set-Content -Path (Join-Path $out 'QUEUE_PID.txt') -Value "$($p.Id) $stamp"
Write-Output "E3 queue PID: $($p.Id)"
