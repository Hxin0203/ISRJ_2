$ErrorActionPreference = 'Stop'
# WP5b jammer-structure sweep lanes; WMI-owned so they outlive SSH. Server B only.
# GPU lanes wait (60 s polls) until a GPU is below 85 percent utilization.
$root = 'D:\code\Xin\ISRJ_222'
if ($env:COMPUTERNAME -ne 'DESKTOP-RFFERM3') { throw 'Server B only.' }
$log = Join-Path $root 'outputs\paper2_baseline_suite_20261126\WP5b'
New-Item -ItemType Directory -Force -Path $log | Out-Null
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
foreach ($lane in @('cpu', 'gpuA', 'gpuB')) {
    $cmd = "cmd.exe /c set PYTHONUTF8=1& set PYTHONDONTWRITEBYTECODE=1& cd /d $root && .venv\Scripts\python.exe -m tools.paper2_baseline_suite.wp5b_suite queue --lane $lane > `"$log\queue_${lane}_$stamp.stdout.log`" 2> `"$log\queue_${lane}_$stamp.stderr.log`""
    $r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine = $cmd; CurrentDirectory = $root}
    Write-Output "WP5b lane=$lane return=$($r.ReturnValue) pid=$($r.ProcessId)"
}
