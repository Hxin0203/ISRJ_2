$ErrorActionPreference = 'Stop'
# Start the baseline-suite controller as a WMI-owned process (outlives SSH). Server B only.
$root = 'D:\code\Xin\ISRJ_222'
if ($env:COMPUTERNAME -ne 'DESKTOP-RFFERM3') { throw 'Server B only.' }
$log = Join-Path $root 'outputs\paper2_baseline_suite_20261126\controller_logs'
New-Item -ItemType Directory -Force -Path $log | Out-Null
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$cmd = "cmd.exe /c set PYTHONUTF8=1& set PYTHONDONTWRITEBYTECODE=1& set CUDA_VISIBLE_DEVICES=& cd /d $root && .venv\Scripts\python.exe -m tools.paper2_baseline_suite.controller > `"$log\controller_$stamp.stdout.log`" 2> `"$log\controller_$stamp.stderr.log`""
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine = $cmd; CurrentDirectory = $root}
Write-Output "controller launch return=$($r.ReturnValue) pid=$($r.ProcessId)"
