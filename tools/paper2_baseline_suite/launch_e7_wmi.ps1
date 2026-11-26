$ErrorActionPreference = 'Stop'
# E7 CVEDNet frozen evaluation; waits for a GPU below 85% and runs WMI-owned (outlives SSH).
$root = 'D:\code\Xin\ISRJ_222'
if ($env:COMPUTERNAME -ne 'DESKTOP-RFFERM3') { throw 'Server B only.' }
$log = Join-Path $root 'outputs\paper2_baseline_suite_20261126\E7'
New-Item -ItemType Directory -Force -Path $log | Out-Null
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$cmd = "cmd.exe /c set PYTHONUTF8=1& set PYTHONDONTWRITEBYTECODE=1& cd /d $root && .venv\Scripts\python.exe -m tools.paper2_baseline_suite.external_suite launch > `"$log\launch_$stamp.stdout.log`" 2> `"$log\launch_$stamp.stderr.log`""
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine = $cmd; CurrentDirectory = $root}
Write-Output "E7 launch return=$($r.ReturnValue) pid=$($r.ProcessId)"
