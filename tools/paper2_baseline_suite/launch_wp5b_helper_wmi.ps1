$ErrorActionPreference = 'Stop'
# WP5b helper lane (reverse end of gpuA list); WMI-owned so it outlives SSH. Server B only.
$root = 'D:\code\Xin\ISRJ_222'
if ($env:COMPUTERNAME -ne 'DESKTOP-RFFERM3') { throw 'Server B only.' }
$log = Join-Path $root 'outputs\paper2_baseline_suite_20261126\WP5b'
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$cmd = "cmd.exe /c set PYTHONUTF8=1& set PYTHONDONTWRITEBYTECODE=1& cd /d $root && .venv\Scripts\python.exe -m tools.paper2_baseline_suite.wp5b_helper_lane > `"$log\helper_$stamp.stdout.log`" 2> `"$log\helper_$stamp.stderr.log`""
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine = $cmd; CurrentDirectory = $root}
Write-Output "WP5b helper return=$($r.ReturnValue) pid=$($r.ProcessId)"
