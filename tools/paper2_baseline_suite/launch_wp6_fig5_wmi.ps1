$ErrorActionPreference = 'Stop'
# WP6 Fig. 5 inference (3 DEV parents); WMI-owned so it outlives SSH. Server B only.
$root = 'D:\code\Xin\ISRJ_222'
if ($env:COMPUTERNAME -ne 'DESKTOP-RFFERM3') { throw 'Server B only.' }
$log = Join-Path $root 'outputs\paper2_baseline_suite_20261126\WP6'
New-Item -ItemType Directory -Force -Path $log | Out-Null
Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match 'wp6_fig5_infer' } |
    ForEach-Object { Write-Output "already running pid=$($_.ProcessId)"; exit 0 }
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$cmd = "cmd.exe /c set PYTHONUTF8=1& set PYTHONDONTWRITEBYTECODE=1& cd /d $root && .venv\Scripts\python.exe -m tools.paper2_baseline_suite.wp6_fig5_infer launch > `"$log\fig5_$stamp.stdout.log`" 2> `"$log\fig5_$stamp.stderr.log`""
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine = $cmd; CurrentDirectory = $root}
Write-Output "WP6 fig5 return=$($r.ReturnValue) pid=$($r.ProcessId)"
