$ErrorActionPreference = 'Stop'
# Stop only this task's own WP6 fig5 launcher (matched by its module name), then relaunch with logs.
if ($env:COMPUTERNAME -ne 'DESKTOP-RFFERM3') { throw 'Server B only.' }
$own = Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match 'tools\.paper2_baseline_suite\.wp6_fig5_infer' }
foreach ($p in $own) { Stop-Process -Id $p.ProcessId -Force; Write-Output "stopped own fig5 pid=$($p.ProcessId)" }
Start-Sleep -Seconds 2
& 'D:\code\Xin\ISRJ_222\tools\paper2_baseline_suite\launch_wp6_fig5_wmi.ps1'
