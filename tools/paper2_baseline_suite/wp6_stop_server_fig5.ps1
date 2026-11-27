$ErrorActionPreference = 'Stop'
# Stop only this task's own queued WP6 fig5 launcher on server B (superseded by the local export).
if ($env:COMPUTERNAME -ne 'DESKTOP-RFFERM3') { throw 'Server B only.' }
Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -match 'tools\.paper2_baseline_suite\.wp6_fig5_infer' } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force; Write-Output "stopped own fig5 pid=$($_.ProcessId)" }
