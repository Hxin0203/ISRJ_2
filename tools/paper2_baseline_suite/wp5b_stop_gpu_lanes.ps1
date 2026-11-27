$ErrorActionPreference = 'Stop'
# Stop only this task's own WP5b schedulers on server B (gpuA/gpuB lanes and the helper lane).
# Running ``wp5b_suite eval`` children are NOT stopped; they finish their job and write DONE.json.
if ($env:COMPUTERNAME -ne 'DESKTOP-RFFERM3') { throw 'Server B only.' }
$all = Get-CimInstance Win32_Process -Filter "Name='python.exe'"
$own = $all | Where-Object { $_.CommandLine -match 'wp5b_suite queue --lane gpu[AB]' -or $_.CommandLine -match 'wp5b_helper_lane' }
foreach ($p in $own) { Stop-Process -Id $p.ProcessId -Force; Write-Output "stopped scheduler pid=$($p.ProcessId)" }
$all | Where-Object { $_.CommandLine -match 'wp5b_suite eval' } | ForEach-Object { Write-Output "left running eval pid=$($_.ProcessId)" }
