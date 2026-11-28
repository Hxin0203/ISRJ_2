$ErrorActionPreference = 'Stop'
$root = 'D:\code\Xin\ISRJ_222'
$python = Join-Path $root '.venv\Scripts\python.exe'
$state = Join-Path $root 'outputs\original_paper_replication_256_v1\pipeline'
$log = Join-Path $state 'postprocessing_recovery.log'
if ($env:COMPUTERNAME -ne 'DESKTOP-RFFERM3') { throw 'Postprocessing recovery must run on server B.' }
$existing = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -match '^powershell\.exe$' -and $_.CommandLine -like '*recover_original_replication_postprocessing.ps1*'
}
if ($existing) { exit 0 }
while (-not (Test-Path (Join-Path $root 'outputs\original_paper_replication_256_v1\R3_main\ALL_COMPLETE.json'))) {
    Start-Sleep -Seconds 60
}
Set-Location $root
& $python -u -m tools.original_replication_suite summarize >> $log 2>&1
