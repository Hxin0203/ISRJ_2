$ErrorActionPreference = 'Stop'
$root = 'D:\code\Xin\ISRJ_222'
$python = Join-Path $root '.venv\Scripts\python.exe'
$state = Join-Path $root 'outputs\original_paper_replication_256_v1\pipeline'
$log = Join-Path $state 'postprocessing_recovery.log'
if ($env:COMPUTERNAME -ne 'DESKTOP-RFFERM3') { throw 'Postprocessing recovery must run on server B.' }
$existing = Get-CimInstance Win32_Process | Where-Object {
    $_.ProcessId -ne $PID -and
    (($_.Name -match '^powershell\.exe$' -and $_.CommandLine -like '*-File*recover_original_replication_postprocessing.ps1*') -or
     ($_.Name -match '^python(w)?\.exe$' -and $_.CommandLine -like '*-m tools.original_replication_mechanism_summary watch*'))
}
if ($existing) { exit 0 }
Set-Location $root
& $python -u -m tools.original_replication_mechanism_summary watch >> $log 2>&1
if ($LASTEXITCODE -ne 0) { throw "Mechanism postprocessing exited with $LASTEXITCODE" }
