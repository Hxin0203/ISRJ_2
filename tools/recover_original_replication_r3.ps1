$ErrorActionPreference = 'Stop'
$root = 'D:\code\Xin\ISRJ_222'
$python = Join-Path $root '.venv\Scripts\python.exe'
$out = Join-Path $root 'outputs\original_paper_replication_256_v1\R3_main'
$log = Join-Path $out 'r3_recovery_controller.log'
if ($env:COMPUTERNAME -ne 'DESKTOP-RFFERM3') { throw 'R3 recovery must run on server B.' }
New-Item -ItemType Directory -Force -Path $out | Out-Null
$existing = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -match '^python(w)?\.exe$' -and $_.CommandLine -like '*tools.original_replication_suite queue*'
}
if ($existing) { exit 0 }
while ($true) {
    $workers = Get-CimInstance Win32_Process | Where-Object {
        $_.Name -match '^python(w)?\.exe$' -and $_.CommandLine -like '*tools.original_replication_suite eval*'
    }
    if (-not $workers) { break }
    Start-Sleep -Seconds 60
}
Set-Location $root
& $python -u -m tools.original_replication_suite queue >> $log 2>&1
