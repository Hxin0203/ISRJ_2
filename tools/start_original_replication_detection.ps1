$ErrorActionPreference = 'Stop'
$root = 'D:/code/Xin/ISRJ_222'
$state = Join-Path $root 'outputs/original_paper_replication_256_v1/R5_detection'
$python = Join-Path $root '.venv/Scripts/python.exe'
if ($env:COMPUTERNAME -ne 'DESKTOP-RFFERM3') { throw 'R5 requires server B.' }
New-Item -ItemType Directory -Force -Path $state | Out-Null
$existing = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -match '^python(w)?\.exe$' -and
    $_.CommandLine -like '*-m tools.original_replication_detection queue*'
}
if ($existing) {
    $existing | Select-Object ProcessId, CommandLine | ConvertTo-Json
    exit 0
}
$created = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
    CommandLine = "powershell.exe -NoProfile -WindowStyle Hidden -Command `"Set-Location '$root'; & '$python' -u -m tools.original_replication_detection queue 1>> '$state/controller.stdout.log' 2>> '$state/controller.stderr.log'`""
    CurrentDirectory = $root
}
if ($created.ReturnValue -ne 0) { throw "R5 launch failed with code $($created.ReturnValue)" }
[ordered]@{
    launcher_pid = $created.ProcessId
    launched_utc = [DateTime]::UtcNow.ToString('o')
    controller_sha256 = (Get-FileHash (Join-Path $root 'tools/original_replication_detection.py')).Hash.ToLower()
    policy = 'GPU utilization <85% and host physical memory <90%; both GPUs support concurrent jobs'
} | ConvertTo-Json | Set-Content -Encoding UTF8 (Join-Path $state 'LAUNCH_RECEIPT.json')
Get-Content (Join-Path $state 'LAUNCH_RECEIPT.json') -Raw
