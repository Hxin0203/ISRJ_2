$ErrorActionPreference = 'Stop'
$root = 'D:/code/Xin/ISRJ_222'
$out = Join-Path $root 'outputs/original_paper_replication_256_v1/R3_by_mechanism'
$python = Join-Path $root '.venv/Scripts/python.exe'
if ($env:COMPUTERNAME -ne 'DESKTOP-RFFERM3') { throw 'R3 requires server B.' }
New-Item -ItemType Directory -Force -Path $out | Out-Null
$existing = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -match '^python(w)?\.exe$' -and
    $_.CommandLine -like '*-m tools.original_replication_classical_by_mechanism queue*'
}
if ($existing) { exit 0 }
$created = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
    CommandLine = "powershell.exe -NoProfile -WindowStyle Hidden -Command `"Set-Location '$root'; & '$python' -u -m tools.original_replication_classical_by_mechanism queue 1>> '$out/controller.stdout.log' 2>> '$out/controller.stderr.log'`""
    CurrentDirectory = $root
}
if ($created.ReturnValue -ne 0) { throw 'R3 mechanism recovery launch failed.' }
[ordered]@{launcher_pid=$created.ProcessId;launched_utc=[DateTime]::UtcNow.ToString('o')} |
    ConvertTo-Json | Set-Content -Encoding UTF8 (Join-Path $out 'LAUNCH_RECEIPT.json')
Get-Content (Join-Path $out 'LAUNCH_RECEIPT.json') -Raw
