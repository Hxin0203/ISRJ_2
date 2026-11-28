$ErrorActionPreference = 'Stop'
$root = 'D:\code\Xin\ISRJ_222'
$python = Join-Path $root '.venv\Scripts\python.exe'
$state = Join-Path $root 'outputs\original_paper_replication_256_v1\pipeline'
if ($env:COMPUTERNAME -ne 'DESKTOP-RFFERM3') { throw 'The post-priority pipeline must run on server B.' }
New-Item -ItemType Directory -Force -Path $state | Out-Null
$existing = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -match '^python(w)?\.exe$' -and $_.CommandLine -like '*tools.original_replication_postpriority*'
}
if ($existing) { exit 0 }
$cmd = "powershell.exe -NoProfile -WindowStyle Hidden -Command `"Set-Location '$root'; while(-not (Test-Path '$root\\outputs\\original_paper_replication_256_v1\\R3_main\\ALL_COMPLETE.json')){ Start-Sleep -Seconds 60 }; & '$python' -u -m tools.original_replication_r1 1>> '$state\\r1.stdout.log' 2>> '$state\\r1.stderr.log'`""
$created = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = $cmd; CurrentDirectory = $root }
[ordered]@{ pid = $created.ProcessId; return_code = $created.ReturnValue; launched_utc = [DateTime]::UtcNow.ToString('o') } |
    ConvertTo-Json | Set-Content -Encoding UTF8 (Join-Path $state 'R1_LAUNCH_RECEIPT.json')
Get-Content (Join-Path $state 'R1_LAUNCH_RECEIPT.json') -Raw
