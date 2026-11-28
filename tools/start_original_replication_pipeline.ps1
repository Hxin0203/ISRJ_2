$ErrorActionPreference = 'Stop'
$root = 'D:\code\Xin\ISRJ_222'
$python = Join-Path $root '.venv\Scripts\python.exe'
$state = Join-Path $root 'outputs\original_paper_replication_256_v1\pipeline'
if ($env:COMPUTERNAME -ne 'DESKTOP-RFFERM3') { throw 'The replication pipeline must run on server B.' }
New-Item -ItemType Directory -Force -Path $state | Out-Null
$existing = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -match '^python(w)?\.exe$' -and $_.CommandLine -like '*tools.original_replication_pipeline*'
}
if ($existing) {
    $existing | Select-Object ProcessId,CommandLine | ConvertTo-Json
    exit 0
}
$created = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
    CommandLine = "powershell.exe -NoProfile -WindowStyle Hidden -Command `"Set-Location '$root'; & '$python' -u -m tools.original_replication_pipeline 1>> '$state/pipeline.stdout.log' 2>> '$state/pipeline.stderr.log'`""
    CurrentDirectory = $root
}
if ($created.ReturnValue -ne 0) { throw "Pipeline creation failed with return code $($created.ReturnValue)" }
[ordered]@{ pid = $created.ProcessId; launched_utc = [DateTime]::UtcNow.ToString('o') } |
    ConvertTo-Json | Set-Content -Encoding UTF8 (Join-Path $state 'LAUNCH_RECEIPT.json')
Get-Content -LiteralPath (Join-Path $state 'LAUNCH_RECEIPT.json') -Raw
