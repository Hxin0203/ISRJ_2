$ErrorActionPreference = 'Stop'
$root = 'D:/code/Xin/ISRJ_222'
$python = Join-Path $root '.venv/Scripts/python.exe'
$state = Join-Path $root 'evidence/outputs/recovp_hybrid_continue512_20261127'
if ($env:COMPUTERNAME -ne 'DESKTOP-RFFERM3') {
    throw 'The continuation controller must run on server B.'
}
New-Item -ItemType Directory -Force -Path $state | Out-Null
$existing = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -match '^python(w)?\.exe$' -and
    $_.CommandLine -like '*-m tools.run_recovp_hybrid_continue512*'
}
if ($existing) {
    $existing | Select-Object ProcessId, CommandLine | ConvertTo-Json
    exit 0
}
# Create the detached controller through Windows management, independent of SSH.
$created = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
    CommandLine = "powershell.exe -NoProfile -WindowStyle Hidden -Command `"Set-Location '$root'; & '$python' -m tools.run_recovp_hybrid_continue512 1>> '$state/controller.stdout.log' 2>> '$state/controller.stderr.log'`""
    CurrentDirectory = $root
}
if ($created.ReturnValue -ne 0) {
    throw "Controller process creation failed with return code $($created.ReturnValue)"
}
[ordered]@{
    stage = 'CONTROLLER_LAUNCHED'
    pid = $created.ProcessId
    started_utc = [DateTime]::UtcNow.ToString('o')
    source = 'tools/run_recovp_hybrid_continue512.py'
    state = $state
} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $state 'LAUNCH.json') -Encoding UTF8
Get-Content -LiteralPath (Join-Path $state 'LAUNCH.json') -Raw
