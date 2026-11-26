$ErrorActionPreference = 'Stop'

$root = 'D:\code\Xin\ISRJ_222'
$python = Join-Path $root '.venv\Scripts\python.exe'
$stateDir = Join-Path $root 'evidence\outputs\recovp_hybrid_split_gpu_pipeline'
$launchPath = Join-Path $stateDir 'LAUNCH.json'

if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Python executable missing: $python"
}
New-Item -ItemType Directory -Path $stateDir -Force | Out-Null

$existing = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -match '^python(w)?\.exe$' -and
    $_.CommandLine -like '*tools.run_recovp_hybrid_split_pipeline*'
}
if ($existing) {
    throw "A split GPU pipeline controller already exists: $($existing.ProcessId -join ',')"
}

$wrapper = Join-Path $root 'tools\run_recovp_hybrid_split_pipeline.cmd'
$commandLine = "cmd.exe /d /c $wrapper"
$created = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
    CommandLine = $commandLine
    CurrentDirectory = $root
}
if ($created.ReturnValue -ne 0 -or -not $created.ProcessId) {
    throw "Windows process manager could not create the split pipeline (return=$($created.ReturnValue))"
}

$launch = [ordered]@{
    stage = 'SPLIT_GPU_PIPELINE_STARTED'
    started_local = (Get-Date).ToString('o')
    hostname = $env:COMPUTERNAME
    repository_root = $root
    pid = $created.ProcessId
    process_creation = 'Win32_Process.Create'
    command_line = $commandLine
    executable = $python
    arguments = @('-m', 'tools.run_recovp_hybrid_split_pipeline', '--poll-seconds', '60', '--retry-seconds', '60')
    gpu_lanes = @{ '1' = @(53, 67, 79); '0' = @(97, 109) }
    gpu0_gate = 'physical GPU 0 utilization below 85 percent before each seed dispatch; concurrent use allowed'
    stdout = (Join-Path $stateDir 'controller.stdout.log')
}
$launch | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $launchPath -Encoding UTF8
Get-Content -LiteralPath $launchPath
