$ErrorActionPreference = 'Stop'

$root = 'D:\code\Xin\ISRJ_222'
$python = 'D:\Anaconda\python.exe'
$stateDir = Join-Path $root 'evidence\outputs\recovp_hybrid_8seed_pipeline'
$stdoutPath = Join-Path $stateDir 'controller.stdout.log'
$stderrPath = Join-Path $stateDir 'controller.stderr.log'
$launchPath = Join-Path $stateDir 'LAUNCH.json'

if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Python executable missing: $python"
}
New-Item -ItemType Directory -Path $stateDir -Force | Out-Null

$existing = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -match '^python(w)?\.exe$' -and
    $_.CommandLine -like '*tools.run_recovp_hybrid_8seed_pipeline*'
}
if ($existing) {
    throw "An 8-seed pipeline controller already exists: $($existing.ProcessId -join ',')"
}

$process = Start-Process -FilePath $python `
    -ArgumentList @('-m', 'tools.run_recovp_hybrid_8seed_pipeline', '--poll-seconds', '60', '--retry-seconds', '60') `
    -WorkingDirectory $root `
    -RedirectStandardOutput $stdoutPath `
    -RedirectStandardError $stderrPath `
    -WindowStyle Hidden `
    -PassThru

$launch = [ordered]@{
    stage = 'CONTROLLER_STARTED'
    started_local = (Get-Date).ToString('o')
    hostname = $env:COMPUTERNAME
    repository_root = $root
    pid = $process.Id
    executable = $python
    arguments = @('-m', 'tools.run_recovp_hybrid_8seed_pipeline', '--poll-seconds', '60', '--retry-seconds', '60')
    stdout = $stdoutPath
    stderr = $stderrPath
}
$launch | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $launchPath -Encoding UTF8
Get-Content -LiteralPath $launchPath
