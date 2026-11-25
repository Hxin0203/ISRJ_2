# Start outside the SSH session's process job so disconnecting keeps the run alive.
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$taskRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
if ($env:COMPUTERNAME -ine 'DESKTOP-RFFERM3' -or
    $taskRoot.TrimEnd('\') -ine 'D:\code\Xin\ISRJ_222') {
    throw 'Post-training dispatch requires server B and D:/code/Xin/ISRJ_222'
}
Set-Location -LiteralPath $taskRoot

$existingControllers = @(Get-CimInstance Win32_Process | Where-Object {
    $_.Name -match '^python' -and
    $_.CommandLine -match ' -m tools\.finish_recovp_hybrid_seed17(?:\s|$)'
})
if ($existingControllers.Count -gt 0) {
    throw 'An existing controller must be preserved; inspect POSTTRAINING_STATUS.json'
}

$taskStamp = [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssZ')
$taskEvidence = Join-Path $taskRoot 'evidence\outputs\recovp_hybrid_seed17'
$null = New-Item -ItemType Directory -Path $taskEvidence -Force
$taskOut = Join-Path $taskEvidence ('controller_' + $taskStamp + '.stdout.log')
$taskErr = Join-Path $taskEvidence ('controller_' + $taskStamp + '.stderr.log')
$taskPython = Join-Path $taskRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $taskPython -PathType Leaf)) {
    throw 'Server experiment virtual environment is missing'
}

# WMI owns this hidden process, independently of the short-lived SSH command.
$taskStartup = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{
    ShowWindow = [uint16]0
    CreateFlags = [uint32]8 # DETACHED_PROCESS; no inherited SSH console.
}
$taskCommand = 'cmd.exe /d /s /c ""{0}" -B -m tools.finish_recovp_hybrid_seed17 1>"{1}" 2>"{2}""' -f $taskPython, $taskOut, $taskErr
$taskCreated = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
    CommandLine = $taskCommand
    CurrentDirectory = $taskRoot
    ProcessStartupInformation = $taskStartup
}
if ($taskCreated.ReturnValue -ne 0) {
    throw ('Independent process creation failed: ' + $taskCreated.ReturnValue)
}
$taskReceipt = [ordered]@{
    stage = 'POSTTRAINING_LAUNCH_REQUESTED'
    launched_utc = [DateTime]::UtcNow.ToString('o')
    method = 'hidden WMI process; independent of SSH session'
    wrapper_pid = $taskCreated.ProcessId
    root = $taskRoot
    launcher_source_sha256 = (Get-FileHash -LiteralPath $PSCommandPath -Algorithm SHA256).Hash.ToLowerInvariant()
    stdout_path = $taskOut
    stderr_path = $taskErr
}
$taskJson = $taskReceipt | ConvertTo-Json
[IO.File]::WriteAllText((Join-Path $taskEvidence 'POSTTRAINING_LAUNCH.json'), $taskJson, [Text.UTF8Encoding]::new($false))
$taskJson
