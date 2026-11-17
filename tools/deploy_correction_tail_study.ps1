$ErrorActionPreference = 'Stop'
$experimentRoot = Split-Path -Parent $PSScriptRoot
if ($env:COMPUTERNAME -ne 'DESKTOP-RFFERM3' -or $experimentRoot -ne 'D:\code\Xin\ISRJ_222') {
    throw 'Server B and dedicated workspace required.'
}
$experimentPython = Join-Path $experimentRoot '.venv\Scripts\python.exe'
$env:PYTHONUTF8 = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
Push-Location $experimentRoot
try {
    & $experimentPython -B -m tools.correction_tail_study.operations engineering --arm original
    if ($LASTEXITCODE -ne 0) { throw 'Original arm CPU engineering failed' }
    & $experimentPython -B -m tools.correction_tail_study.operations engineering --arm tail
    if ($LASTEXITCODE -ne 0) { throw 'Tail arm CPU engineering failed' }
    $experimentPause = Join-Path $experimentRoot 'outputs\unet_dilation_screen_20261117_v1\PAUSE'
    $experimentOwner = 'ISRJ2_CORRECTION_TAIL_STUDY_20261117_V1'
    $experimentDilationState = Get-Content (Join-Path (Split-Path $experimentPause) 'STATUS.json') -Raw | ConvertFrom-Json
    if ($experimentDilationState.stage -ne 'WAIT_DEPENDENCY') { throw 'Dilation is no longer waiting; refuse priority change.' }
    if (Test-Path -LiteralPath $experimentPause) { throw 'Existing PAUSE retained; inspect before deployment.' }
    foreach ($experimentRun in @('correction_original_control_20261117_v1','correction_mf_tail_20261117_v1')) {
        if (Test-Path -LiteralPath (Join-Path $experimentRoot "outputs\$experimentRun\QUEUE_BINDING.json")) {
            throw 'Queue already registered; refuse duplicate deployment.'
        }
    }
    [System.IO.File]::WriteAllText($experimentPause, $experimentOwner, [System.Text.UTF8Encoding]::new($false))
    foreach ($experimentArm in @('original','tail')) {
        $experimentCommand = "powershell.exe -NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File `"$experimentRoot\tools\start_correction_tail_study.ps1`" -Arm $experimentArm"
        $experimentResult = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine=$experimentCommand; CurrentDirectory=$experimentRoot}
        if ($experimentResult.ReturnValue -ne 0) { throw "WMI launch failed: $experimentArm" }
        Write-Output "$experimentArm launcher PID: $($experimentResult.ProcessId)"
    }
} finally {
    Pop-Location
}
