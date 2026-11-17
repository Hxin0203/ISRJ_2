$ErrorActionPreference = 'Stop'
$experimentRoot = Split-Path -Parent $PSScriptRoot
if ($env:COMPUTERNAME -ne 'DESKTOP-RFFERM3' -or $experimentRoot -ne 'D:\code\Xin\ISRJ_222') {
    throw 'Server B and dedicated ISRJ_222 root required; no local training.'
}
$env:CUDA_VISIBLE_DEVICES = '0'
$env:CUDA_DEVICE_ORDER = 'PCI_BUS_ID'
$env:PYTHONUTF8 = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
$experimentPython = Join-Path $experimentRoot '.venv\Scripts\python.exe'
$experimentOutput = Join-Path $experimentRoot 'outputs\unet_noise_norm_screen_20261117_v1'
New-Item -ItemType Directory -Force -Path $experimentOutput | Out-Null
$experimentStamp = Get-Date -Format 'yyyyMMdd_HHmmss_ffff'
$experimentProcess = Start-Process -FilePath $experimentPython -ArgumentList '-m tools.unet_norm_screen.run' -WorkingDirectory $experimentRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $experimentOutput "controller_$experimentStamp.stdout.log") -RedirectStandardError (Join-Path $experimentOutput "controller_$experimentStamp.stderr.log") -PassThru
Write-Output "GPU0 normalization controller PID: $($experimentProcess.Id)"
