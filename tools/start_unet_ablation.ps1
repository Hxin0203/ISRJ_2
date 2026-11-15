$ErrorActionPreference = 'Stop'
$experimentRoot = Split-Path -Parent $PSScriptRoot
if ($env:COMPUTERNAME -ne 'DESKTOP-RFFERM3' -or $experimentRoot -ne 'D:\code\Xin\ISRJ_222') {
    throw 'Server-only launcher. Deploy to rsdr-b:D:/code/Xin/ISRJ_222; local training is prohibited.'
}
$env:CUDA_VISIBLE_DEVICES = '1'
$env:PYTHONUTF8 = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
$experimentPython = 'D:\code\Xin\ISRJ_222\.venv\Scripts\python.exe'
$experimentOutput = Join-Path $experimentRoot 'outputs\unet_ablation_3seed_256_20261115'
New-Item -ItemType Directory -Force -Path $experimentOutput | Out-Null
$experimentStamp = Get-Date -Format 'yyyyMMdd_HHmmss_ffff'
$experimentProcess = Start-Process -FilePath $experimentPython -ArgumentList '-m tools.unet_ablation.run' -WorkingDirectory $experimentRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $experimentOutput "controller_$experimentStamp.stdout.log") -RedirectStandardError (Join-Path $experimentOutput "controller_$experimentStamp.stderr.log") -PassThru
Write-Output "U-Net/ablation controller PID: $($experimentProcess.Id)"
