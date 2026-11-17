$ErrorActionPreference = 'Stop'
$experimentRoot = Split-Path -Parent $PSScriptRoot
if ($env:COMPUTERNAME -ne 'DESKTOP-RFFERM3' -or $experimentRoot -ne 'D:\code\Xin\ISRJ_222') {
    throw 'Server-only launcher. Deploy to rsdr-b:D:/code/Xin/ISRJ_222; local training is prohibited.'
}
$env:CUDA_VISIBLE_DEVICES = '1'
$env:PYTHONUTF8 = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
$experimentPython = 'D:\code\Xin\ISRJ_222\.venv\Scripts\python.exe'
$experimentOutput = Join-Path $experimentRoot 'outputs\unet_mf_tail_screen_20261117_v2'
New-Item -ItemType Directory -Force -Path $experimentOutput | Out-Null
$experimentStamp = Get-Date -Format 'yyyyMMdd_HHmmss_ffff'
$experimentProcess = Start-Process -FilePath $experimentPython -ArgumentList '-m tools.unet_tail_loss.run' -WorkingDirectory $experimentRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $experimentOutput "controller_$experimentStamp.stdout.log") -RedirectStandardError (Join-Path $experimentOutput "controller_$experimentStamp.stderr.log") -PassThru
Write-Output "U-Net matched-filter-tail controller PID: $($experimentProcess.Id)"
