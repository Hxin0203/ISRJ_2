param([Parameter(Mandatory=$true)][ValidateSet('original','tail')][string]$Arm)
$ErrorActionPreference = 'Stop'
$experimentRoot = Split-Path -Parent $PSScriptRoot
if ($env:COMPUTERNAME -ne 'DESKTOP-RFFERM3' -or $experimentRoot -ne 'D:\code\Xin\ISRJ_222') {
    throw 'Server B and dedicated ISRJ_222 root required; no local training.'
}
$env:ISRJ2_CORRECTION_ARM = $Arm
$env:CUDA_VISIBLE_DEVICES = '1'
$env:CUDA_DEVICE_ORDER = 'PCI_BUS_ID'
$env:PYTHONUTF8 = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
$experimentPython = Join-Path $experimentRoot '.venv\Scripts\python.exe'
$experimentRun = if ($Arm -eq 'original') { 'correction_original_control_20261117_v1' } else { 'correction_mf_tail_20261117_v1' }
$experimentOutput = Join-Path $experimentRoot "outputs\$experimentRun"
New-Item -ItemType Directory -Force -Path $experimentOutput | Out-Null
$experimentStamp = Get-Date -Format 'yyyyMMdd_HHmmss_ffff'
$experimentProcess = Start-Process -FilePath $experimentPython -ArgumentList '-B -m tools.correction_tail_study.run' -WorkingDirectory $experimentRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $experimentOutput "controller_$experimentStamp.stdout.log") -RedirectStandardError (Join-Path $experimentOutput "controller_$experimentStamp.stderr.log") -PassThru
Write-Output "$Arm CorrectionNet controller PID: $($experimentProcess.Id)"
