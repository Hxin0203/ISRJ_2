param([Parameter(Mandatory=$true)][ValidateSet('dev','equal-pfa')][string]$Action,
      [Parameter(Mandatory=$true)][ValidateSet('max_tf','max_tf_isrj1','gmm_omp','bcs_oracle')][string]$Method)
$ErrorActionPreference = 'Stop'
# E6 classical baselines, server-B CPU only, WMI-owned so they outlive the SSH session.
$root = 'D:\code\Xin\ISRJ_222'
if ($env:COMPUTERNAME -ne 'DESKTOP-RFFERM3') { throw 'Server B only.' }
$log = Join-Path $root "outputs\paper2_baseline_suite_20261126\E6\logs"
New-Item -ItemType Directory -Force -Path $log | Out-Null
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$name = "$Action`_$Method`_$stamp"
$cmd = "cmd.exe /c set CUDA_VISIBLE_DEVICES=& set PYTHONUTF8=1& set PYTHONDONTWRITEBYTECODE=1& set OMP_NUM_THREADS=4& set MKL_NUM_THREADS=4& cd /d $root && .venv\Scripts\python.exe -m tools.paper2_baseline_suite.classical_suite $Action $Method > `"$log\$name.stdout.log`" 2> `"$log\$name.stderr.log`""
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine = $cmd; CurrentDirectory = $root}
Write-Output "$name return=$($r.ReturnValue) pid=$($r.ProcessId)"
