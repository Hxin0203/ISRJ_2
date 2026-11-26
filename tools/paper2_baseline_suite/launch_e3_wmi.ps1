$ErrorActionPreference = 'Stop'
# Launch start_e3.ps1 from a WMI-owned process so the queue outlives the SSH session.
$cmd = 'powershell.exe -NoProfile -ExecutionPolicy Bypass -File D:\code\Xin\ISRJ_222\tools\paper2_baseline_suite\start_e3.ps1'
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine = $cmd; CurrentDirectory = 'D:\code\Xin\ISRJ_222'}
Write-Output "WMI launcher return=$($r.ReturnValue) pid=$($r.ProcessId)"
