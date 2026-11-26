$ErrorActionPreference = 'Continue'
# Read-only status of the baseline suite on server B (no process is touched).
$root = 'D:\code\Xin\ISRJ_222\outputs\paper2_baseline_suite_20261126'
$cpu = Get-CimInstance Win32_Processor
$os = Get-CimInstance Win32_OperatingSystem
"cores={0} cpu_load%={1} free_GB={2} total_GB={3}" -f ($cpu | Measure-Object NumberOfLogicalProcessors -Sum).Sum, ($cpu | Measure-Object LoadPercentage -Average).Average, [math]::Round($os.FreePhysicalMemory/1MB,1), [math]::Round($os.TotalVisibleMemorySize/1MB,1)
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
if (Test-Path "$root\E3\QUEUE_STATUS.json") { "E3 queue: " + (Get-Content "$root\E3\QUEUE_STATUS.json" -Raw) }
Get-ChildItem "$root\E3" -Recurse -Filter ROW.json -ErrorAction SilentlyContinue | ForEach-Object { "E3 row: " + $_.Directory.Parent.Name + "/" + $_.Directory.Name }
Get-ChildItem "$root\E3" -Recurse -Filter FAILED.json -ErrorAction SilentlyContinue | ForEach-Object { "E3 FAILED: " + $_.FullName }
foreach ($m in 'max_tf','gmm_omp','bcs_oracle') {
  $d = "$root\E6\dev256\$m"; if (Test-Path $d) { "E6 dev {0}: chunks={1} done={2}" -f $m, (Get-ChildItem $d -Filter 'chunk_*.npz' | Where-Object { $_.Name -notlike '*.tmp.npz' }).Count, (Test-Path "$d\SUMMARY.json") }
  $d = "$root\E6\equal_pfa\$m"; if (Test-Path $d) { "E6 eqpfa {0}: parts={1} done={2}" -f $m, (Get-ChildItem $d -Filter '*_*.npz' | Where-Object { $_.Name -notlike '*.tmp.npz' -and $_.Name -ne 'SCORES.npz' }).Count, (Test-Path "$d\ROW.json") }
}
Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -like '*paper2_baseline_suite*' } | ForEach-Object { "proc {0}: {1}" -f $_.ProcessId, ($_.CommandLine -replace '.*-m ','') }
