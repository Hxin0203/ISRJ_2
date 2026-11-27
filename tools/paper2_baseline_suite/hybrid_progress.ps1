$ErrorActionPreference = 'Continue'
# Read-only: Hybrid seed training progress and controller heartbeat on server B.
$root = 'D:\code\Xin\ISRJ_222\outputs'
foreach ($s in 17,29,41,53,67,79,97,109) {
  $d = "$root\recovp_hybrid_seed$s\seed_$s"
  $c = Test-Path "$d\TRAINING_COMPLETE.json"
  $l = Get-ChildItem $d -Filter 'training_epoch_*.json' -ErrorAction SilentlyContinue | Sort-Object Name | Select-Object -Last 1
  "seed $s complete=$c last=$($l.Name) at=$($l.LastWriteTime)"
}
$st = Get-Content "$root\paper2_baseline_suite_20261126\CONTROLLER_STATUS.json" -Raw | ConvertFrom-Json
"controller stage=$($st.stage) heartbeat=$($st.heartbeat_utc) included=$($st.hybrid_seeds_included -join ',')"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match 'recovp|paper2_baseline_suite' } | ForEach-Object { "proc $($_.ProcessId): " + ($_.CommandLine -replace '.*-m ','' -replace '.*\\tools\\','tools\') }
