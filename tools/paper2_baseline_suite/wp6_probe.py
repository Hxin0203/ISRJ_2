"""Read-only: WP6/WP5b process list and GPU load on server B."""
import subprocess

print(subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used", "--format=csv,noheader"],
                     capture_output=True, text=True).stdout.strip())
ps = ("Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { $_.CommandLine -match 'wp6|wp5b' } | "
      "ForEach-Object { '{0} {1} {2}' -f $_.ProcessId, $_.CreationDate.ToString('HH:mm:ss'), "
      "($_.CommandLine -replace '\\s+',' ').Substring([Math]::Max(0, $_.CommandLine.Length - 110)) }")
print(subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True).stdout)
