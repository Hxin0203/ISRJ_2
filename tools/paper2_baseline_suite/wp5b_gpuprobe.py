"""Read-only: GPU utilization and the python processes running on server B."""
import subprocess

print(subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used", "--format=csv,noheader"],
                     capture_output=True, text=True).stdout)
print(subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
                     capture_output=True, text=True).stdout)
ps = ("Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
      "ForEach-Object { '{0} {1} {2}' -f $_.ProcessId, $_.CreationDate.ToString('MM-dd HH:mm'), "
      "($_.CommandLine -replace '\\s+',' ').Substring(0, [Math]::Min(220, $_.CommandLine.Length)) }")
print(subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True).stdout)
