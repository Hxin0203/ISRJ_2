$ErrorActionPreference = 'Stop'
$pipelineRoot = Split-Path -Parent $PSScriptRoot
$pipelinePython = (Get-Command python).Source
$pipelineOutput = Join-Path $pipelineRoot 'outputs\paper2_long_pipeline_20261114'
New-Item -ItemType Directory -Force -Path $pipelineOutput | Out-Null
$pipelineStamp = Get-Date -Format 'yyyyMMdd_HHmmss_ffff'
$pipelineProcess = Start-Process -FilePath $pipelinePython -ArgumentList '-m tools.run_long_pipeline' -WorkingDirectory $pipelineRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $pipelineOutput "controller_$pipelineStamp.stdout.log") -RedirectStandardError (Join-Path $pipelineOutput "controller_$pipelineStamp.stderr.log") -PassThru
Write-Output "Pipeline controller PID: $($pipelineProcess.Id)"
