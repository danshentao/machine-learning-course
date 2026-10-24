$ErrorActionPreference='Stop'
$Root='E:\机器学习课设\第三代鸟类识别现代模型对照\experiments\detector_training_improvements_norm_aligned'
$Status=Join-Path $Root 'reports\formal_training_status.json'
if (-not (Test-Path $Status)) { throw '找不到训练状态文件，不能安全恢复。' }
$state=Get-Content -Raw $Status | ConvertFrom-Json
if ($state.status -ne 'paused') { throw "当前状态为 $($state.status)，此入口只接受用户暂停后的状态。" }
$pp='E:\机器学习课设\自然场景鸟类检测PPYOLOE\.venv\Scripts\python.exe'
$torch='E:\机器学习课设\第三代鸟类识别现代模型对照\.venv\Scripts\python.exe'
& (Join-Path $Root 'start_training.ps1') -Resume
$fin=Start-Process -FilePath $torch -ArgumentList @((Join-Path $Root 'scripts\finalize_after_training.py')) -WorkingDirectory $Root -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $Root 'logs\finalizer_wait.stdout.log') -RedirectStandardError (Join-Path $Root 'logs\finalizer_wait.stderr.log')
@{pid=$fin.Id;started_at=(Get-Date).ToString('o');script=(Join-Path $Root 'scripts\finalize_after_training.py')} | ConvertTo-Json | Set-Content (Join-Path $Root 'reports\post_training_finalizer_process.json')
$watch=Start-Process -FilePath $pp -ArgumentList @((Join-Path $Root 'scripts\watch_training_recovery.py')) -WorkingDirectory $Root -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $Root 'logs\watchdog.stdout.log') -RedirectStandardError (Join-Path $Root 'logs\watchdog.stderr.log')
@{pid=$watch.Id;started_at=(Get-Date).ToString('o');script=(Join-Path $Root 'scripts\watch_training_recovery.py')} | ConvertTo-Json | Set-Content (Join-Path $Root 'reports\training_watchdog_process.json')
Write-Output "已从检查点恢复；finalizer PID=$($fin.Id)，watchdog PID=$($watch.Id)"
