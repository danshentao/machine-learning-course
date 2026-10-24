param([switch]$Resume)
$ErrorActionPreference='Stop'
$Root='E:\机器学习课设\第三代鸟类识别现代模型对照\experiments\detector_training_improvements_norm_aligned'
$Python='E:\机器学习课设\自然场景鸟类检测PPYOLOE\.venv\Scripts\python.exe'
$env:PADDLE_HOME=Join-Path $Root '.cache\paddle'
$env:TEMP=Join-Path $Root '.cache\tmp'
$env:TMP=$env:TEMP
$env:PYTHONUNBUFFERED='1'
$env:PATH='E:\机器学习课设\自然场景鸟类检测PPYOLOE\.venv\Lib\site-packages\nvidia\cudnn\bin;E:\机器学习课设\自然场景鸟类检测PPYOLOE\.venv\Lib\site-packages\nvidia\cublas\bin;'+$env:PATH
$env:PYTHONPATH='E:\机器学习课设\自然场景鸟类检测PPYOLOE\third_party\PaddleDetection;'+$env:PYTHONPATH
New-Item -ItemType Directory -Force -Path "$Root\logs","$Root\.cache\tmp","$Root\.cache\paddle","$Root\reports" | Out-Null
$status=Join-Path $Root 'reports\formal_training_status.json'
if ((Test-Path $status) -and -not $Resume) { throw '状态文件已存在；若需从检查点继续，请加 -Resume。' }
$stdout=Join-Path $Root 'logs\training_driver.stdout.log'
$stderr=Join-Path $Root 'logs\training_driver.stderr.log'
$driver=Join-Path $Root 'scripts\run_formal_training.py'
$proc=Start-Process -FilePath $Python -ArgumentList @($driver) -WorkingDirectory 'E:\机器学习课设\自然场景鸟类检测PPYOLOE\third_party\PaddleDetection' -WindowStyle Hidden -PassThru -RedirectStandardOutput $stdout -RedirectStandardError $stderr
$record=@{pid=$proc.Id;started_at=(Get-Date).ToString('o');python=$Python;driver=$driver;stdout=$stdout;stderr=$stderr;resume=[bool]$Resume}
$record | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $Root 'reports\training_process.json') -Encoding utf8
Write-Output ("Started normalization-aligned A/B/C training PID {0}" -f $proc.Id)
