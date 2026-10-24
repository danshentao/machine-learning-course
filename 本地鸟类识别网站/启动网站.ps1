$ErrorActionPreference = 'Stop'
$websiteRoot = $PSScriptRoot
$pythonPath = Join-Path (Split-Path $websiteRoot -Parent) 'RFDETR与PPYOLOE鸟检测对照\runtime\python311-embed\python.exe'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONIOENCODING = 'utf-8'
$url = 'http://127.0.0.1:8765'
try {
    $status = Invoke-RestMethod "$url/api/status" -TimeoutSec 2
    if ($status.models) { Start-Process $url; exit }
} catch {}
$logRoot = Join-Path $websiteRoot 'data'
New-Item -ItemType Directory -Path $logRoot -Force | Out-Null
Start-Process -FilePath $pythonPath -ArgumentList @('-u', (Join-Path $websiteRoot 'server.py'), '--port', '8765') -WorkingDirectory $websiteRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $logRoot 'server.log') -RedirectStandardError (Join-Path $logRoot 'server_error.log')
for ($attempt = 0; $attempt -lt 30; $attempt++) {
    Start-Sleep -Milliseconds 300
    try {
        $status = Invoke-RestMethod "$url/api/status" -TimeoutSec 1
        if ($status.models) { Start-Process $url; exit }
    } catch {}
}
throw '网站未成功启动，请查看 data/server_error.log。'
