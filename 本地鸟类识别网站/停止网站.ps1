$ErrorActionPreference = 'Stop'
$pythonPath = Join-Path (Split-Path $PSScriptRoot -Parent) 'RFDETR与PPYOLOE鸟检测对照\runtime\python311-embed\python.exe'
try {
    $records = Invoke-RestMethod 'http://127.0.0.1:8765/api/history' -TimeoutSec 2
    if ($records | Where-Object { $_.status -in @('queued', 'running') }) {
        Write-Host '还有识别任务，请等识别完成后再停止网站。'
        exit
    }
} catch {}
$pidFile = Join-Path $PSScriptRoot 'data\server.pid'
if (Test-Path -LiteralPath $pidFile) {
    $processId = 0
    if ([int]::TryParse((Get-Content -LiteralPath $pidFile -Raw), [ref]$processId)) {
        $websiteProcess = Get-Process -Id $processId -ErrorAction SilentlyContinue
        if ($websiteProcess -and $websiteProcess.Path -eq $pythonPath) {
            Stop-Process -Id $processId -ErrorAction SilentlyContinue
            for ($attempt = 0; $attempt -lt 20 -and (Get-Process -Id $processId -ErrorAction SilentlyContinue); $attempt++) { Start-Sleep -Milliseconds 200 }
        }
    }
}
if (Test-Path -LiteralPath $pidFile) {
    $remainingPid = 0
    if ([int]::TryParse((Get-Content -LiteralPath $pidFile -Raw), [ref]$remainingPid)) {
        $remaining = Get-Process -Id $remainingPid -ErrorAction SilentlyContinue
        if ($remaining -and $remaining.Path -eq $pythonPath) { throw '网站服务进程未停止；历史记录保持不变。' }
        Remove-Item -LiteralPath $pidFile -Force
    }
}
Write-Host '本地网站已停止，历史记录保持不变。'
