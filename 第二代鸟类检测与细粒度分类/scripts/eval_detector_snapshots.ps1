param([string]$Config = "configs/ppyoloe_plus_bird_640.yml")
$ErrorActionPreference = "Stop"
$root = "E:\机器学习课设\第二代鸟类检测与细粒度分类"
$dir = "$root\runs\detector"
$rows = @()
$snapshots = Get-ChildItem -LiteralPath $dir -Filter "*.pdparams" | Where-Object { $_.BaseName -match '^\d+$' } | Sort-Object { [int]$_.BaseName }
$final = Get-Item -LiteralPath "$dir\model_final.pdparams" -ErrorAction SilentlyContinue
if ($final) { $snapshots += $final }
if (-not $snapshots) { throw "No detector snapshots found in $dir" }
foreach ($file in $snapshots) {
    if ($file.BaseName -eq "model_final") { $epoch = 100; $prefix = "$dir\model_final" }
    else { $epoch = [int]$file.BaseName + 1; $prefix = "$dir\$($file.BaseName)" }
    $tag = "v2_epoch_$epoch"
    Write-Host "Evaluating epoch ${epoch}: $prefix"
    & "$root\scripts\eval_detector.ps1" -Config $Config -Weights $prefix -Split $tag
    $log = Get-Content -Raw -Encoding UTF8 "$root\runs\eval_$tag\eval.log"
    $apLine = ($log -split "`r?`n") | Where-Object { $_ -match 'Average Precision.*IoU=0\.50\s+\|' -and $_ -notmatch 'IoU=0\.50:0\.95' } | Select-Object -First 1
    $match = [regex]::Match([string]$apLine, '=\s*([0-9.]+)\s*$')
    if (-not $match.Success) { throw "Could not parse AP50 from $root\runs\eval_$tag\eval.log" }
    $rows += [pscustomobject]@{epoch=$epoch;checkpoint=$prefix;AP50=[double]::Parse($match.Groups[1].Value,[cultureinfo]::InvariantCulture);evaluation_log="$root\runs\eval_$tag\eval.log"}
}
$rows | Export-Csv -LiteralPath "$root\runs\detector\posthoc_checkpoint_ap50.csv" -NoTypeInformation -Encoding UTF8
$best = $rows | Sort-Object @{Expression='AP50';Descending=$true},@{Expression='epoch';Descending=$true} | Select-Object -First 1
Copy-Item -LiteralPath "$($best.checkpoint).pdparams" -Destination "$dir\best_model_ap50.pdparams" -Force
$finalSource = "$dir\model_final.pdparams"
if (Test-Path -LiteralPath $finalSource) { Copy-Item -LiteralPath $finalSource -Destination "$dir\final_model.pdparams" -Force }
$selection = [ordered]@{selected_epoch=$best.epoch;selected_AP50=$best.AP50;checkpoint=$best.checkpoint;metric='official PaddleDetection COCO bbox AP at IoU=0.50';selection='post-hoc comparison across all 5-epoch snapshots plus final checkpoint, same fixed 172-image validation split';seed=0;in_training_eval_enabled=$false;snapshot_interval_epochs=5}
$selection | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath "$dir\checkpoint_selection.json" -Encoding UTF8
Write-Host "Best epoch=$($best.epoch) AP50=$($best.AP50)"
