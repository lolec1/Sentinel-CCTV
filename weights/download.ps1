# weights/download.ps1 — PowerShell script to download weights
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$weightPath = Join-Path $scriptDir "yolov8n.pt"

if (-not (Test-Path $weightPath)) {
    Write-Host "Downloading YOLOv8n weights..."
    Invoke-WebRequest -Uri "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt" -OutFile $weightPath
} else {
    Write-Host "Weights already present at $weightPath"
}
