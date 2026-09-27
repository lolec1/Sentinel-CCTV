#!/usr/bin/env bash
# weights/download.sh — Download model weights if not already present
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

if [ ! -f "$DIR/yolov8n.pt" ]; then
    echo "Downloading YOLOv8n weights..."
    curl -L -o "$DIR/yolov8n.pt" "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt"
else
    echo "Weights already present at $DIR/yolov8n.pt"
fi
