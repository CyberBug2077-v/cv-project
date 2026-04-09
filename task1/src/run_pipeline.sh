#!/bin/bash
# Wait for autoencoder training to finish, then launch pretrained segmentation.
# Usage: bash src/run_pipeline.sh

set -e
cd "$(dirname "$0")/.."

echo "[pipeline] Waiting for autoencoder to finish..."
while kill -0 $(pgrep -f "train_autoencoder.py" | head -1) 2>/dev/null; do
    # Poll every few seconds until the autoencoder job exits.
    sleep 10
done
echo "[pipeline] Autoencoder done. Launching pretrained segmentation training..."

conda run -n ML python3 -u src/train_pretrained_seg.py > outputs/pretrained_seg/run.log 2>&1
echo "[pipeline] Pretrained seg training complete."
