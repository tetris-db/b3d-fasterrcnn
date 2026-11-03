#!/bin/sh
# Download the frcnn-b3d weights from the Hugging Face Hub with the hf CLI, for detect.py.
# Writes config.yaml and model_final.pth to OUT_DIR (default: detect.py's default --weights on the host)
# and checks model_final.pth against weights.sha256.
# Usage: ./download_weights.sh [OUT_DIR]   (needs the hf CLI: pip install -U huggingface_hub)
set -eu
cd "$(dirname "$0")"
. ./hub.env
OUT_DIR=${1:-${DATA:-/data/cwkt/data}/tetris-annotate/weights/frcnn-b3d}
hf download "$REPO_ID" config.yaml model_final.pth --local-dir "$OUT_DIR"
# The weights must be the model behind the pre-labels.
(cd "$OUT_DIR" && shasum -a 256 -c "$OLDPWD/weights.sha256")
