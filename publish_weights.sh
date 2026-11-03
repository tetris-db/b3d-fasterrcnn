#!/bin/sh
# Publish the trained frcnn-b3d weights (train_frcnn.py output) to the Hugging Face Hub with the hf CLI.
# Uploads config.yaml, model_final.pth, and model_card.md (as README.md) to REPO_ID in hub.env, after
# checking that model_final.pth is the model that generated the pre-labels (weights.sha256).
# Usage: ./publish_weights.sh <run directory with config.yaml and model_final.pth>   (needs `hf auth login`)
set -eu
cd "$(dirname "$0")"
. ./hub.env
RUN_DIR=${1:?usage: $0 <run directory>}
# Refuse to publish anything but the pre-labeling model under this name.
(cd "$RUN_DIR" && shasum -a 256 -c "$OLDPWD/weights.sha256")
hf repos create "$REPO_ID" --public --exist-ok
hf upload "$REPO_ID" "$RUN_DIR/config.yaml" config.yaml --commit-message "Add detectron2 config"
hf upload "$REPO_ID" "$RUN_DIR/model_final.pth" model_final.pth --commit-message "Add frcnn-b3d weights"
hf upload "$REPO_ID" model_card.md README.md --commit-message "Add model card"
echo "published https://huggingface.co/$REPO_ID"
