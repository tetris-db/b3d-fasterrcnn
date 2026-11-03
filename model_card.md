---
license: other
library_name: detectron2
tags:
- object-detection
- vehicle-detection
- faster-rcnn
---

# b3d-fasterrcnn

A detectron2 Faster R-CNN X101-32x8d-FPN, fine-tuned from its COCO checkpoint on the human-labeled vehicle crops of the B3D dataset (Berkeley DeepDrive Drone, one `vehicle` class) at native 4K scale.
It generated the pre-labels that were human-corrected into the B3D test-set ground truth used in the Tetris revision.

- `config.yaml`: the detectron2 config of the training run (input resizing disabled, 500 detections per image).
- `model_final.pth`: the final weights (sha256 `f7caab8f1994c737eb3a3c5ac84a31a70db1b1651bff4ae43ceb66804d011174`).

Training, detection, and tracking code: https://github.com/chanwutk/b3d-fasterrcnn (`download_weights.sh` fetches these files, and `detect.py` runs them).
