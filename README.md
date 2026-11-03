# b3d-fasterrcnn

Pre-labels for the human-verified B3D ground truth in the Tetris revision (R1.O1).
A detectron2 Faster R-CNN X101-FPN is fine-tuned on B3D's human-labeled crops, run on every source frame of the 20 B3D test clips (jnc0/2/6/7, segments 1, 4, 8, 12, 13), and tracked with Tetris's OC-SORT.
The resulting tracks (`frcnn-b3d`, score threshold 0.2) are the pre-labels the human annotation started from; the 20 `tracking.jsonl` files in the annotation project's `sources/labels/` are byte-identical to this pipeline's output.

## Pipeline

Everything runs in one image built from `Dockerfile` (B3D's detectron2 environment: Python 3.7, PyTorch 1.10, CUDA 11.3), through `./run`, which starts a throwaway container with `/data/cwkt/data/polyis-data/sources` read-only at `/sources` and `/data/cwkt/data/tetris-annotate` at `/out`.

```sh
docker build -t b3d-fasterrcnn .
./run setup.py build_ext --inplace                   # compile the Cython OC-SORT in ocsort/

# Either download the published weights (needs the hf CLI on the host)...
./download_weights.sh                                # /data/cwkt/data/tetris-annotate/weights/frcnn-b3d
# ...or train them.
./run prep_crops.py                                  # /out/dataset-crops-b3d
CUDA_VISIBLE_DEVICES=0,1,2,3 ./run train_frcnn.py    # /out/runs/frcnn-b3d (4 GPUs, 6 epochs)

./run detect.py                                      # /out/detections30-frcnn-b3d/<dataset>/teNN.jsonl
CUDA_VISIBLE_DEVICES= ./run track.py                 # /out/gt30-frcnn-b3d-t0.2/<dataset>/teNN.mp4/
```

`detect.py` reads `/out/weights/frcnn-b3d` by default; pass `--weights /out/runs/frcnn-b3d` to use a training run instead.
The defaults reproduce the run behind the pre-labels.
The original detection pass reused the even frames of an identical earlier 15 FPS pass of the same model and filled in the odd frames; running `detect.py` from scratch gives the same rows.

`publish_weights.sh <run directory>` uploads `config.yaml`, `model_final.pth`, and `model_card.md` to the Hugging Face repo in `hub.env` with the `hf` CLI, after checking `model_final.pth` against `weights.sha256`; `download_weights.sh` checks the same hash.

`ocsort/` is Tetris's Cython OC-SORT (`polyis/tracker/ocsort/cython` at commit 8404d4f) with `lapjv` from github.com/chanwutk/lap (BSD-2-Clause, `ocsort/lapjv/LICENSE`), built with Tetris's compiler flags and directives (`setup.py`).
Rebuilt here, it reproduces Tetris's tracks byte for byte.

## What each step does

- **Training data** (`prep_crops.py`): B3D's labeled crops (`/sources/b3d-vision`) with B3D's own train/val/test split, at native scale.
  Crops longer than 1,408 px on a side are cut into overlapping windows (256 px overlap, no resizing or padding), and boxes are clipped to each window and kept if at least 25% remains.
- **Training** (`train_frcnn.py`): detectron2 `faster_rcnn_X_101_32x8d_FPN_3x` from its COCO checkpoint, following B3D's `train.py` (github.com/b3d-project/b3d).
  One `vehicle` class, native scale (`MIN_SIZE_TRAIN = (0,)`, `MIN_SIZE_TEST = 0`), anchors 16-256 px, flips, 90-degree rotations and mild brightness/contrast, SGD at 0.01 with batch 8 and 10x drops at 70% and 90% of 6 epochs, and validation every epoch.
  Logs go to wandb `chanwutk-berkeley/Tetris`.
- **Detection** (`detect.py`): every source frame (29.97 FPS) of each test segment, at native 4K resolution.
  The model runs once on the mask's crop box padded by 480 px (the largest labeled vehicle is 476 px) and clamped to the frame.
  Pixels outside the road mask are not blacked out, so vehicles crossing the mask edge are seen whole.
  Boxes with score at least 0.1 are written in source-frame coordinates.
- **Tracking** (`track.py`): each detection is mapped into Tetris's 1080x720 experiment frame (crop offset, rotation of vertical regions, resize, as in `tetris/preprocess/p000_preprocess_dataset.py`) and shrunk to its visible extent, the bounding box of its pixels inside the road mask.
  It is kept if its score is at least 0.2 and at least 25% of its pixels lie inside the mask.
  The kept boxes are tracked at 30 FPS with Tetris's ground-truth tracker (`tetris/preprocess/p003_preprocess_groundtruth_tracking.py`): the Cython OC-SORT in `ocsort/` with the settings of `tetris/configs/trackers.yaml` and linear gap interpolation, copied into `track.py`.
  `{detection,tracking}30.jsonl` hold every source frame.
  `{detection,tracking}.jsonl` hold the experiment clip's frames: clip frame `k` of `jncN/test/teNN.mp4` is source frame `start + (start mod 2) + 2k`, the even source frames of the segment.

## Why this model

Against the old automatic ground truth (tiled RetinaNet plus OC-SORT), detecting and tracking at 30 FPS, mean HOTA over jnc0/2/6/7 at score threshold 0.2 was 0.842 for `frcnn-b3d`, against 0.840 for the same model trained on a crop-group split and 0.829 for an RF-DETR-Large alternative.
The spread between models (at most 0.013) was smaller than between datasets, so this does not rank detectors; `frcnn-b3d` was chosen as the closest to the existing ground truth, with high recall so reviewers mostly delete rather than draw boxes.
The pre-labels hold 308,126 boxes in 1,508 tracks at the clip frames.
The other models and the comparison scripts are in this repository's history (commits before the cleanup).
