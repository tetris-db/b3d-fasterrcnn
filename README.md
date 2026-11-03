# b3d-fasterrcnn

Vehicle detection and tracking pre-labels for the B3D test clips (jnc0/2/6/7, segments 1, 4, 8, 12, 13).
A detectron2 Faster R-CNN X101-FPN, fine-tuned on B3D's human-labeled crops, detects vehicles on every 4K source frame, and OC-SORT tracks them in the 1080x720 experiment frame.
Weights: [tetris-db/b3d-fasterrcnn](https://huggingface.co/tetris-db/b3d-fasterrcnn).

The finished tracks are already in this repository, at `results/gt30-frcnn-b3d-t0.2/jncX/teXX.mp4/tracking30.jsonl` (every source frame, 30 FPS) and `tracking.jsonl` (the clip's 15 FPS frames), so nothing needs to be run to use them.

## Setup

```sh
docker build -t b3d-fasterrcnn .
./run setup.py build_ext --inplace   # compile the Cython OC-SORT in ocsort/
./download_data.sh                   # B3D crops and the four test videos (needs gdown)
./download_weights.sh                # the published weights (needs the hf CLI)
```

`./run <script>.py` runs a script in a throwaway container of the image, with `$DATA/polyis-data/sources` (where `download_data.sh` writes) read-only at `/sources` and `$DATA/tetris-annotate` at `/out`.
`DATA` defaults to `/data/cwkt/data`.

## Detect and track

```sh
./run detect.py                      # /out/detections30-frcnn-b3d/<dataset>/teNN.jsonl
CUDA_VISIBLE_DEVICES= ./run track.py # /out/gt30-frcnn-b3d-t0.2/<dataset>/teNN.mp4/
```

- `detect.py` runs the model once per source frame on the road mask's crop box padded by 480 px, without blacking out pixels outside the mask, and keeps boxes with score at least 0.1, in 4K source coordinates.
- `track.py` maps each box into the experiment frame (crop, rotate vertical regions, resize to 1080x720), shrinks it to its visible extent inside the mask, keeps it if its score is at least 0.2 and at least 25% of it is inside the mask, and tracks at 30 FPS with OC-SORT and linear gap interpolation.
  `{detection,tracking}30.jsonl` hold every source frame; `{detection,tracking}.jsonl` hold the clip's frames (clip frame `k` is source frame `start + (start mod 2) + 2k`).

## Training

```sh
./run prep_crops.py                                  # /out/dataset-crops-b3d
CUDA_VISIBLE_DEVICES=0,1,2,3 ./run train_frcnn.py    # /out/runs/frcnn-b3d
./run detect.py --weights /out/runs/frcnn-b3d        # detect with the trained model
./publish_weights.sh /data/cwkt/data/tetris-annotate/runs/frcnn-b3d   # upload to the repo in hub.env
```

- `prep_crops.py` uses B3D's own train/val/test split at native scale; crops longer than 1,408 px are cut into overlapping windows with clipped boxes.
- `train_frcnn.py` fine-tunes `faster_rcnn_X_101_32x8d_FPN_3x` from its COCO checkpoint, following B3D's `train.py`: one `vehicle` class, no resizing, flips, 90-degree rotations and mild brightness/contrast, SGD at 0.01 with batch 8 for 6 epochs on 4 GPUs, logged to wandb.
- `publish_weights.sh` uploads only weights matching `weights.sha256`, the model behind `results/`.

## Results

`results/` holds the outputs of the published model for the 20 test clips, in Tetris's `003_groundtruth` format (one JSON row per frame, tracks as `[id, x1, y1, x2, y2]`):

- `gt30-frcnn-b3d-t0.2/jncX/teXX.mp4/tracking30.jsonl`: the tracks, every source frame (30 FPS); `detection30.jsonl` holds the boxes they were tracked from.
- `gt30-frcnn-b3d-t0.2/jncX/teXX.mp4/tracking.jsonl`: the same tracks at the clip's 15 FPS frames, the pre-labels used for human annotation; `detection.jsonl` likewise.
- `detections30-frcnn-b3d/jncX/teXX.jsonl`: `detect.py`'s raw detections in 4K source coordinates.

## Provenance

- `Dockerfile` is based on B3D's ([b3d-project/b3d](https://github.com/b3d-project/b3d)), pinned to the versions that produced `results/`.
- `ocsort/` is Tetris's Cython OC-SORT with `lapjv` from [chanwutk/lap](https://github.com/chanwutk/lap) (BSD-2-Clause), built with Tetris's compiler flags.
- `data/annotations.xml` holds each video's crop box and road polygons; `data/sources.sha256` and `weights.sha256` pin the inputs and weights.
