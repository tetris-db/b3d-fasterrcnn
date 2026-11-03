# tetris-annotate

Pre-labels for the human-verified B3D ground truth promised in R1.O1 of the Tetris revision.
An RF-DETR-Large detector, independent of the B3D RetinaNet that Tetris is evaluated with, is fine-tuned on B3D's human-labeled crops and run on the B3D test frames. Humans then review and correct every frame.
Detection only; tracking is out of scope for now.

## Running

Runs on bombe GPU 4 (the only idle GPU on 2026-10-19). Long runs go inside the `annotate` tmux session, because bombe has no linger and its rootless Docker daemon stops when the last SSH session closes.

```sh
rsync -az --exclude viz/ --exclude __pycache__/ tetris-annotate/ bombe:/work/cwkt/projects/tetris-annotate/
ssh bombe
cd /work/cwkt/projects/tetris-annotate && CUDA_VISIBLE_DEVICES=4 ./run <script>.py
```

`./run` mounts `/data/cwkt/data/polyis-data/sources` read-only at `/sources` and `/data/cwkt/data/tetris-annotate` at `/out`.
`python b3d.py` runs the tiling self-check.

## Decisions

- **Training data** (`prep_data.py`): B3D's 16,002 labeled crops (`/sources/b3d-vision`), kept at native 4K scale. Crops are cut into 704×704 windows (256 px overlap) and padded with black. Boxes are clipped to their window and kept if at least 25% remains. The data is re-split by crop group (file-name prefix) 80/10/10, because B3D's own split puts crops of the same frame in different splits.
- **Training** (`train.py`): RF-DETR-Large at 704 with no rescaling (`multi_scale` and `scale_jitter` off), the `AUG_AERIAL` augmentations, auto batch (13 × 2 accumulation), and early stopping on validation mAP. The learning rate stays constant at RF-DETR's defaults: the `step` schedule only drops at epoch 100, and EMA weights do the smoothing. If recall falls short, try `lr_scheduler="cosine"` first. Logs go to wandb `chanwutk-berkeley/Tetris`.
- **Faster R-CNN comparison** (`prep_crops.py`, `train_frcnn.py`, run on ace): detectron2 `faster_rcnn_X_101_32x8d_FPN_3x` from its COCO checkpoint, following `modules/b3d/train.py` in B3D's own Docker image (`modules/b3d/Dockerfile` + `Dockerfile.detectron2`). Trained on B3D's crops at native scale, with crops over 1,408 px windowed for memory, on our crop-group split (`frcnn-groups`) and on B3D's own split (`frcnn-b3d`, which leaks frames into both held-out sets). On the 704-tile evaluation (`eval_threshold.py --model detectron2`, results in `/out/eval/`), RF-DETR beats `frcnn-groups` on both held-out sets (valid AP 0.698 vs 0.613; recall at 0.2: 98.0% vs 94.1%), but those tiles (black-padded windows with cut boxes) are RF-DETR's training format. On the held-out crops, each model tiled at 704 and merged as on video (`--tile`), the gap is smaller: valid AP50 0.974 vs 0.973, AP 0.673 vs 0.590, recall at 0.2 98.8% vs 97.1%. Faster R-CNN does best at native scale (upscaling 1.5x or 2x lowers recall), and tiling vs one whole-image pass makes no difference (AP within 0.003). RF-DETR stays the pre-labeler for recall and box tightness.
- **Detection region**: native-resolution 4K frames, with the mask's crop box padded by 480 px (the largest labeled vehicle is 476 px), clamped to the frame. Pixels outside the mask are not blacked out, so vehicles crossing the mask edge are seen whole. 704×704 tiles with 256 px overlap; tiles that don't touch the mask grown by 480 px are skipped.
- **Ground-truth box** (`track.py`): the visible extent. Map each detection to experiment coordinates (crop offset, rotation for vertical regions, resize to 1080×720, as in `tetris/preprocess/p000_preprocess_dataset.py`) and take the bounding box of its pixels inside the mask. Keep it if at least 25% of the box's pixels are inside the mask and its score is at least 0.2. At 0.2, held-out recall against B3D's labels is 98.8% (test) and 98.0% (valid); see `eval_threshold.py`.
- **Frames**: clip frame `k` of `jncN/test/teNN.mp4` (segment `NN` of 18) is source frame `start + (start mod 2) + 2k`, i.e. the even source frames in the segment. `check_alignment.py` verifies this against ace's current clips, which are copied to `/out/experiment-clips`; bombe's own `polyis-data/datasets` copy is stale (2025-11).
- **Tracking** (`track.py`): OC-SORT on the ground-truth boxes, using p003's tracker (tetris's compiled Cython OC-SORT, so run it with `IMAGE=polyis-polyis` on ace, whose Python 3.13 matches the build), `configs/trackers.yaml`, and `register_tracked_detections` gap interpolation, imported read-only from `/tetris`. With `--fps 30` (detections from `detect.py --step 1`), it tracks every source frame and writes `{detection,tracking}30.jsonl`, then samples the clip frames into `{detection,tracking}.jsonl`. `track.py --verify` reproduces the existing `003_groundtruth/tracking.jsonl` for jnc0 te01 exactly (0 of 1,075 frames differ).
- **Agreement with the old RetinaNet GT** (`compare_gt.py`, HOTA over the 20 test clips, detect and track at 30 FPS): mean over jnc0/2/6/7 at score threshold 0.2 / 0.5 is RF-DETR 0.829 / 0.839, `frcnn-groups` 0.840 / 0.838, `frcnn-b3d` 0.842 / 0.839, the same at 30 FPS and at the 15 FPS clip frames. The spread between models (≤ 0.013) is smaller than the spread between datasets (jnc2 0.80–0.82, jnc6 0.84–0.87), and Faster R-CNN shares the old GT's CNN lineage, so this does not rank the detectors. The old 30 FPS reference is rebuilt from the old detections with the Cython OC-SORT and reproduces p003's saved (every 2nd frame) tracks exactly on all 20 clips; `compare_gt.py` asserts this. Results are in `/out/compare30-<model>-t<threshold>/agreement.json`.
- **Output**: `/out/groundtruth/<dataset>/teNN.mp4/{detection,tracking}.jsonl`, in the same format as the cache's `003_groundtruth`.
- **Pre-labeler for review**: `frcnn-b3d` at score threshold 0.2 (`/out/gt30-frcnn-b3d-t0.2`), the closest to the existing ground truth (HOTA 0.842) and high-recall, so reviewers mostly delete rather than draw boxes.
- **Review** (`viz_tracks.py`, `export_cvat.py`): track videos in `/out/viz/tracks-frcnn-b3d/`; CVAT tasks in `/out/cvat/<dataset>_teNN/`, each the experiment clip plus `annotations.zip` (CVAT "MOT 1.1" import, loaded as tracks; IDs match the videos). In CVAT, create a task from the clip with label `vehicle`, then Actions → Upload annotations → MOT 1.1.

## Human review plan

The accepted R1.O1 plan (`tetris-paper-feedback/wip/r1o1.tex`) commits to human-verified detection and tracking ground truth for the B3D test set (jnc0/2/6/7, 5 clips each, 20,483 frames at 15 FPS). The labels are initialized by a model other than the evaluated detector (RetinaNet), and humans review and correct every test frame for missed objects, false detections, inaccurate boxes, and inconsistent identities. R1's discussion comment asks to see these results in the revision.

Pre-labels: `frcnn-b3d`, threshold 0.2, 308,126 boxes in 1,508 tracks (median 181 frames; 128 tracks under 1 s, most likely fragments or false positives).

### Labeling rules (draft guide, to finalize after the pilot)

- Object: every vehicle (car, bus, truck, motorcycle) on the road inside the mask; no pedestrians or bicycles. One label, `vehicle`, matching the evaluation's class.
- Box: the visible extent inside the mask. Areas outside the mask are black in the clip, so the box covers only the vehicle pixels the clip shows. Drop a vehicle (mark it outside) while less than about a quarter of it is visible, matching the pre-label rule.
- Identity: one track per physical vehicle for the whole time it is visible in the clip. If it is hidden briefly (under a bridge, by a bus) and reappears, keep the same ID; mark the hidden frames outside.
- Parked or stopped vehicles are labeled like moving ones.
- Ambiguous cases (shadows, heavy occlusion, vehicles half outside the mask): annotate them and add a `doubt` tag rather than skipping them. They are listed in the guide as they come up.

### Workflow

1. Export keyframes, not every frame. MOT 1.1 makes all 308k boxes keyframes, so fixing a drifting box means editing every frame. A CVAT-for-video XML export that keeps a keyframe only where linear interpolation misses the track by more than 2 px needs 34k keyframes (98k at 1 px, 14k at 4 px), and CVAT interpolates the rest.
2. Review hints per clip: tracks under 1 s, and boxes where RF-DETR and `frcnn-b3d` disagree (one model has a box the other lacks). They help reviewers find likely errors faster. Every frame is still reviewed.
3. Pilot: one person reviews one clip (e.g. jnc2 te04, the densest), with times recorded. Finalize the rules and the per-clip time estimate from this pilot.
4. Review all 20 clips, one reviewer per clip.
5. Second pass: a different person checks a sample from every clip (e.g. every 30th frame and every ID switch flagged by the tool), fixing errors and recording the error rate found.
6. Independence check: annotate a sample from scratch, without pre-labels, and score the reviewed labels against it. This measures how much the pre-labels anchor the result, which is what R1.O1 is about.
7. Import back to `{detection,tracking}.jsonl` in the `003_groundtruth` format (needs an `import_cvat.py`), then score the reviewed GT against the pre-labels and the old GT to report how much review changed.

### What reruns after the labels exist

- Every B3D accuracy number in the paper: Tetris configurations, the reference pipeline (no longer 1.0 by construction), OTIF, LEAP, ablations, the exhaustive-search comparison, and the tolerance-to-gap table. This is re-scoring cached tracking outputs with `p110`, not rerunning detection, provided the outputs of every scored configuration are still cached.
- R1.O4 (precision/recall before and after packing) plans to use these labels.
- Configurations stay selected on the validation split with the old automatic labels; only the test set is human-verified, which the paper must state.
