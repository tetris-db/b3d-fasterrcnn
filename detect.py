"""Pre-label B3D test clips: tiled RF-DETR detection at native scale over the padded mask region.

For each test segment, reads the 15 FPS frames the experiments use (even absolute source
frames, as in tetris/preprocess/p000_preprocess_dataset.py), tiles the mask's crop box
padded by PAD without blacking out the mask, and merges detections across tiles.
Boxes are written in full-frame 4K source coordinates; conversion to experiment
coordinates and score thresholding happen later, so this expensive step runs once.
"""
from __future__ import annotations

import argparse
import json
import os

import cv2
import numpy as np

from b3d import OVERLAP, TILE, VIDEOS, load_crop_and_mask, tiles


# Crop-box padding in pixels; at least the longest labeled B3D vehicle (476 px).
PAD = 480
# Segments per source video, and the five test segments (jncN/test/teNN.mp4) of every B3D dataset.
NUM_SEGMENTS = 18
TEST_SEGMENTS = [1, 4, 8, 12, 13]
# Keep every 2nd source frame: 29.97 FPS -> 15 FPS.
STEP = 2
# A box within this many pixels of a tile edge shared with another tile may be a cut-off fragment.
EDGE = 4
# Merge rules: duplicates overlap by IoU; fragments are mostly contained in the full box.
DUPLICATE_IOU = 0.5
FRAGMENT_IOS = 0.6


def parse_args():
    parser = argparse.ArgumentParser(description='Tiled RF-DETR pre-labels for B3D test clips')
    parser.add_argument('--model', choices=['rfdetr', 'detectron2'], default='rfdetr')
    parser.add_argument('--weights', default='/out/runs/large-lowlr/checkpoint_best_total.pth')
    parser.add_argument('--whole', action='store_true')
    parser.add_argument('--threshold', type=float, default=0.1)
    parser.add_argument('--datasets', nargs='+', default=list(VIDEOS))
    parser.add_argument('--segments', nargs='+', type=int, default=TEST_SEGMENTS)
    parser.add_argument('--max-frames', type=int, default=None)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--step', type=int, default=STEP)
    parser.add_argument('--reuse', default=None)
    parser.add_argument('--out', default='/out/detections')
    return parser.parse_args()


def segment_range(total_frames: int, segment: int) -> tuple[int, int]:
    """Source-frame range [start, end) of a segment; mirrors tetris get_segment_frame_range."""
    size = total_frames / NUM_SEGMENTS
    return int(segment * size), min(int((segment + 1) * size), total_frames)


def detection_region(path: str, width: int, height: int) -> tuple[tuple[int, int, int, int], list[tuple[int, int]]]:
    """Padded crop box (left, top, right, bottom) in the frame, and tile corners inside it."""
    (left, top, right, bottom), mask = load_crop_and_mask(path, width, height)
    # Pad the crop box and clamp it to the frame.
    region = (max(0, left - PAD), max(0, top - PAD), min(width, right + PAD), min(height, bottom + PAD))
    # Keep tiles within PAD of the road domain, so vehicles crossing the mask edge are covered.
    full = np.zeros((height, width), dtype=np.uint8)
    full[top:bottom, left:right] = mask
    grown = cv2.dilate(full, np.ones((2 * PAD + 1, 2 * PAD + 1), dtype=np.uint8))
    return region, tiles(grown[region[1]:region[3], region[0]:region[2]])


def fragment_flags(boxes: np.ndarray, x: int, y: int, width: int, height: int) -> np.ndarray:
    """True for tile-local boxes touching a tile edge that another tile overlaps (not the region border)."""
    flags = np.zeros(len(boxes), dtype=bool)
    if x > 0:
        flags |= boxes[:, 0] <= EDGE
    if y > 0:
        flags |= boxes[:, 1] <= EDGE
    if x + TILE < width:
        flags |= boxes[:, 2] >= TILE - EDGE
    if y + TILE < height:
        flags |= boxes[:, 3] >= TILE - EDGE
    return flags


def merge(boxes: np.ndarray, scores: np.ndarray, fragments: np.ndarray) -> np.ndarray:
    """Greedy merge across tiles; returns indices to keep.

    Complete boxes go first, then by score. A box is dropped when it duplicates a kept box
    (IoU >= DUPLICATE_IOU), or when it is a possible fragment mostly inside a kept box
    (intersection over the smaller box >= FRAGMENT_IOS). Fragments with no complete
    counterpart, such as a bus longer than the tile overlap, survive.
    """
    order = np.lexsort((-scores, fragments))
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    keep: list[int] = []
    for i in order:
        if keep:
            kept = np.array(keep)
            # Intersection of box i with every kept box.
            width = np.clip(np.minimum(boxes[i, 2], boxes[kept, 2]) - np.maximum(boxes[i, 0], boxes[kept, 0]), 0, None)
            height = np.clip(np.minimum(boxes[i, 3], boxes[kept, 3]) - np.maximum(boxes[i, 1], boxes[kept, 1]), 0, None)
            inter = width * height
            if (inter / (areas[i] + areas[kept] - inter) >= DUPLICATE_IOU).any():
                continue
            if fragments[i] and (inter / np.minimum(areas[i], areas[kept]) >= FRAGMENT_IOS).any():
                continue
        keep.append(int(i))
    return np.array(keep, dtype=int)


def detect_frame(predict, frame: np.ndarray, region: tuple[int, int, int, int], corners: list[tuple[int, int]],
                 batch: int, whole: bool = False) -> list[list[float]]:
    """Detections [x1, y1, x2, y2, score] for one BGR frame, in full-frame coordinates.

    predict maps a list of BGR images to (xyxy boxes, scores) pairs (eval_threshold.load_predictor).
    whole runs the padded region in one pass, for CNN detectors that take any input size.
    """
    left, top, right, bottom = region
    width, height = right - left, bottom - top
    image = frame[top:bottom, left:right]
    if whole:
        boxes, scores = predict([np.ascontiguousarray(image)])[0]
        boxes = boxes + [left, top, left, top]
        return [[*map(float, boxes[i].round(2)), round(float(scores[i]), 4)] for i in range(len(boxes))]
    all_boxes, all_scores, all_fragments = [], [], []
    for start in range(0, len(corners), batch):
        chunk = corners[start:start + batch]
        # Native-scale 704x704 tiles; the region is at least one tile on each side for every B3D video.
        images = [np.ascontiguousarray(image[y:y + TILE, x:x + TILE]) for x, y in chunk]
        for (x, y), (boxes, confidence) in zip(chunk, predict(images)):
            all_fragments.append(fragment_flags(boxes, x, y, width, height))
            # Shift tile-local boxes into full-frame coordinates.
            all_boxes.append(boxes + [left + x, top + y, left + x, top + y])
            all_scores.append(confidence)
    boxes, scores, fragments = np.concatenate(all_boxes), np.concatenate(all_scores), np.concatenate(all_fragments)
    keep = merge(boxes, scores, fragments)
    return [[*map(float, boxes[i].round(2)), round(float(scores[i]), 4)] for i in keep]


def main():
    args = parse_args()
    # Imported here: eval_threshold imports this module's merge helpers.
    from eval_threshold import load_predictor
    predict = load_predictor(args.model, args.weights, args.threshold, 1.0)

    for dataset in args.datasets:
        path = VIDEOS[dataset]
        cap = cv2.VideoCapture(path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        region, corners = detection_region(path, width, height)
        print(f'{dataset}: region {region}, {len(corners)} tiles per frame', flush=True)

        for segment in args.segments:
            # Partial test runs get their own file name so a full run never mistakes them for finished clips.
            suffix = f'.first{args.max_frames}' if args.max_frames is not None else ''
            output = os.path.join(args.out, dataset, f'te{segment:02d}{suffix}.jsonl')
            # Finished clips are skipped, so an interrupted run can simply be restarted.
            if os.path.exists(output):
                print(f'{dataset} te{segment:02d}: done, skipping', flush=True)
                continue
            os.makedirs(os.path.dirname(output), exist_ok=True)
            start, end = segment_range(total, segment)
            # Frames this model already detected with the same settings (e.g. a 15 FPS run reused at 30 FPS).
            reused = {}
            previous = os.path.join(args.reuse, dataset, f'te{segment:02d}.jsonl') if args.reuse else None
            if previous and os.path.exists(previous):
                reused = {r['source_frame']: r['detections'] for r in map(json.loads, open(previous))}
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
            frame_idx = 0
            with open(output + '.tmp', 'w') as f:
                for source_frame in range(start, end):
                    ok, frame = cap.read()
                    if not ok:
                        break
                    # Step 2 keeps the same frames as p000 (even absolute source indices); step 1 keeps all.
                    if source_frame % args.step != 0:
                        continue
                    if source_frame in reused:
                        detections = reused[source_frame]
                    else:
                        detections = detect_frame(predict, frame, region, corners, args.batch, args.whole)
                    f.write(json.dumps({'frame_idx': frame_idx, 'source_frame': source_frame, 'detections': detections}) + '\n')
                    frame_idx += 1
                    if frame_idx % 100 == 0:
                        print(f'{dataset} te{segment:02d}: {frame_idx} frames', flush=True)
                    if args.max_frames is not None and frame_idx >= args.max_frames:
                        break
            os.replace(output + '.tmp', output)
            print(f'{dataset} te{segment:02d}: {frame_idx} frames written', flush=True)
        cap.release()


if __name__ == '__main__':
    main()
