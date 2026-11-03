"""Pre-label B3D test clips: Faster R-CNN detection at native scale over the padded mask region.

For each test segment, reads every source frame (29.97 FPS), crops the mask's crop box padded
by PAD without blacking out the mask, and runs the fine-tuned detectron2 model on that region in
one pass. --weights is a directory with config.yaml and model_final.pth: by default the published
weights from download_weights.sh, or a train_frcnn.py run directory such as /out/runs/frcnn-b3d.
Boxes are written in full-frame 4K source coordinates; conversion to experiment coordinates,
score thresholding, and tracking happen in track.py, so this expensive step runs once.

Output: <out>/<dataset>/teNN.jsonl, one row per source frame:
  {"frame_idx": i, "source_frame": start + i, "detections": [[x1, y1, x2, y2, score], ...]}
"""
from __future__ import annotations

import argparse
import json
import os

import cv2
import numpy as np

from b3d import TEST_SEGMENTS, VIDEOS, load_crop_and_mask, segment_range


# Crop-box padding in pixels; at least the longest labeled B3D vehicle (476 px).
PAD = 480


def parse_args():
    parser = argparse.ArgumentParser(description='Faster R-CNN pre-labels for B3D test clips')
    parser.add_argument('--weights', default='/out/weights/frcnn-b3d')
    parser.add_argument('--threshold', type=float, default=0.1)
    parser.add_argument('--datasets', nargs='+', default=list(VIDEOS))
    parser.add_argument('--segments', nargs='+', type=int, default=TEST_SEGMENTS)
    parser.add_argument('--max-frames', type=int, default=None)
    parser.add_argument('--out', default='/out/detections30-frcnn-b3d')
    return parser.parse_args()


def load_predictor(weights: str, threshold: float):
    """A function from one BGR image to (xyxy boxes, scores) arrays, from a train_frcnn.py run directory."""
    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor
    cfg = get_cfg()
    cfg.merge_from_file(os.path.join(weights, 'config.yaml'))
    cfg.MODEL.WEIGHTS = os.path.join(weights, 'model_final.pth')
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
    # The run's config already disables resizing (MIN_SIZE_TEST = 0); DefaultPredictor takes BGR.
    detector = DefaultPredictor(cfg)

    def predict(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        o = detector(image)['instances'].to('cpu')
        return o.pred_boxes.tensor.numpy().astype(float), o.scores.numpy().astype(float)
    return predict


def detection_region(path: str, width: int, height: int) -> tuple[int, int, int, int]:
    """The mask's crop box (left, top, right, bottom) padded by PAD and clamped to the frame."""
    (left, top, right, bottom), _ = load_crop_and_mask(path, width, height)
    return max(0, left - PAD), max(0, top - PAD), min(width, right + PAD), min(height, bottom + PAD)


def detect_frame(predict, frame: np.ndarray, region: tuple[int, int, int, int]) -> list[list[float]]:
    """Detections [x1, y1, x2, y2, score] for one BGR frame, in full-frame coordinates."""
    left, top, right, bottom = region
    boxes, scores = predict(np.ascontiguousarray(frame[top:bottom, left:right]))
    # Shift region-local boxes into full-frame coordinates.
    boxes = boxes + [left, top, left, top]
    return [[*map(float, boxes[i].round(2)), round(float(scores[i]), 4)] for i in range(len(boxes))]


def main():
    args = parse_args()
    predict = load_predictor(args.weights, args.threshold)

    for dataset in args.datasets:
        path = VIDEOS[dataset]
        cap = cv2.VideoCapture(path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        region = detection_region(path, width, height)
        print(f'{dataset}: region {region}', flush=True)

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
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
            frame_idx = 0
            with open(output + '.tmp', 'w') as f:
                for source_frame in range(start, end):
                    ok, frame = cap.read()
                    if not ok:
                        break
                    detections = detect_frame(predict, frame, region)
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
