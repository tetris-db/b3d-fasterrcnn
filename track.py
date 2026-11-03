"""Turn detect.py output into ground-truth detections and OC-SORT tracks in experiment coordinates.

Per clip, each 4K detection is mapped into the 1080x720 experiment frame the way
tetris/preprocess/p000_preprocess_dataset.py builds it (crop, rotate vertical regions,
resize). It is shrunk to its visible extent inside the road mask and kept if its score is at
least --threshold and at least MIN_VISIBLE of its pixels lie in the mask. The kept boxes are
tracked at 30 FPS over every source frame of the segment, with the same Cython OC-SORT code,
settings, and gap interpolation as tetris/preprocess/p003_preprocess_groundtruth_tracking.py,
imported read-only from /tetris. Runs in the polyis image (IMAGE=polyis-polyis ./run track.py),
whose Python matches the compiled tracker.

Output mirrors the cache's 003_groundtruth layout under <out>/<dataset>/teNN.mp4/:
  detection30.jsonl, tracking30.jsonl  every source frame (30 FPS)
  detection.jsonl, tracking.jsonl      the same, sampled at the experiment clip's frames (15 FPS);
                                       these are the pre-labels given to human review
"""
import argparse
import json
import math
import os
import sys

import cv2
import numpy as np
import yaml

from b3d import TEST_SEGMENTS, VIDEOS, load_crop_and_mask, segment_range

# tetris checkout, mounted read-only by ./run.
TETRIS = '/tetris'
sys.path.insert(0, TETRIS)
# p003's tracker: the compiled Cython OC-SORT, built for the polyis image's Python (IMAGE=polyis-polyis ./run).
from polyis.tracker.ocsort.cython.ocsort_wrapper import OCSort  # noqa: E402
from polyis.utilities import register_tracked_detections, save_tracking_results  # noqa: E402


# Experiment frame size (p000 output).
WIDTH, HEIGHT = 1080, 720
# Fraction of a detection's pixels that must lie inside the road mask (the visible-extent rule).
MIN_VISIBLE = 0.25


def parse_args():
    parser = argparse.ArgumentParser(description='Experiment-coordinate GT detections and OC-SORT tracks')
    parser.add_argument('--detections', default='/out/detections30-frcnn-b3d')
    parser.add_argument('--out', default='/out/gt30-frcnn-b3d-t0.2')
    parser.add_argument('--threshold', type=float, default=0.2)
    parser.add_argument('--datasets', nargs='+', default=list(VIDEOS))
    parser.add_argument('--segments', nargs='+', type=int, default=TEST_SEGMENTS)
    return parser.parse_args()


def create_ocsort() -> OCSort:
    """OC-SORT with tetris's settings; mirrors create_tracker('ocsortcython') in tetris polyis/utilities.py, as p003 uses."""
    config = yaml.safe_load(open(os.path.join(TETRIS, 'configs', 'trackers.yaml')))['ocsort']
    return OCSort(
        img_size=(HEIGHT, WIDTH),
        det_thresh=config['det_thresh'],
        max_age=config['max_age'],
        min_hits=config['min_hits'],
        iou_threshold=config['iou_threshold'],
        delta_t=config['delta_t'],
        asso_func=config['asso_func'],
        inertia=config['inertia'],
        use_byte=config['use_byte'],
    )


def track(frames: list[list[list[float]]]) -> dict[int, list[list[float]]]:
    """Track per-frame [x1, y1, x2, y2, score] lists; the same loop as p003's track()."""
    tracker = create_ocsort()
    frame_tracks: dict[int, list[list[float]]] = {}
    trajectories: dict[int, list] = {}
    for frame_idx, detections in enumerate(frames):
        dets = np.array(detections, dtype=np.float64)[:, :5] if detections else np.empty((0, 5))
        register_tracked_detections(tracker.update(dets), frame_idx, frame_tracks, trajectories)
    return frame_tracks


def to_experiment(detection: list[float], box: tuple[int, int, int, int], mask: np.ndarray) -> list[float] | None:
    """Visible extent of a 4K detection in experiment coordinates, or None if under MIN_VISIBLE."""
    x1, y1, x2, y2 = detection[:4]
    left, top, _, _ = box
    height, width = mask.shape
    # Shift into crop-region coordinates.
    x1, y1, x2, y2 = x1 - left, y1 - top, x2 - left, y2 - top
    # Pixels of the box that fall inside the crop region.
    px1, py1 = max(0, math.floor(x1)), max(0, math.floor(y1))
    px2, py2 = min(width, math.ceil(x2)), min(height, math.ceil(y2))
    if px2 <= px1 or py2 <= py1:
        return None
    inside = mask[py1:py2, px1:px2] > 0
    if inside.sum() < MIN_VISIBLE * (x2 - x1) * (y2 - y1):
        return None
    # Bounding box of the in-mask pixels, never larger than the detection itself.
    rows, cols = np.flatnonzero(inside.any(axis=1)), np.flatnonzero(inside.any(axis=0))
    vx1, vy1 = max(x1, px1 + cols[0]), max(y1, py1 + rows[0])
    vx2, vy2 = min(x2, px1 + cols[-1] + 1), min(y2, py1 + rows[-1] + 1)
    # p000 rotates vertical regions clockwise (np.rot90 k=-1): (x, y) -> (height - y, x).
    if height > width:
        vx1, vy1, vx2, vy2 = height - vy2, vx1, height - vy1, vx2
        width, height = height, width
    # p000 then resizes the region to WIDTH x HEIGHT.
    sx, sy = WIDTH / width, HEIGHT / height
    return [vx1 * sx, vy1 * sy, vx2 * sx, vy2 * sy, detection[4]]


def main():
    args = parse_args()

    for dataset in args.datasets:
        cap = cv2.VideoCapture(VIDEOS[dataset])
        box, mask = load_crop_and_mask(VIDEOS[dataset], int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        for segment in args.segments:
            rows = [json.loads(line) for line in open(os.path.join(args.detections, dataset, f'te{segment:02d}.jsonl'))]
            assert [r['frame_idx'] for r in rows] == list(range(len(rows))), 'frames must be contiguous from 0'
            # Visible-extent boxes above the score threshold, per input frame.
            frames = []
            for row in rows:
                kept = [to_experiment(d, box, mask) for d in row['detections'] if d[4] >= args.threshold]
                frames.append([d for d in kept if d is not None])
            frame_tracks = track(frames)
            # Pad to every frame, as save_tracking_results only spans first..last tracked frame.
            for frame_idx in range(len(frames)):
                frame_tracks.setdefault(frame_idx, [])
            out = os.path.join(args.out, dataset, f'te{segment:02d}.mp4')
            os.makedirs(out, exist_ok=True)
            start, end = segment_range(total, segment)
            assert rows[0]['source_frame'] == start and len(rows) == end - start, f'{dataset} te{segment:02d}: not a full 30 FPS segment'
            write_detections(frames, os.path.join(out, 'detection30.jsonl'))
            save_tracking_results(dict(frame_tracks), os.path.join(out, 'tracking30.jsonl'))
            # Clip frame k is source frame start + (start mod 2) + 2k, i.e. input row (start mod 2) + 2k.
            sampled = list(range(start % 2, end - start, 2))
            frames = [frames[i] for i in sampled]
            frame_tracks = {k: frame_tracks[i] for k, i in enumerate(sampled)}
            write_detections(frames, os.path.join(out, 'detection.jsonl'))
            save_tracking_results(frame_tracks, os.path.join(out, 'tracking.jsonl'))
            ids = {t[0] for tracks in frame_tracks.values() for t in tracks}
            print(f'{dataset} te{segment:02d}: {len(rows)} frames tracked at 30 FPS, {len(frames)} clip frames, '
                  f'{sum(map(len, frames))} boxes, {len(ids)} tracks', flush=True)


def write_detections(frames: list[list[list[float]]], path: str):
    with open(path, 'w') as f:
        for frame_idx, detections in enumerate(frames):
            f.write(json.dumps({'frame_idx': frame_idx, 'detections': detections}) + '\n')


if __name__ == '__main__':
    main()
