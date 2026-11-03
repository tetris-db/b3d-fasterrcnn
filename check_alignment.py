"""Check that detect.py's frame k is frame k of the experiment clip (tetris p000 output).

Re-creates p000's transform (crop, mask, rotate vertical regions, resize to 1080x720) on
source frames s-1, s, s+1 around detect.py's source frame s, and compares each to the
experiment frame. The smallest difference should be at offset 0.
"""
import argparse
import os

import cv2
import numpy as np

from b3d import VIDEOS, load_crop_and_mask
from detect import STEP, TEST_SEGMENTS, segment_range


# Copy of ace's current experiment clips (polyis-data/datasets/<dataset>/test); bombe's own copy is stale.
DATASETS = '/out/experiment-clips'
# p000's output resolution.
WIDTH, HEIGHT = 1080, 720


def parse_args():
    parser = argparse.ArgumentParser(description='Check frame alignment with the experiment clips')
    parser.add_argument('--datasets', nargs='+', default=list(VIDEOS))
    parser.add_argument('--segments', nargs='+', type=int, default=TEST_SEGMENTS)
    return parser.parse_args()


def p000_transform(frame: np.ndarray, box: tuple[int, int, int, int], mask: np.ndarray) -> np.ndarray:
    """p000's per-frame transform: crop, mask, rotate vertical regions clockwise, resize."""
    left, top, right, bottom = box
    region = frame[top:bottom, left:right] * (mask[..., None] > 0)
    if region.shape[0] > region.shape[1]:
        region = np.rot90(region, k=-1)
    return cv2.resize(np.ascontiguousarray(region), (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR)


def read(cap: cv2.VideoCapture, index: int) -> np.ndarray:
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    assert ok, f'Could not read frame {index}'
    return frame


def main():
    args = parse_args()
    for dataset in args.datasets:
        source = cv2.VideoCapture(VIDEOS[dataset])
        total = int(source.get(cv2.CAP_PROP_FRAME_COUNT))
        box, mask = load_crop_and_mask(VIDEOS[dataset], int(source.get(3)), int(source.get(4)))
        for segment in args.segments:
            clip = cv2.VideoCapture(os.path.join(DATASETS, dataset, 'test', f'te{segment:02d}.mp4'))
            count = int(clip.get(cv2.CAP_PROP_FRAME_COUNT))
            start, _ = segment_range(total, segment)
            # detect.py's source frame for clip frame k: the k-th even index at or after start.
            first = start + (-start % STEP)
            report = []
            for k in [0, count // 2, count - 1]:
                target = read(clip, k).astype(np.float32)
                source_frame = first + STEP * k
                # Mean absolute pixel difference at offsets -1, 0, +1 source frames.
                errors = [float(np.abs(p000_transform(read(source, source_frame + d), box, mask) - target).mean()) for d in (-1, 0, 1)]
                report.append(f'k={k}: ' + ' '.join(f'{e:.1f}' for e in errors) + (' ok' if np.argmin(errors) == 1 else ' MISALIGNED'))
            print(f'{dataset} te{segment:02d} (start {start}, {count} frames): ' + ' | '.join(report), flush=True)
            clip.release()
        source.release()


if __name__ == '__main__':
    main()
