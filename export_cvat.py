"""Export pre-label tracks as CVAT tasks for human review, one per B3D test-split clip.

Each task is <out>/<dataset>_teNN/ with the experiment clip (create the CVAT task from this
video) and annotations.zip in CVAT's "MOT 1.1" import format (gt/gt.txt, gt/labels.txt),
which CVAT loads as tracks. Track IDs are the ones in tracking.jsonl and in viz_tracks.py's
videos, so a reviewer can cross-reference them. MOT frames are 1-based; boxes are in the
clip's 1080x720 pixels.
"""
import argparse
import json
import os
import shutil
import zipfile

from b3d import VIDEOS
from check_alignment import DATASETS
from detect import TEST_SEGMENTS

LABEL = 'vehicle'


def parse_args():
    parser = argparse.ArgumentParser(description='Export pre-label tracks as CVAT MOT 1.1 tasks')
    parser.add_argument('--tracks', default='/out/groundtruth')
    parser.add_argument('--datasets', nargs='+', default=list(VIDEOS))
    parser.add_argument('--segments', nargs='+', type=int, default=TEST_SEGMENTS)
    parser.add_argument('--out', default='/out/cvat')
    return parser.parse_args()


def mot_lines(tracking: str) -> list[str]:
    """MOT rows: frame, id, left, top, width, height, not_ignored, class_id, visibility."""
    lines = []
    for row in map(json.loads, open(tracking)):
        for track_id, x1, y1, x2, y2 in row['tracks']:
            lines.append(f"{row['frame_idx'] + 1},{int(track_id)},{x1:.2f},{y1:.2f},{x2 - x1:.2f},{y2 - y1:.2f},1,1,1.0")
    return lines


def main():
    args = parse_args()
    for dataset in args.datasets:
        for segment in args.segments:
            name = f'te{segment:02d}.mp4'
            task = os.path.join(args.out, f'{dataset}_te{segment:02d}')
            os.makedirs(task, exist_ok=True)
            shutil.copy(os.path.join(DATASETS, dataset, 'test', name), os.path.join(task, f'{dataset}_{name}'))
            lines = mot_lines(os.path.join(args.tracks, dataset, name, 'tracking.jsonl'))
            with zipfile.ZipFile(os.path.join(task, 'annotations.zip'), 'w', zipfile.ZIP_DEFLATED) as z:
                z.writestr('gt/gt.txt', '\n'.join(lines) + '\n')
                z.writestr('gt/labels.txt', LABEL + '\n')
            print(f'{task}: {len(lines)} boxes, {len({line.split(",")[1] for line in lines})} tracks', flush=True)


if __name__ == '__main__':
    main()
