"""Render the pre-label tracks over the B3D test-split experiment clips.

Uses tetris's own create_tracking_visualization: one color per track ID, an ID label,
and a fading tail of past box centers. Clips are the current experiment videos
(/out/experiment-clips, copied from ace), so the overlay shows exactly what is evaluated.
"""
import argparse
import json
import os
import sys
from multiprocessing import Pool

from detect import TEST_SEGMENTS

# tetris checkout, mounted read-only by ./run.
sys.path.insert(0, '/tetris')
from polyis.utilities import create_tracking_visualization  # noqa: E402

from check_alignment import DATASETS
from b3d import VIDEOS


def parse_args():
    parser = argparse.ArgumentParser(description='Render pre-label tracks on the test-split clips')
    parser.add_argument('--tracks', default='/out/groundtruth')
    parser.add_argument('--datasets', nargs='+', default=list(VIDEOS))
    parser.add_argument('--segments', nargs='+', type=int, default=TEST_SEGMENTS)
    parser.add_argument('--out', default='/out/viz/tracks')
    parser.add_argument('--workers', type=int, default=8)
    return parser.parse_args()


def render(job: tuple[str, str, str]):
    video, tracking, output = job
    tracks = {row['frame_idx']: row['tracks'] for row in map(json.loads, open(tracking))}
    create_tracking_visualization(video, tracks, output, speed_up=1, process_id=0)
    print(output, flush=True)


def main():
    args = parse_args()
    jobs = [
        (os.path.join(DATASETS, dataset, 'test', f'te{segment:02d}.mp4'),
         os.path.join(args.tracks, dataset, f'te{segment:02d}.mp4', 'tracking.jsonl'),
         os.path.join(args.out, f'{dataset}_te{segment:02d}.mp4'))
        for dataset in args.datasets for segment in args.segments
    ]
    with Pool(args.workers) as pool:
        pool.map(render, jobs)


if __name__ == '__main__':
    main()
