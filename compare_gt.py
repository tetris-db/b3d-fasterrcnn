"""Agreement between the new pre-label tracks and the old RetinaNet ground truth (003_groundtruth).

The old tracks are rebuilt from the old 30 FPS detections with the same Cython OC-SORT run as p003,
checked against p003's saved tracks on every clip, then sampled at the frames the experiment clips actually contain (even source frames). On
clips whose segment starts on an odd frame, p003 sampled one source frame early, so this
realigns them. Agreement is scored with tetris's own TrackEval setup (as in
tetris/evaluation/p110_accuracy_compute.py), with the old tracks in the ground-truth role.

--fps 30 compares at the source frame rate instead: the old reference is p003's 30 FPS tracking
before its subsampling (rebuilt from the old 30 FPS detections), against tracking30.jsonl from
track.py --fps 30. Both cover every source frame of the segment, so no realignment is needed.
"""
import argparse
import json
import os
import sys
import warnings

import numpy as np

from detect import TEST_SEGMENTS, segment_range
from track import TETRIS, track

sys.path.insert(0, os.path.join(TETRIS, 'modules', 'TrackEval'))
import trackeval  # noqa: E402
from trackeval.metrics import CLEAR, HOTA  # noqa: E402
from polyis.trackeval.dataset import Dataset  # noqa: E402
from polyis.utilities import save_tracking_results  # noqa: E402


# Source frame counts of the four B3D videos, for each segment's start frame.
TOTAL_FRAMES = {'jnc0': 38695, 'jnc2': 33774, 'jnc6': 37391, 'jnc7': 37625}


def parse_args():
    parser = argparse.ArgumentParser(description='Compare new pre-label tracks with the old ground truth')
    parser.add_argument('--old', default='/out/old-gt')
    parser.add_argument('--new', default='/out/groundtruth')
    parser.add_argument('--out', default='/out/compare')
    parser.add_argument('--datasets', nargs='+', default=list(TOTAL_FRAMES))
    parser.add_argument('--fps', type=int, choices=[15, 30], default=15)
    return parser.parse_args()


def old_tracks(old_dir: str, dataset: str, segment: int) -> dict[int, list[list[float]]]:
    """Old tracks at every source frame: p003's 30 FPS OC-SORT run on the old detections, before subsampling.

    p003 saved only every 2nd frame (source frames start + 2k), so the rebuild must reproduce
    those saved tracks exactly (same IDs, boxes to 1e-3) before the other frames are trusted.
    """
    clip = os.path.join(old_dir, dataset, 'execution', f'te{segment:02d}.mp4', '003_groundtruth')
    detections = [row['detections'] for row in map(json.loads, open(os.path.join(clip, 'detection.jsonl')))]
    tracks = track(detections)
    saved = [row['tracks'] for row in map(json.loads, open(os.path.join(clip, 'tracking.jsonl')))]
    for k, expected in enumerate(saved):
        mine = sorted(map(tuple, np.round(tracks.get(2 * k, []), 3).tolist()))
        assert mine == sorted(map(tuple, np.round(expected, 3).tolist())), f'{dataset} te{segment:02d}: frame {2 * k} differs from p003'
    return tracks


def evaluate(gt_dir: str, track_dir: str, seqs: list[str], out: str, track_file: str = 'tracking.jsonl') -> dict:
    """Run TrackEval HOTA and CLEAR the way p110 does; return per-sequence vehicle results."""
    os.makedirs(out, exist_ok=True)
    config = {
        'output_fol': out, 'output_sub_fol': 'agreement', 'input_gt': 'tracking.jsonl', 'input_track': track_file, 'skip': 1,
        'tracker': 'new', 'seq_list': seqs, 'input_dir': track_dir, 'input_gt_dir': gt_dir, 'input_track_dir': track_dir,
    }
    evaluator = trackeval.Evaluator({
        'USE_PARALLEL': False, 'BREAK_ON_ERROR': True, 'PRINT_RESULTS': False, 'PRINT_CONFIG': False,
        'TIME_PROGRESS': False, 'OUTPUT_SUMMARY': False, 'OUTPUT_DETAILED': False, 'PLOT_CURVES': False,
        'OUTPUT_EMPTY_CLASSES': False, 'LOG_ON_ERROR': os.path.join(out, 'LOG.txt'),
    })
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        results = evaluator.evaluate([Dataset(config)], [HOTA(), CLEAR({'THRESHOLD': 0.5, 'PRINT_CONFIG': False})])
    assert results[1]['Dataset']['sort'] == 'Success', results[1]
    return {seq: r['vehicle'] for seq, r in results[0]['Dataset']['sort'].items()}


def summary(r: dict) -> dict:
    """Scalar HOTA terms (averaged over localization thresholds, as TrackEval reports) and CLEAR counts."""
    hota, clear = r['HOTA'], r['CLEAR']
    return {
        'HOTA': float(np.mean(hota['HOTA'])), 'DetA': float(np.mean(hota['DetA'])),
        'AssA': float(np.mean(hota['AssA'])), 'LocA': float(np.mean(hota['LocA'])),
        'recall@0.5': float(clear['CLR_Re']), 'precision@0.5': float(clear['CLR_Pr']),
        'old_only': int(clear['CLR_FN']), 'new_only': int(clear['CLR_FP']), 'matched': int(clear['CLR_TP']),
    }


def main():
    args = parse_args()
    report = {}
    for dataset in args.datasets:
        seqs = [f'te{s:02d}.mp4' for s in TEST_SEGMENTS]
        track_file = 'tracking30.jsonl' if args.fps == 30 else 'tracking.jsonl'
        gt_dir = os.path.join(args.out, 'old-30fps' if args.fps == 30 else 'old-aligned', dataset)
        for segment, seq in zip(TEST_SEGMENTS, seqs):
            frames = sum(1 for _ in open(os.path.join(args.new, dataset, seq, track_file)))
            tracks = old_tracks(args.old, dataset, segment)
            # At 15 FPS, clip frame k is source frame start + (start mod 2) + 2k; p003 sampled start + 2k.
            offset, step = (0, 1) if args.fps == 30 else (segment_range(TOTAL_FRAMES[dataset], segment)[0] % 2, 2)
            os.makedirs(os.path.join(gt_dir, seq), exist_ok=True)
            save_tracking_results({k: tracks.get(offset + step * k, []) for k in range(frames)}, os.path.join(gt_dir, seq, 'tracking.jsonl'))
        results = evaluate(gt_dir, os.path.join(args.new, dataset), seqs, os.path.join(args.out, 'trackeval', dataset), track_file)
        report[dataset] = {seq: summary(r) for seq, r in results.items()}
        for seq, s in report[dataset].items():
            print(f"{dataset} {seq:12s} HOTA {s['HOTA']:.3f}  DetA {s['DetA']:.3f}  AssA {s['AssA']:.3f}  LocA {s['LocA']:.3f}  "
                  f"recall {s['recall@0.5']:.3f}  precision {s['precision@0.5']:.3f}  old-only {s['old_only']}  new-only {s['new_only']}", flush=True)
    with open(os.path.join(args.out, 'agreement.json'), 'w') as f:
        json.dump(report, f, indent=2)


if __name__ == '__main__':
    main()
