"""Checks for the cross-tile merge in detect.py. Run: ./run test_detect.py"""
import numpy as np

from b3d import TILE
from detect import fragment_flags, merge, segment_range


def test_fragment_flags():
    # Middle tile of a large region: all four edges are shared with other tiles.
    boxes = np.array([[0, 100, 50, 150], [300, 300, 350, 350], [650, 100, TILE, 150]], dtype=float)
    assert fragment_flags(boxes, 448, 448, 2000, 2000).tolist() == [True, False, True]
    # Top-left tile: its left and top edges are the region border, so touching them is not a fragment.
    assert fragment_flags(boxes, 0, 0, 2000, 2000).tolist() == [False, False, True]


def test_merge():
    boxes = np.array([
        [100, 100, 200, 150],  # 0: complete car
        [102, 101, 199, 151],  # 1: same car from a neighboring tile (duplicate)
        [100, 100, 140, 150],  # 2: fragment of car 0 cut at a tile edge
        [210, 100, 300, 150],  # 3: separate adjacent car
        [900, 100, 1300, 180],  # 4: long bus seen only as a fragment
    ], dtype=float)
    scores = np.array([0.9, 0.8, 0.95, 0.7, 0.6])
    fragments = np.array([False, False, True, False, True])
    assert sorted(merge(boxes, scores, fragments).tolist()) == [0, 3, 4]
    # No detections at all.
    assert merge(np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=bool)).tolist() == []


def test_segment_range():
    # Matches tetris get_segment_frame_range for jnc00 (38,695 frames, 18 segments).
    assert segment_range(38695, 1) == (2149, 4299)
    assert segment_range(38695, 17)[1] == 38695


if __name__ == '__main__':
    test_fragment_flags()
    test_merge()
    test_segment_range()
    print('ok')
