"""Match each B3D crop group to our four test videos by SIFT + RANSAC on the static scene."""
import argparse
import json
import os

import cv2
import numpy as np

from b3d import VIDEOS, VISION


# Lowe ratio-test threshold for SIFT descriptor matches.
RATIO = 0.75


def parse_args():
    parser = argparse.ArgumentParser(description='Match B3D crop groups to our test videos')
    parser.add_argument('--crops-per-group', type=int, default=3)
    parser.add_argument('--out', default='/out/groups.json')
    return parser.parse_args()


def inliers(sift, matcher, crop: np.ndarray, frame_features) -> int:
    """Number of RANSAC homography inliers between a crop and a video frame."""
    keypoints, descriptors = sift.detectAndCompute(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), None)
    frame_keypoints, frame_descriptors = frame_features
    if descriptors is None or len(keypoints) < 8:
        return 0
    # Keep matches that pass Lowe's ratio test.
    good = [m for m, n in matcher.knnMatch(descriptors, frame_descriptors, k=2) if m.distance < RATIO * n.distance]
    if len(good) < 8:
        return 0
    # Count matches consistent with one planar mapping (rotation, scale, and offset of the crop).
    source = np.float32([keypoints[m.queryIdx].pt for m in good])
    target = np.float32([frame_keypoints[m.trainIdx].pt for m in good])
    _, mask = cv2.findHomography(source, target, cv2.RANSAC, 8.0)
    return int(mask.sum()) if mask is not None else 0


def main():
    args = parse_args()
    sift = cv2.SIFT_create()
    matcher = cv2.BFMatcher()

    # SIFT features of one mid-video frame per target video.
    frames = {}
    for dataset, path in VIDEOS.items():
        cap = cv2.VideoCapture(path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) // 2)
        ok, frame = cap.read()
        assert ok, f'Could not read {path}'
        cap.release()
        frames[dataset] = sift.detectAndCompute(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), None)

    # The largest crops of each group carry the most static scene.
    crops: dict[str, list[tuple[int, str]]] = {}
    for split in ['train', 'val', 'test']:
        for image in json.load(open(os.path.join(VISION, 'annotations', f'{split}.json')))['images']:
            group = image['file_name'].split('_')[0]
            crops.setdefault(group, []).append((image['width'] * image['height'], os.path.join(VISION, 'images', split, image['file_name'])))

    result = {}
    for group in sorted(crops):
        paths = [p for _, p in sorted(crops[group], reverse=True)[:args.crops_per_group]]
        # Median over the group's crops, per video, so one odd crop cannot decide the match.
        scores = {d: int(np.median([inliers(sift, matcher, cv2.imread(p), f) for p in paths])) for d, f in frames.items()}
        best = max(scores, key=scores.get)
        result[group] = {'crops': len(crops[group]), 'scores': scores, 'best': best}
        print(group, len(crops[group]), scores, '->', best, flush=True)
    with open(args.out, 'w') as f:
        json.dump(result, f, indent=2)


if __name__ == '__main__':
    main()
