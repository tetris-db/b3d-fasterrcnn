"""Draw detect.py output on a few frames of a clip, with the mask outline and the padded region."""
import argparse
import json
import os

import cv2

from b3d import VIDEOS, load_crop_and_mask
from detect import detection_region


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize detect.py output')
    parser.add_argument('detections')
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--frames', nargs='+', type=int, default=[0, 25, 49])
    parser.add_argument('--threshold', type=float, default=0.3)
    parser.add_argument('--out', default='/out/viz/detections')
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    rows = {r['frame_idx']: r for r in map(json.loads, open(args.detections))}
    cap = cv2.VideoCapture(VIDEOS[args.dataset])
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    (left, top, right, bottom), mask = load_crop_and_mask(VIDEOS[args.dataset], width, height)
    (rl, rt, rr, rb), _ = detection_region(VIDEOS[args.dataset], width, height)
    # Mask outline in full-frame coordinates.
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c + [left, top] for c in contours]

    for frame_idx in args.frames:
        row = rows[frame_idx]
        cap.set(cv2.CAP_PROP_POS_FRAMES, row['source_frame'])
        ok, frame = cap.read()
        assert ok
        # Yellow: road mask; white: crop box; gray: padded detection region.
        cv2.drawContours(frame, contours, -1, (0, 255, 255), 3)
        cv2.rectangle(frame, (left, top), (right, bottom), (255, 255, 255), 2)
        cv2.rectangle(frame, (rl, rt), (rr - 1, rb - 1), (128, 128, 128), 2)
        # Green: kept at --threshold; red: stored but below it.
        for x1, y1, x2, y2, score in row['detections']:
            color = (0, 255, 0) if score >= args.threshold else (0, 0, 255)
            cv2.rectangle(frame, (round(x1), round(y1)), (round(x2), round(y2)), color, 2)
        name = f"{args.dataset}_{os.path.basename(args.detections)[:-6]}_f{frame_idx}.jpg"
        cv2.imwrite(os.path.join(args.out, name), frame[rt:rb, rl:rr], [cv2.IMWRITE_JPEG_QUALITY, 90])
        kept = sum(d[4] >= args.threshold for d in row['detections'])
        print(name, f"{kept} >= {args.threshold}, {len(row['detections'])} stored")


if __name__ == '__main__':
    main()
