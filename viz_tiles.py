"""Draw the planned 704x704 detector tiles over one frame of each B3D test video."""
import argparse
import os

import cv2
import numpy as np

from b3d import OVERLAP, TILE, VIDEOS, load_crop_and_mask, tiles


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize the inference tiling plan')
    parser.add_argument('--tile', type=int, default=TILE)
    parser.add_argument('--overlap', type=int, default=OVERLAP)
    parser.add_argument('--out', default='/out/viz/tiles')
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    # One color per tile so overlaps stay distinguishable.
    palette = [(0, 255, 0), (255, 0, 255), (0, 255, 255), (255, 128, 0), (0, 128, 255), (255, 255, 0)]

    for dataset, path in VIDEOS.items():
        # Read the middle frame at native resolution.
        cap = cv2.VideoCapture(path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) // 2)
        ok, frame = cap.read()
        assert ok, f'Could not read {path}'
        height, width = frame.shape[:2]
        cap.release()

        # Crop to the domain box and dim pixels outside the road mask.
        (left, top, right, bottom), mask = load_crop_and_mask(path, width, height)
        region = frame[top:bottom, left:right].copy()
        region[mask == 0] //= 4
        # Save one native-resolution tile to compare vehicle scale with the training crops.
        corners = tiles(mask, args.tile, args.overlap)
        x, y = corners[len(corners) // 2]
        cv2.imwrite(os.path.join(args.out, f'{dataset}_sample_tile.jpg'), region[y:y + args.tile, x:x + args.tile])

        # Outline every kept tile and number it.
        for i, (x, y) in enumerate(corners):
            color = palette[i % len(palette)]
            cv2.rectangle(region, (x + 3 * (i % 3), y + 3 * (i % 3)), (x + args.tile - 1, y + args.tile - 1), color, 6)
            cv2.putText(region, str(i), (x + 20, y + 70), cv2.FONT_HERSHEY_SIMPLEX, 2.5, color, 6)
        cv2.imwrite(os.path.join(args.out, f'{dataset}_tiles.jpg'), region)
        print(f'{dataset}: frame {width}x{height}, region {right - left}x{bottom - top} at ({left},{top}), '
              f'{len(corners)} tiles kept, masked-in {np.count_nonzero(mask) / mask.size:.0%}')


if __name__ == '__main__':
    main()
