"""Contact sheet of one crop per B3D crop group, plus our four videos, to match groups to videos by eye."""
import argparse
import json
import os

import cv2
import numpy as np

from b3d import VIDEOS, VISION, load_crop_and_mask


# Thumbnail edge length in the contact sheet.
THUMB = 360


def parse_args():
    parser = argparse.ArgumentParser(description='Contact sheet of B3D crop groups and our test videos')
    parser.add_argument('--out', default='/out/viz/groups')
    return parser.parse_args()


def thumbnail(image: np.ndarray, label: str) -> np.ndarray:
    """Fit an image into a THUMB x THUMB black square and stamp a label on it."""
    scale = THUMB / max(image.shape[:2])
    small = cv2.resize(image, (round(image.shape[1] * scale), round(image.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    cell = np.zeros((THUMB, THUMB, 3), dtype=np.uint8)
    cell[:small.shape[0], :small.shape[1]] = small
    cv2.putText(cell, label, (8, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 3)
    return cell


def grid(cells: list[np.ndarray], columns: int) -> np.ndarray:
    """Tile cells row by row, padding the last row with black."""
    cells = cells + [np.zeros_like(cells[0])] * (-len(cells) % columns)
    return np.vstack([np.hstack(cells[i:i + columns]) for i in range(0, len(cells), columns)])


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    # Pick the largest crop of each group, since it shows the most scene context.
    largest: dict[str, tuple[int, str]] = {}
    for split in ['train', 'val', 'test']:
        for image in json.load(open(os.path.join(VISION, 'annotations', f'{split}.json')))['images']:
            group = image['file_name'].split('_')[0]
            area = image['width'] * image['height']
            if area > largest.get(group, (0, ''))[0]:
                largest[group] = (area, os.path.join(VISION, 'images', split, image['file_name']))
    cells = [thumbnail(cv2.imread(path), group) for group, (_, path) in sorted(largest.items())]
    cv2.imwrite(os.path.join(args.out, 'groups.jpg'), grid(cells, 8), [cv2.IMWRITE_JPEG_QUALITY, 85])

    # One mid-video frame of each target video, cropped to its mask box.
    refs = []
    for dataset, path in VIDEOS.items():
        cap = cv2.VideoCapture(path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) // 2)
        ok, frame = cap.read()
        assert ok, f'Could not read {path}'
        cap.release()
        (left, top, right, bottom), _ = load_crop_and_mask(path, frame.shape[1], frame.shape[0])
        refs.append(thumbnail(frame[top:bottom, left:right], dataset))
    cv2.imwrite(os.path.join(args.out, 'videos.jpg'), grid(refs, 4), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(f'{len(cells)} groups')


if __name__ == '__main__':
    main()
