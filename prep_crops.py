"""B3D's labeled crops as a detectron2 COCO dataset, at native scale and without padding.

Uses B3D's own train/val/test JSONs (the split the frcnn-b3d model was trained on).
Crops are kept as-is (files copied byte for byte), except crops longer than MAX_SIDE on a
side, which would not fit in GPU memory at native scale: those are cut into overlapping
windows of at most MAX_SIDE (no resizing, no padding), with boxes clipped to each window.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil

import cv2

from b3d import VISION


# Longest crop side trained at native scale in one piece.
MAX_SIDE = 1408
# Window overlap in pixels; covers the p99 longest side of B3D labeled vehicles (227 px).
OVERLAP = 256
# Fraction of a box that must stay inside a window for the box to be kept.
MIN_VISIBLE = 0.25


def parse_args():
    parser = argparse.ArgumentParser(description='B3D crops as a native-scale detectron2 dataset')
    parser.add_argument('--out', default='/out/dataset-crops-b3d')
    return parser.parse_args()


def load_crops() -> list[tuple[str, dict, list[dict]]]:
    """All B3D crops as (image path, COCO image entry, its annotations), across B3D's three splits."""
    crops = []
    for split in ['train', 'val', 'test']:
        coco = json.load(open(os.path.join(VISION, 'annotations', f'{split}.json')))
        # Index this split's boxes by image id.
        boxes: dict[int, list[dict]] = {}
        for annotation in coco['annotations']:
            boxes.setdefault(annotation['image_id'], []).append(annotation)
        for image in coco['images']:
            crops.append((os.path.join(VISION, 'images', split, image['file_name']), image, boxes.get(image['id'], [])))
    return crops


def window_origins(length: int) -> list[int]:
    """Evenly spaced window starts covering [0, length) with at least OVERLAP pixels shared."""
    # Fewest windows whose stride does not exceed MAX_SIDE - OVERLAP, with both edge windows flush.
    count = math.ceil((length - MAX_SIDE) / (MAX_SIDE - OVERLAP)) + 1
    return [round(i * (length - MAX_SIDE) / (count - 1)) for i in range(count)]


def windows(length: int) -> list[tuple[int, int]]:
    """(start, size) of windows covering [0, length), each at most MAX_SIDE long."""
    if length <= MAX_SIDE:
        return [(0, length)]
    return [(start, MAX_SIDE) for start in window_origins(length)]


def window_boxes(annotations: list[dict], x0: int, y0: int) -> list[list[float]]:
    """Boxes (COCO xywh) clipped to the MAX_SIDE square at (x0, y0), dropping those under MIN_VISIBLE."""
    kept = []
    for annotation in annotations:
        x, y, w, h = annotation['bbox']
        # Clip the box to the window and measure what fraction survives.
        left, top = max(x, x0), max(y, y0)
        right, bottom = min(x + w, x0 + MAX_SIDE), min(y + h, y0 + MAX_SIDE)
        if right <= left or bottom <= top or (right - left) * (bottom - top) < MIN_VISIBLE * w * h:
            continue
        kept.append([left - x0, top - y0, right - left, bottom - top])
    return kept


def main():
    args = parse_args()
    crops = load_crops()
    # Split name of every crop: B3D's own split directory.
    split_of = [{'train': 'train', 'val': 'valid', 'test': 'test'}[path.split(os.sep)[-2]] for path, _, _ in crops]

    datasets = {}
    for split in ['train', 'valid', 'test']:
        shutil.rmtree(os.path.join(args.out, split), ignore_errors=True)
        os.makedirs(os.path.join(args.out, split))
        datasets[split] = {'images': [], 'annotations': [], 'categories': [{'id': 1, 'name': 'vehicle', 'supercategory': 'none'}]}

    for (path, image, annotations), split in zip(crops, split_of):
        dataset = datasets[split]
        width, height = image['width'], image['height']
        pieces = [(x, y, w, h) for y, h in windows(height) for x, w in windows(width)]
        picture = cv2.imread(path) if len(pieces) > 1 else None
        for x0, y0, w, h in pieces:
            if len(pieces) == 1:
                # Small enough: keep the original file and its boxes untouched.
                name = image['file_name']
                shutil.copyfile(path, os.path.join(args.out, split, name))
                boxes = [a['bbox'] for a in annotations]
            else:
                # Oversized: a native-scale window, boxes clipped to it.
                window = picture[y0:y0 + h, x0:x0 + w]
                if not window.any():
                    continue
                name = f"{image['file_name'][:-4]}_{x0}_{y0}.jpg"
                cv2.imwrite(os.path.join(args.out, split, name), window, [cv2.IMWRITE_JPEG_QUALITY, 95])
                # The uncut axis spans the whole crop, so clipping to a MAX_SIDE square is exact.
                boxes = window_boxes(annotations, x0, y0)
            image_id = len(dataset['images'])
            dataset['images'].append({'id': image_id, 'file_name': name, 'width': w, 'height': h})
            for box in boxes:
                dataset['annotations'].append({
                    'id': len(dataset['annotations']), 'image_id': image_id, 'category_id': 1,
                    'bbox': box, 'area': box[2] * box[3], 'iscrowd': 0,
                })

    for split, dataset in datasets.items():
        with open(os.path.join(args.out, split, '_annotations.coco.json'), 'w') as f:
            json.dump(dataset, f)
        sides = [max(i['width'], i['height']) for i in dataset['images']]
        print(f"{split}: {len(dataset['images'])} images, {len(dataset['annotations'])} boxes, longest side {max(sides)}")


if __name__ == '__main__':
    main()
