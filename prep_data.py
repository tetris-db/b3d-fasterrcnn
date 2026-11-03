"""Turn B3D's labeled crops into fixed-scale 704x704 training tiles in RF-DETR's COCO layout.

Crops are never resized. Crops larger than a tile are cut into overlapping windows,
and every window is padded with black to TILE x TILE. Boxes are clipped to their window
and kept only if at least MIN_VISIBLE of the box remains (the ground-truth rule too).
B3D's own split mixes crops of one frame across splits, so this re-splits by crop group
(the file-name prefix), keeping near-duplicate frames on one side.
"""
import argparse
import json
import os
import random
import shutil

import cv2
import numpy as np

from b3d import TILE, VISION, tile_origins


# Fraction of a box that must stay inside a window for the box to be kept.
MIN_VISIBLE = 0.25


def parse_args():
    parser = argparse.ArgumentParser(description='Prepare fixed-scale RF-DETR training tiles from B3D crops')
    parser.add_argument('--out', default='/out/dataset')
    parser.add_argument('--valid-frac', type=float, default=0.1)
    parser.add_argument('--test-frac', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=0)
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


def split_groups(crops: list[tuple[str, dict, list[dict]]], valid_frac: float, test_frac: float, seed: int) -> dict[str, str]:
    """Map each crop group (file-name prefix) to 'train', 'valid', or 'test' by whole groups."""
    # Count crops per group.
    sizes: dict[str, int] = {}
    for _, image, _ in crops:
        group = image['file_name'].split('_')[0]
        sizes[group] = sizes.get(group, 0) + 1
    # Shuffle groups deterministically, then fill test, then valid, up to their target crop counts.
    groups = sorted(sizes)
    random.Random(seed).shuffle(groups)
    targets = [('test', test_frac * len(crops)), ('valid', valid_frac * len(crops))]
    assignment, filled = {}, 0
    for group in groups:
        if targets and filled >= targets[0][1]:
            targets.pop(0)
            filled = 0
        assignment[group] = targets[0][0] if targets else 'train'
        filled += sizes[group] if targets else 0
    return assignment


def window_boxes(annotations: list[dict], x0: int, y0: int, size: int = TILE) -> list[list[float]]:
    """Boxes (COCO xywh) clipped to the size x size window at (x0, y0), dropping those under MIN_VISIBLE."""
    kept = []
    for annotation in annotations:
        x, y, w, h = annotation['bbox']
        # Clip the box to the window and measure what fraction survives.
        left, top = max(x, x0), max(y, y0)
        right, bottom = min(x + w, x0 + size), min(y + h, y0 + size)
        if right <= left or bottom <= top or (right - left) * (bottom - top) < MIN_VISIBLE * w * h:
            continue
        kept.append([left - x0, top - y0, right - left, bottom - top])
    return kept


def main():
    args = parse_args()
    crops = load_crops()
    assignment = split_groups(crops, args.valid_frac, args.test_frac, args.seed)

    # Start each split from an empty directory with a single 'vehicle' category.
    datasets = {}
    for split in ['train', 'valid', 'test']:
        shutil.rmtree(os.path.join(args.out, split), ignore_errors=True)
        os.makedirs(os.path.join(args.out, split))
        datasets[split] = {'images': [], 'annotations': [], 'categories': [{'id': 1, 'name': 'vehicle', 'supercategory': 'none'}]}

    for path, image, annotations in crops:
        split = assignment[image['file_name'].split('_')[0]]
        dataset = datasets[split]
        crop = cv2.imread(path)
        height, width = crop.shape[:2]
        for y0 in tile_origins(height):
            for x0 in tile_origins(width):
                # Cut the window at native scale and pad it with black to TILE x TILE.
                window = np.zeros((TILE, TILE, 3), dtype=np.uint8)
                piece = crop[y0:y0 + TILE, x0:x0 + TILE]
                window[:piece.shape[0], :piece.shape[1]] = piece
                # Skip windows that are only the black fill around rotated crops.
                if not window.any():
                    continue
                name = f"{image['file_name'][:-4]}_{x0}_{y0}.jpg"
                cv2.imwrite(os.path.join(args.out, split, name), window, [cv2.IMWRITE_JPEG_QUALITY, 95])
                image_id = len(dataset['images'])
                dataset['images'].append({'id': image_id, 'file_name': name, 'width': TILE, 'height': TILE})
                # Windows without boxes stay in as negatives.
                for box in window_boxes(annotations, x0, y0):
                    dataset['annotations'].append({
                        'id': len(dataset['annotations']), 'image_id': image_id, 'category_id': 1,
                        'bbox': box, 'area': box[2] * box[3], 'iscrowd': 0,
                    })

    # Write each split's annotations where RF-DETR looks for them.
    for split, dataset in datasets.items():
        with open(os.path.join(args.out, split, '_annotations.coco.json'), 'w') as f:
            json.dump(dataset, f)
        groups = sorted(g for g, s in assignment.items() if s == split)
        empty = len(dataset['images']) - len({a['image_id'] for a in dataset['annotations']})
        print(f"{split}: {len(dataset['images'])} tiles ({empty} without boxes), "
              f"{len(dataset['annotations'])} boxes, groups {' '.join(groups)}")


if __name__ == '__main__':
    main()
