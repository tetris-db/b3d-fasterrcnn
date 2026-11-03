"""B3D's labeled crops as detectron2 COCO datasets, at native scale and without padding.

Two splits:
  groups: the same crop-group split as prep_data.py (RF-DETR's data), so held-out scores compare.
  b3d:    B3D's own train/val/test JSONs (which mix crops of one frame across splits).
Crops are kept as-is (files copied byte for byte), except crops longer than MAX_SIDE on a
side, which would not fit in GPU memory at native scale: those are cut into overlapping
windows of at most MAX_SIDE (no resizing, no padding), with boxes clipped as in prep_data.py.
"""
import argparse
import json
import os
import shutil

import cv2

from b3d import OVERLAP, VISION, tile_origins
from prep_data import load_crops, split_groups, window_boxes


# Longest crop side trained at native scale in one piece (2 x the 704 detector tile).
MAX_SIDE = 1408


def parse_args():
    parser = argparse.ArgumentParser(description='B3D crops as native-scale detectron2 datasets')
    parser.add_argument('--split', choices=['groups', 'b3d'], required=True)
    parser.add_argument('--out', default=None)
    parser.add_argument('--seed', type=int, default=0)
    return parser.parse_args()


def windows(length: int) -> list[tuple[int, int]]:
    """(start, size) of windows covering [0, length), each at most MAX_SIDE long."""
    if length <= MAX_SIDE:
        return [(0, length)]
    return [(start, MAX_SIDE) for start in tile_origins(length, MAX_SIDE, OVERLAP)]


def main():
    args = parse_args()
    out = args.out or f'/out/dataset-crops-{args.split}'
    crops = load_crops()
    # Split name of every crop: by crop group (as for RF-DETR), or B3D's own split directory.
    if args.split == 'groups':
        assignment = split_groups(crops, 0.1, 0.1, args.seed)
        split_of = [assignment[image['file_name'].split('_')[0]] for _, image, _ in crops]
    else:
        split_of = [{'train': 'train', 'val': 'valid', 'test': 'test'}[path.split(os.sep)[-2]] for path, _, _ in crops]

    datasets = {}
    for split in ['train', 'valid', 'test']:
        shutil.rmtree(os.path.join(out, split), ignore_errors=True)
        os.makedirs(os.path.join(out, split))
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
                shutil.copyfile(path, os.path.join(out, split, name))
                boxes = [a['bbox'] for a in annotations]
            else:
                # Oversized: a native-scale window, boxes clipped to it.
                window = picture[y0:y0 + h, x0:x0 + w]
                if not window.any():
                    continue
                name = f"{image['file_name'][:-4]}_{x0}_{y0}.jpg"
                cv2.imwrite(os.path.join(out, split, name), window, [cv2.IMWRITE_JPEG_QUALITY, 95])
                # The uncut axis spans the whole crop, so clipping to a MAX_SIDE square is exact.
                boxes = window_boxes(annotations, x0, y0, MAX_SIDE)
            image_id = len(dataset['images'])
            dataset['images'].append({'id': image_id, 'file_name': name, 'width': w, 'height': h})
            for box in boxes:
                dataset['annotations'].append({
                    'id': len(dataset['annotations']), 'image_id': image_id, 'category_id': 1,
                    'bbox': box, 'area': box[2] * box[3], 'iscrowd': 0,
                })

    for split, dataset in datasets.items():
        with open(os.path.join(out, split, '_annotations.coco.json'), 'w') as f:
            json.dump(dataset, f)
        sides = [max(i['width'], i['height']) for i in dataset['images']]
        print(f"{split}: {len(dataset['images'])} images, {len(dataset['annotations'])} boxes, longest side {max(sides)}")


if __name__ == '__main__':
    main()
