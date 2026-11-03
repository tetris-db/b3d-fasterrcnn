"""Draw labels on a sample of images: B3D's raw crops, or prepared tiles with --dataset."""
import argparse
import json
import os
import random

import cv2

from b3d import VISION


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize B3D training labels')
    parser.add_argument('--dataset', default=None)
    parser.add_argument('--count', type=int, default=24)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out', default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    # Raw B3D keeps annotations/<split>.json and images/<split>/; prepared tiles keep both in <split>/.
    if args.dataset:
        sources = {s: (os.path.join(args.dataset, s, '_annotations.coco.json'), os.path.join(args.dataset, s))
                   for s in ['train', 'valid', 'test']}
    else:
        sources = {s: (os.path.join(VISION, 'annotations', f'{s}.json'), os.path.join(VISION, 'images', s))
                   for s in ['train', 'val', 'test']}
    out = args.out or ('/out/viz/prepared' if args.dataset else '/out/viz/labels')
    os.makedirs(out, exist_ok=True)
    rng = random.Random(args.seed)

    cocos = {split: json.load(open(path)) for split, (path, _) in sources.items()}
    total = sum(len(coco['images']) for coco in cocos.values())
    for split, coco in cocos.items():
        # Index this split's boxes by image id.
        boxes: dict[int, list[dict]] = {}
        for annotation in coco['annotations']:
            boxes.setdefault(annotation['image_id'], []).append(annotation)
        # Sample images in proportion to split size (at least two per split).
        share = max(2, round(args.count * len(coco['images']) / total))
        for image in rng.sample(coco['images'], share):
            picture = cv2.imread(os.path.join(sources[split][1], image['file_name']))
            for annotation in boxes.get(image['id'], []):
                x, y, w, h = annotation['bbox']
                # Green for visible vehicles, orange for ones CVAT marked occluded.
                color = (0, 140, 255) if annotation.get('attributes', {}).get('occluded') else (0, 255, 0)
                cv2.rectangle(picture, (round(x), round(y)), (round(x + w), round(y + h)), color, 2)
            # Name files so the split and box count are visible in a file browser.
            name = f"{split}_{image['file_name'][:-4]}_{len(boxes.get(image['id'], []))}boxes.jpg"
            cv2.imwrite(os.path.join(out, name), picture)
            print(name, f"{image['width']}x{image['height']}")


if __name__ == '__main__':
    main()
