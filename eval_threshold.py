"""COCO AP, and recall/precision at several score thresholds, on the prepared held-out tiles.

Scores any of our detectors the same way, so they compare directly:
  --model rfdetr      RF-DETR checkpoint (--weights), run in the tetris-annotate image.
  --model detectron2  detectron2 run directory (--weights, with config.yaml and model_final.pth),
                      run in the tetris-annotate-d2 image (Python 3.7).
Images are fed at native scale by default. Options for checking Faster R-CNN's input:
  --scale S  (detectron2 only) resize each image by S before detection, map boxes back.
  --tile     cut each image into overlapping 704x704 tiles and merge them as detect.py does on
             video frames, instead of one whole-image pass (for crop datasets larger than a tile).
"""
import argparse
import json
import os

import cv2
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from b3d import TILE, tile_origins
from detect import fragment_flags, merge


# Minimum IoU for a detection to count as finding a labeled vehicle.
MATCH_IOU = 0.5


def parse_args():
    parser = argparse.ArgumentParser(description='COCO AP and recall/precision vs score threshold')
    parser.add_argument('--model', choices=['rfdetr', 'detectron2'], default='rfdetr')
    parser.add_argument('--weights', default='/out/runs/large-lowlr/checkpoint_best_total.pth')
    parser.add_argument('--dataset', default='/out/dataset')
    parser.add_argument('--splits', nargs='+', default=['test', 'valid'])
    parser.add_argument('--thresholds', nargs='+', type=float, default=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6])
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--scale', type=float, default=1.0)
    parser.add_argument('--tile', action='store_true')
    parser.add_argument('--out', default=None)
    return parser.parse_args()


def tiled(predict):
    """Wrap a predictor to run 704x704 overlapping tiles per image and merge them like detect.py."""
    def run(images):
        results = []
        for image in images:
            height, width = image.shape[:2]
            corners = [(x, y) for y in tile_origins(height) for x in tile_origins(width)]
            pieces = []
            for x, y in corners:
                # Pad edge tiles with black, as the training tiles were.
                piece = np.zeros((TILE, TILE, 3), dtype=np.uint8)
                part = image[y:y + TILE, x:x + TILE]
                piece[:part.shape[0], :part.shape[1]] = part
                pieces.append(piece)
            boxes, scores, fragments = [], [], []
            for (x, y), (b, s) in zip(corners, predict(pieces)):
                fragments.append(fragment_flags(b, x, y, width, height))
                boxes.append(b + [x, y, x, y])
                scores.append(s)
            boxes, scores, fragments = np.concatenate(boxes), np.concatenate(scores), np.concatenate(fragments)
            keep = merge(boxes, scores, fragments)
            # Clip to the image, since padded edge tiles can predict past it.
            results.append((np.clip(boxes[keep], 0, [width, height, width, height]), scores[keep]))
        return results
    return run


def load_predictor(model: str, weights: str, threshold: float, scale: float):
    """A function from a list of BGR images to a list of (xyxy boxes, scores) arrays."""
    assert model == 'detectron2' or scale == 1.0, 'RF-DETR resizes every input to 704; --scale applies to detectron2 only'
    if model == 'rfdetr':
        from rfdetr import RFDETRLarge
        detector = RFDETRLarge(pretrain_weights=weights)

        def predict(images):
            results = detector.predict([cv2.cvtColor(i, cv2.COLOR_BGR2RGB) for i in images], threshold=threshold)
            return [(np.asarray(r.xyxy, dtype=float).reshape(-1, 4), np.asarray(r.confidence, dtype=float).reshape(-1)) for r in results]
        return predict

    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor
    cfg = get_cfg()
    cfg.merge_from_file(os.path.join(weights, 'config.yaml'))
    cfg.MODEL.WEIGHTS = os.path.join(weights, 'model_final.pth')
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
    # The run's config already disables resizing (MIN_SIZE_TEST = 0); DefaultPredictor takes BGR.
    detector = DefaultPredictor(cfg)

    def predict(images):
        results = []
        for image in images:
            # Optional test-time rescale; boxes are mapped back to the input's pixels.
            if scale != 1.0:
                image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
            o = detector(image)['instances'].to('cpu')
            results.append((o.pred_boxes.tensor.numpy().astype(float) / scale, o.scores.numpy().astype(float)))
        return results
    return predict


def iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """IoU of one xyxy box against many."""
    width = np.clip(np.minimum(box[2], boxes[:, 2]) - np.maximum(box[0], boxes[:, 0]), 0, None)
    height = np.clip(np.minimum(box[3], boxes[:, 3]) - np.maximum(box[1], boxes[:, 1]), 0, None)
    inter = width * height
    return inter / ((box[2] - box[0]) * (box[3] - box[1]) + (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) - inter)


def coco_ap(annotation_path: str, detections: list) -> dict:
    """COCO box AP, AP50, AP75 with up to 500 detections per tile (dense tiles exceed COCO's 100)."""
    truth = COCO(annotation_path)
    evaluation = COCOeval(truth, truth.loadRes(detections) if detections else COCO(), 'bbox')
    evaluation.params.maxDets = [1, 100, 500]
    evaluation.evaluate()
    evaluation.accumulate()
    evaluation.summarize()
    return {'AP': float(evaluation.stats[0]), 'AP50': float(evaluation.stats[1]), 'AP75': float(evaluation.stats[2])}


def main():
    args = parse_args()
    predict = load_predictor(args.model, args.weights, min(args.thresholds), args.scale)
    if args.tile:
        predict = tiled(predict)
    report = {}

    for split in args.splits:
        annotation_path = os.path.join(args.dataset, split, '_annotations.coco.json')
        coco = json.load(open(annotation_path))
        labels = {}
        for annotation in coco['annotations']:
            x, y, w, h = annotation['bbox']
            labels.setdefault(annotation['image_id'], []).append([x, y, x + w, y + h])
        # Per prediction: its score and whether greedy score-ordered matching paired it with a label.
        scores, matched, detections = [], [], []
        images = coco['images']
        for start in range(0, len(images), args.batch):
            chunk = images[start:start + args.batch]
            pictures = [cv2.imread(os.path.join(args.dataset, split, i['file_name'])) for i in chunk]
            for image, (boxes, confidence) in zip(chunk, predict(pictures)):
                truth = np.array(labels.get(image['id'], []), dtype=float).reshape(-1, 4)
                free = np.ones(len(truth), dtype=bool)
                # Highest score first, so the match for any threshold is a prefix of this one.
                for k in np.argsort(-confidence):
                    scores.append(float(confidence[k]))
                    overlaps = np.where(free, iou(boxes[k], truth), 0) if len(truth) else np.zeros(0)
                    hit = len(overlaps) > 0 and overlaps.max() >= MATCH_IOU
                    if hit:
                        free[overlaps.argmax()] = False
                    matched.append(hit)
                    x1, y1, x2, y2 = boxes[k]
                    detections.append({'image_id': image['id'], 'category_id': 1, 'score': float(confidence[k]),
                                       'bbox': [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]})
        scores, matched = np.array(scores), np.array(matched)
        total = len(coco['annotations'])
        report[split] = {'coco': coco_ap(annotation_path, detections), 'thresholds': {}}
        ap = report[split]['coco']
        print(f"{split}: {len(images)} tiles, {total} labeled vehicles; AP {ap['AP']:.4f}  AP50 {ap['AP50']:.4f}  AP75 {ap['AP75']:.4f}")
        for threshold in args.thresholds:
            keep = scores >= threshold
            recall = float((matched & keep).sum() / total)
            precision = float((matched & keep).sum() / max(1, keep.sum()))
            report[split]['thresholds'][threshold] = {'recall': recall, 'precision': precision, 'detections': int(keep.sum())}
            print(f'  threshold {threshold:.2f}: recall {recall:.4f}  precision {precision:.4f}  detections {keep.sum()}', flush=True)

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, 'w') as f:
            json.dump(report, f, indent=2)


if __name__ == '__main__':
    main()
