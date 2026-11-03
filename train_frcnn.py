"""Fine-tune detectron2 Faster R-CNN X101-FPN on B3D crops (prep_crops.py), following B3D's train.py.

modules/b3d/train.py registers the COCO crops, loads a model-zoo config with its COCO checkpoint,
sets one class, and trains with detectron2's DefaultTrainer. This keeps that and changes what the
comparison with RF-DETR needs: native scale (no resizing), RF-DETR's label-preserving aerial
augmentations, detectron2's standard fine-tuning schedule, validation every epoch, and wandb.
Runs in tetris-annotate-d2 (B3D's Docker image plus wandb), which is Python 3.7.
"""
import argparse
import json
import math
import os

import detectron2.data.transforms as T
import detectron2.utils.comm as comm
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import DatasetMapper, build_detection_train_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer, launch
from detectron2.evaluation import COCOEvaluator
from detectron2.utils.logger import setup_logger


# Model-zoo config; its checkpoint is .../139173657/model_final_68b088.pkl (COCO box AP 43.0).
CONFIG = 'COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml'


def parse_args():
    parser = argparse.ArgumentParser(description='Fine-tune Faster R-CNN X101-FPN on B3D crops')
    parser.add_argument('--dataset', default='/out/dataset-crops-groups')
    parser.add_argument('--output', default='/out/runs/frcnn-groups')
    parser.add_argument('--run', default='annotate-frcnn-x101-groups')
    parser.add_argument('--epochs', type=int, default=6)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--gpus', type=int, default=2)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--resume', action='store_true')
    return parser.parse_args()


class Trainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        # Flips, 90-degree rotations, and mild brightness/contrast, as RF-DETR's AUG_AERIAL; no resizing.
        augmentations = [
            T.RandomFlip(horizontal=True, vertical=False),
            T.RandomFlip(horizontal=False, vertical=True),
            T.RandomRotation([0, 90, 180, 270], expand=True, sample_style='choice'),
            T.RandomBrightness(0.85, 1.15),
            T.RandomContrast(0.85, 1.15),
        ]
        return build_detection_train_loader(cfg, mapper=DatasetMapper(cfg, is_train=True, augmentations=augmentations))

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        # Dense crops hold more than COCO's default 100 vehicles.
        return COCOEvaluator(dataset_name, output_dir=os.path.join(cfg.OUTPUT_DIR, 'eval'),
                             max_dets_per_image=cfg.TEST.DETECTIONS_PER_IMAGE)


def build_cfg(args):
    # Register the prepared splits, as B3D's train.py registers its train.json.
    for split in ['train', 'valid']:
        root = os.path.join(args.dataset, split)
        register_coco_instances(f'b3d_{split}', {}, os.path.join(root, '_annotations.coco.json'), root)
    train_images = len(json.load(open(os.path.join(args.dataset, 'train', '_annotations.coco.json')))['images'])
    iters_per_epoch = math.ceil(train_images / args.batch)

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(CONFIG))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(CONFIG)
    cfg.DATASETS.TRAIN = ('b3d_train',)
    cfg.DATASETS.TEST = ('b3d_valid',)
    cfg.DATALOADER.NUM_WORKERS = args.workers
    # One 'vehicle' class and B3D's ROI sampling, as in modules/b3d/configs/config_refined.json.
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    # One anchor size per FPN level, shifted down one octave for 10-500 px vehicles. B3D's
    # six-size list was for RetinaNet; on this RPN it would discard the pretrained RPN head.
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[16], [32], [64], [128], [256]]
    # Native scale: size 0 makes ResizeShortestEdge a no-op; batches are padded internally.
    cfg.INPUT.MIN_SIZE_TRAIN = (0,)
    cfg.INPUT.MIN_SIZE_TEST = 0
    cfg.TEST.DETECTIONS_PER_IMAGE = 500
    # detectron2's fine-tuning schedule: linear-scaled SGD, 10x drops at 70% and 90%.
    cfg.SOLVER.IMS_PER_BATCH = args.batch
    cfg.SOLVER.BASE_LR = args.lr
    cfg.SOLVER.MAX_ITER = args.epochs * iters_per_epoch
    cfg.SOLVER.STEPS = (int(0.7 * cfg.SOLVER.MAX_ITER), int(0.9 * cfg.SOLVER.MAX_ITER))
    cfg.SOLVER.CHECKPOINT_PERIOD = iters_per_epoch
    cfg.TEST.EVAL_PERIOD = iters_per_epoch
    cfg.OUTPUT_DIR = args.output
    cfg.freeze()
    return cfg


def main(args):
    cfg = build_cfg(args)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank())
    if comm.is_main_process():
        with open(os.path.join(cfg.OUTPUT_DIR, 'config.yaml'), 'w') as f:
            f.write(cfg.dump())
        # wandb mirrors detectron2's TensorBoard events; it must start before the trainer opens them.
        # This image's wandb 0.18 defaults to api.wandb.ai; the login (~/.netrc) is for W&B's new host.
        os.environ.setdefault('WANDB_BASE_URL', 'https://api.forge.coreweave.com')
        import wandb
        wandb.init(entity='chanwutk-berkeley', project='Tetris', name=args.run, config=vars(args),
                   sync_tensorboard=True, resume='allow' if args.resume else None)
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    trainer.train()


if __name__ == '__main__':
    args = parse_args()
    # On ace, NCCL peer-to-peer hangs between GPUs behind different PCIe host bridges
    # (nvidia-smi topo "NODE"); keep P2P within a PCIe switch pair and use shared memory otherwise.
    os.environ.setdefault('NCCL_P2P_LEVEL', 'PIX')
    launch(main, args.gpus, dist_url='auto', args=(args,))
