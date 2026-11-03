"""Fine-tune RF-DETR-Large on the prepared B3D tiles (prep_data.py) at fixed scale."""
import argparse
import os

from rfdetr import RFDETRLarge
from rfdetr.datasets.aug_configs import AUG_AERIAL


def parse_args():
    parser = argparse.ArgumentParser(description='Fine-tune RF-DETR-Large on B3D tiles')
    parser.add_argument('--dataset', default='/out/dataset')
    parser.add_argument('--output', default='/out/runs/large')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', default='auto')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--resume', default=None)
    parser.add_argument('--weights', default=None)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lr-encoder', type=float, default=1.5e-4)
    parser.add_argument('--skip-best-epochs', type=int, default=1)
    parser.add_argument('--run', default='annotate-rfdetr-large')
    return parser.parse_args()


def main():
    args = parse_args()
    # RF-DETR's TrainConfig has no wandb entity field, so set the team through the environment.
    os.environ.setdefault('WANDB_ENTITY', 'chanwutk-berkeley')
    # Start from COCO weights, or from an earlier stage's checkpoint (fresh optimizer and schedule).
    model = RFDETRLarge(pretrain_weights=args.weights) if args.weights else RFDETRLarge()
    model.train(
        dataset_dir=args.dataset,
        output_dir=args.output,
        epochs=args.epochs,
        # Defaults are RF-DETR's; a later low-learning-rate stage passes 10x smaller values.
        lr=args.lr,
        lr_encoder=args.lr_encoder,
        batch_size=args.batch_size if args.batch_size == 'auto' else int(args.batch_size),
        num_workers=args.workers,
        # Tiles are already 704x704 at native resolution, so never rescale them.
        multi_scale=False,
        scale_jitter=False,
        # Flips, 90-degree rotations, and mild brightness/contrast are all label-preserving from overhead.
        aug_config=AUG_AERIAL,
        # Stop once validation mAP plateaus; by default the first epoch never counts as best.
        early_stopping=True,
        early_stopping_patience=5,
        skip_best_epochs=args.skip_best_epochs,
        # Score the held-out crop groups after training.
        run_test=True,
        seed=0,
        resume=args.resume,
        # Log to Weights & Biases (credentials from ~/.netrc, mounted by ./run).
        wandb=True,
        project='Tetris',
        run=args.run,
        # Log training loss every 50 steps instead of only the per-epoch average.
        train_log_on_step=True,
        # Log val/loss every epoch; "auto" skips it because early stopping monitors mAP, not loss.
        compute_val_loss=True,
    )


if __name__ == '__main__':
    main()
