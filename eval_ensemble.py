"""
Checkpoint ensemble + TTA evaluation for ResLSTMUNet.

Loads all epoch_*.pth checkpoints from a directory, runs ensemble+TTA
inference, and reports per-class Dice/Sensitivity/PPV/IoU plus mean
uncertainty per class.

Usage:
    python eval_ensemble.py \
        --ckpt_dir /content/drive/MyDrive/cardiac-project/output_aug/checkpoints/ResUNet_LSTM \
        --test_dir /content/drive/MyDrive/cardiac-project/data/converted/ct/test \
        --n_tta 8        # number of TTA transforms (1-8, default: all 8)
        --n_ckpts 10     # how many checkpoints to use (default: all found)
        --min_epoch 45   # only use checkpoints from this epoch onwards
"""

import argparse
import glob
import os
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import WHSDataset_2D_scale_partSeries
from models.reslstmunet import ResLSTMUNet
from models.src.utils import dice, sensitivity, PPV, cal_iou
from uncertainty import ensemble_predict, tta_predict, ensemble_tta_predict

NUM_CLASS = 8
CLASS_NAMES = [
    "background",
    "myocardium",
    "left atrium",
    "left ventricle",
    "right atrium",
    "right ventricle",
    "ascending aorta",
    "pulmonary artery",
]


def load_model(ckpt_path, device):
    model = ResLSTMUNet(in_channels=1, out_channels=NUM_CLASS,
                        pretrained=False, deep_sup=True, multiscale_att=True)
    state = torch.load(ckpt_path, map_location=device)
    if all(k.startswith("module.") for k in state.keys()):
        state = {k[len("module."):]: v for k, v in state.items()}
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def find_checkpoints(ckpt_dir, n_ckpts, min_epoch=None):
    """Return up to n_ckpts checkpoint paths at or after min_epoch, sorted by epoch (latest first)."""
    paths = sorted(
        glob.glob(os.path.join(ckpt_dir, "*epoch_*.pth")),
        key=lambda p: int(p.split("epoch_")[-1].replace(".pth", "")),
        reverse=True
    )
    if not paths:
        raise FileNotFoundError(f"No epoch_*.pth files found in {ckpt_dir}")
    if min_epoch is not None:
        paths = [p for p in paths if int(p.split("epoch_")[-1].replace(".pth", "")) >= min_epoch]
        if not paths:
            raise ValueError(f"No checkpoints found at or after epoch {min_epoch}")
    if n_ckpts:
        paths = paths[:n_ckpts]
    return paths


def evaluate_ensemble(models, loader, device, mode, n_tta):
    class_dice = np.zeros(NUM_CLASS)
    class_sens = np.zeros(NUM_CLASS)
    class_ppv  = np.zeros(NUM_CLASS)
    class_iou  = np.zeros(NUM_CLASS)
    class_unc  = np.zeros(NUM_CLASS)
    counts     = np.zeros(NUM_CLASS)

    with torch.no_grad():
        for image_serial, label_serial in loader:
            image_serial = [img.to(device) for img in image_serial]

            if mode == "ensemble":
                mean_probs, uncertainty = ensemble_predict(models, image_serial)
            elif mode == "tta":
                mean_probs, uncertainty = tta_predict(models[0], image_serial, n_transforms=n_tta)
            else:  # ensemble+tta
                mean_probs, uncertainty = ensemble_tta_predict(
                    models, image_serial, n_transforms=n_tta
                )

            for t in range(len(mean_probs)):
                pred = mean_probs[t].argmax(dim=1).cpu().numpy()   # (B, H, W)
                unc  = uncertainty[t].cpu().numpy()                 # (B, H, W)
                lbl  = label_serial[t].squeeze(1).long().cpu().numpy()

                pred_flat = pred.flatten()
                lbl_flat  = lbl.flatten()

                for c in range(NUM_CLASS):
                    p = (pred_flat == c).astype(int)
                    g = (lbl_flat  == c).astype(int)
                    class_dice[c] += dice(p, g)
                    class_sens[c] += sensitivity(p, g)
                    class_ppv[c]  += PPV(p, g)
                    class_iou[c]  += cal_iou(p, g)
                    # mean uncertainty within predicted foreground mask
                    mask = pred == c
                    class_unc[c] += unc[mask].mean() if mask.any() else 0.0
                    counts[c]    += 1

    class_dice /= counts
    class_sens /= counts
    class_ppv  /= counts
    class_iou  /= counts
    class_unc  /= counts
    return class_dice, class_sens, class_ppv, class_iou, class_unc


def main():
    parser = argparse.ArgumentParser(
        description="Checkpoint ensemble + TTA evaluation")
    parser.add_argument("--ckpt_dir",    required=False, default=None,
                        help="Directory containing epoch_*.pth checkpoints "
                             "(required for ensemble and ensemble+tta modes)")
    parser.add_argument("--ckpt",        required=False, default=None,
                        help="Single checkpoint path (used for --mode tta)")
    parser.add_argument("--test_dir",    required=True,
                        help="Path to converted test directory")
    parser.add_argument("--crop_d",      type=int, default=18)
    parser.add_argument("--batch_size",  type=int, default=2,
                        help="Keep low — ensemble+TTA uses more VRAM")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--n_tta",       type=int, default=None,
                        help="Number of TTA transforms (default: all 8)")
    parser.add_argument("--n_ckpts",     type=int, default=None,
                        help="Max checkpoints to load (default: all found)")
    parser.add_argument("--min_epoch",   type=int, default=None,
                        help="Only use checkpoints from this epoch onwards")
    parser.add_argument("--mode",        default="ensemble+tta",
                        choices=["ensemble", "tta", "ensemble+tta"],
                        help="ensemble=checkpoints only, tta=single ckpt+TTA, "
                             "ensemble+tta=both (default)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps"  if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.mode == "tta":
        if args.ckpt is None and args.ckpt_dir is None:
            raise ValueError("--mode tta requires either --ckpt or --ckpt_dir")
        ckpt_path = args.ckpt if args.ckpt else \
            find_checkpoints(args.ckpt_dir, args.n_ckpts, args.min_epoch)[0]
        models = [load_model(ckpt_path, device)]
        print(f"Mode: TTA only (using {os.path.basename(ckpt_path)})")
    else:
        if args.ckpt_dir is None:
            raise ValueError(f"--mode {args.mode} requires --ckpt_dir")
        ckpt_paths = find_checkpoints(args.ckpt_dir, args.n_ckpts, args.min_epoch)
        print(f"Loading {len(ckpt_paths)} checkpoints:")
        for p in ckpt_paths:
            print(f"  {os.path.basename(p)}")
        models = [load_model(p, device) for p in ckpt_paths]

    n_tta_display = args.n_tta if args.n_tta else 8
    if args.mode == "ensemble":
        print(f"\nMode: Ensemble only  (total passes per batch: {len(models)})\n")
    elif args.mode == "tta":
        print(f"\nMode: TTA only  (transforms: {n_tta_display}, "
              f"total passes per batch: {n_tta_display + 1})\n")
    else:
        print(f"\nMode: Ensemble + TTA  (transforms: {n_tta_display}, "
              f"total passes per batch: {len(models) * (n_tta_display + 1)})\n")

    dataset = WHSDataset_2D_scale_partSeries([args.test_dir], crop_d=args.crop_d)
    loader  = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                         num_workers=args.num_workers, pin_memory=True)
    print(f"Test sequences: {len(dataset)}\n")

    class_dice, class_sens, class_ppv, class_iou, class_unc = \
        evaluate_ensemble(models, loader, device, args.mode, args.n_tta)

    col = 25
    print(f"{'Class':<{col}} {'Dice':>8} {'Sensitivity':>12} {'PPV':>8} "
          f"{'IoU':>8} {'Uncertainty':>12}")
    print("-" * (col + 52))
    for c in range(NUM_CLASS):
        print(f"{CLASS_NAMES[c]:<{col}} {class_dice[c]:>8.4f} "
              f"{class_sens[c]:>12.4f} {class_ppv[c]:>8.4f} "
              f"{class_iou[c]:>8.4f} {class_unc[c]:>12.6f}")
    print("-" * (col + 52))
    print(f"{'Mean (classes 1-7)':<{col}} {class_dice[1:].mean():>8.4f} "
          f"{class_sens[1:].mean():>12.4f} {class_ppv[1:].mean():>8.4f} "
          f"{class_iou[1:].mean():>8.4f} {class_unc[1:].mean():>12.6f}")


if __name__ == "__main__":
    main()
