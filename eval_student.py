"""
eval_student.py

Single-pass evaluation of the distilled student model.

Reports per-class Dice / Sensitivity / PPV / IoU and mean predicted
uncertainty per class, matching the format of eval_ensemble.py so results
can be compared directly.

Usage:
    python eval_student.py \\
        --ckpt /path/to/distill_student_best_epoch_N.pth \\
        --test_dir /path/to/ct/test \\
        [--save_unc_dir /path/to/output/uncertainty]
"""

import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import WHSDataset_2D_scale_partSeries
from models.reslstmunet import ResLSTMUNet
from models.src.utils import PPV, cal_iou, dice, sensitivity

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


def load_student(ckpt_path: str, device: torch.device) -> ResLSTMUNet:
    model = ResLSTMUNet(
        in_channels=1, out_channels=NUM_CLASS,
        pretrained=False,
        deep_sup=True,
        multiscale_att=True,
        predict_uncertainty=True,
    )
    state = torch.load(ckpt_path, map_location=device)
    if all(k.startswith("module.") for k in state.keys()):
        state = {k[len("module."):]: v for k, v in state.items()}
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def evaluate_student(model: ResLSTMUNet, loader: DataLoader,
                     device: torch.device,
                     save_unc_dir: str = None):
    """
    Returns:
        (class_dice, class_sens, class_ppv, class_iou, class_unc)
        Each is a numpy array of length NUM_CLASS.

    class_unc[c] = mean student uncertainty within the predicted foreground
                   mask for class c (averaged over all slices and patients).
    """
    if save_unc_dir is not None:
        os.makedirs(save_unc_dir, exist_ok=True)

    class_dice = np.zeros(NUM_CLASS)
    class_sens = np.zeros(NUM_CLASS)
    class_ppv  = np.zeros(NUM_CLASS)
    class_iou  = np.zeros(NUM_CLASS)
    class_unc  = np.zeros(NUM_CLASS)
    counts     = np.zeros(NUM_CLASS)

    batch_idx = 0
    with torch.no_grad():
        for image_serial, label_serial in loader:
            image_serial = [img.to(device) for img in image_serial]

            # unc_serial is the last element of the return tuple
            pred_serial, *_, unc_serial = model(image_serial)

            for t in range(len(pred_serial)):
                pred = pred_serial[t].argmax(dim=1).cpu().numpy()   # (B, H, W)
                unc  = unc_serial[t].squeeze(1).cpu().numpy()       # (B, H, W)
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
                    mask = pred == c
                    class_unc[c] += unc[mask].mean() if mask.any() else 0.0
                    counts[c]    += 1

                if save_unc_dir is not None:
                    np.save(
                        os.path.join(save_unc_dir,
                                     f"unc_b{batch_idx:04d}_t{t:02d}.npy"),
                        unc_serial[t].cpu().numpy(),   # (B, 1, H, W)
                    )

            batch_idx += 1

    class_dice /= counts
    class_sens /= counts
    class_ppv  /= counts
    class_iou  /= counts
    class_unc  /= counts
    return class_dice, class_sens, class_ppv, class_iou, class_unc


def print_results(class_dice, class_sens, class_ppv, class_iou, class_unc):
    header = f"{'Class':<22}  {'Dice':>6}  {'Sens':>6}  {'PPV':>6}  {'IoU':>6}  {'Unc':>8}"
    print(header)
    print("-" * len(header))
    for c in range(1, NUM_CLASS):   # skip background
        print(f"{CLASS_NAMES[c]:<22}  "
              f"{class_dice[c]:6.4f}  "
              f"{class_sens[c]:6.4f}  "
              f"{class_ppv[c]:6.4f}  "
              f"{class_iou[c]:6.4f}  "
              f"{class_unc[c]:8.5f}")
    print("-" * len(header))
    fg = slice(1, NUM_CLASS)
    print(f"{'Mean (fg)':<22}  "
          f"{class_dice[fg].mean():6.4f}  "
          f"{class_sens[fg].mean():6.4f}  "
          f"{class_ppv[fg].mean():6.4f}  "
          f"{class_iou[fg].mean():6.4f}  "
          f"{class_unc[fg].mean():8.5f}")


def main():
    parser = argparse.ArgumentParser(
        description="Single-pass student evaluation.")
    parser.add_argument("--ckpt", required=True,
                        help="Path to distilled student checkpoint (.pth).")
    parser.add_argument("--test_dir", required=True,
                        help="Path to test data directory.")
    parser.add_argument("--crop_d", type=int, default=18)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_unc_dir", default=None,
                        help="If set, save uncertainty maps as .npy files here.")
    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")
    print(f"Checkpoint:   {args.ckpt}\n")

    model = load_student(args.ckpt, device)

    dataset = WHSDataset_2D_scale_partSeries(
        [args.test_dir], crop_d=args.crop_d, augment=False)
    loader  = DataLoader(
        dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers, pin_memory=True)

    class_dice, class_sens, class_ppv, class_iou, class_unc = evaluate_student(
        model, loader, device, save_unc_dir=args.save_unc_dir)

    print_results(class_dice, class_sens, class_ppv, class_iou, class_unc)

    if args.save_unc_dir:
        print(f"\nUncertainty maps saved to: {args.save_unc_dir}")


if __name__ == "__main__":
    main()
