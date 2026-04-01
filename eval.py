import argparse
import time
import os
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import WHSDataset_2D_scale_partSeries
from models.reslstmunet import ResLSTMUNet
from models.src.utils import dice, sensitivity, PPV, cal_iou

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
    # Strip DataParallel 'module.' prefix if present
    if all(k.startswith("module.") for k in state.keys()):
        state = {k[len("module."):]: v for k, v in state.items()}
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def evaluate(model, loader, device):
    class_dice = np.zeros(NUM_CLASS)
    class_sens = np.zeros(NUM_CLASS)
    class_ppv  = np.zeros(NUM_CLASS)
    class_iou  = np.zeros(NUM_CLASS)
    counts     = np.zeros(NUM_CLASS)

    with torch.no_grad():
        for image_serial, label_serial in loader:
            image_serial = [img.to(device) for img in image_serial]

            # model returns (pred_serial, pred1_serial, ..., pred4_serial)
            # pred_serial is a list of T tensors, each (B, NUM_CLASS, H, W)
            pred_serial, *_ = model(image_serial)

            for t in range(len(pred_serial)):
                pred = pred_serial[t].argmax(dim=1).cpu().numpy().flatten()
                lbl  = label_serial[t].squeeze(1).long().cpu().numpy().flatten()

                for c in range(NUM_CLASS):
                    p = (pred == c).astype(int)
                    g = (lbl  == c).astype(int)
                    class_dice[c] += dice(p, g)
                    class_sens[c] += sensitivity(p, g)
                    class_ppv[c]  += PPV(p, g)
                    class_iou[c]  += cal_iou(p, g)
                    counts[c]     += 1

    class_dice /= counts
    class_sens /= counts
    class_ppv  /= counts
    class_iou  /= counts
    return class_dice, class_sens, class_ppv, class_iou


def main():
    parser = argparse.ArgumentParser(description="Evaluate ResLSTMUNet on cardiac test data")
    parser.add_argument("--ckpt",     required=True, help="Path to model checkpoint (.pth)")
    parser.add_argument("--test_dir", required=True, help="Path to converted test directory")
    parser.add_argument("--crop_d",   type=int, default=18, help="Sequence length (default: 18)")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = WHSDataset_2D_scale_partSeries([args.test_dir], crop_d=args.crop_d)
    loader  = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                         num_workers=args.num_workers, pin_memory=True)
    print(f"Test sequences: {len(dataset)}")

    model = load_model(args.ckpt, device)
    print(f"Loaded checkpoint: {args.ckpt}\n")

    class_dice, class_sens, class_ppv, class_iou = evaluate(model, loader, device)

    col = 25
    print(f"{'Class':<{col}} {'Dice':>8} {'Sensitivity':>12} {'PPV':>8} {'IoU':>8}")
    print("-" * (col + 40))
    for c in range(NUM_CLASS):
        print(f"{CLASS_NAMES[c]:<{col}} {class_dice[c]:>8.4f} {class_sens[c]:>12.4f} "
              f"{class_ppv[c]:>8.4f} {class_iou[c]:>8.4f}")
    print("-" * (col + 40))
    print(f"{'Mean (classes 1-7)':<{col}} {class_dice[1:].mean():>8.4f} "
          f"{class_sens[1:].mean():>12.4f} {class_ppv[1:].mean():>8.4f} "
          f"{class_iou[1:].mean():>8.4f}")


if __name__ == "__main__":
    main()
