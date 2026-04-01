"""
Spearman correlation: per-sequence TTA uncertainty vs Dice.

For each 18-slice sequence in the test set, computes:
  - Per-class Dice       (over all 18 slices of that sequence)
  - Per-class TTA uncertainty  (mean entropy within predicted class mask,
                                over all 18 slices)

Reports Spearman ρ and p-value per foreground class across all sequences,
saves scatter plots and a per-sequence CSV.

Usage (CT):
    python eval_spearman.py \
        --ckpt     /content/drive/MyDrive/cardiac-project/output_aug_ensemble_ct/statistics/ResUNet_LSTM/ResUNet_LSTM_best_epoch_56.pth \
        --test_dir /content/drive/MyDrive/cardiac-project/data/converted/ct/test \
        --out_dir  /content/drive/MyDrive/cardiac-project/calibration \
        --modality ct

Usage (MR):
    python eval_spearman.py \
        --ckpt     /content/drive/MyDrive/cardiac-project/output_aug_ensemble_mr/statistics/ResUNet_LSTM/ResUNet_LSTM_best_epoch_58.pth \
        --test_dir /content/drive/MyDrive/cardiac-project/data/converted/mr/test \
        --out_dir  /content/drive/MyDrive/cardiac-project/calibration \
        --modality mr
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import stats
from torch.utils.data import DataLoader

from dataset import WHSDataset_2D_scale_partSeries
from models.reslstmunet import ResLSTMUNet
from models.src.utils import dice
from uncertainty import tta_predict

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
FOREGROUND = list(range(1, NUM_CLASS))   # skip background for correlation


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Per-sequence inference
# ---------------------------------------------------------------------------

def run_inference(model, loader, device):
    """
    Run TTA inference over all sequences in the loader.

    Each loader item is one sequence (batch_size=1, T=18 slices).
    Returns two arrays of shape (N_sequences, NUM_CLASS):
        dice_matrix : per-class Dice for each sequence
        unc_matrix  : per-class mean TTA uncertainty for each sequence
    """
    dice_records = []
    unc_records  = []

    with torch.no_grad():
        for image_serial, label_serial in loader:
            image_serial = [img.to(device) for img in image_serial]

            mean_probs, uncertainty = tta_predict(model, image_serial)

            # Concatenate all T slices in the sequence
            # pred  : (T*B, H, W)   B=1 here
            # unc   : (T*B, H, W)
            # label : (T*B, H, W)
            pred_all  = torch.cat(
                [mp.argmax(dim=1) for mp in mean_probs], dim=0
            ).cpu().numpy()
            unc_all   = torch.cat(uncertainty, dim=0).cpu().numpy()
            label_all = torch.cat(
                [lb.squeeze(1).long() for lb in label_serial], dim=0
            ).cpu().numpy()

            pred_flat  = pred_all.flatten()
            label_flat = label_all.flatten()

            seq_dice = np.zeros(NUM_CLASS)
            seq_unc  = np.zeros(NUM_CLASS)

            for c in range(NUM_CLASS):
                p = (pred_flat  == c).astype(int)
                g = (label_flat == c).astype(int)
                seq_dice[c] = dice(p, g)

                mask = pred_all == c
                seq_unc[c]  = unc_all[mask].mean() if mask.any() else 0.0

            dice_records.append(seq_dice)
            unc_records.append(seq_unc)

    return np.array(dice_records), np.array(unc_records)


# ---------------------------------------------------------------------------
# Scatter plot
# ---------------------------------------------------------------------------

def plot_scatter(unc_matrix, dice_matrix, rho_arr, pval_arr, modality, out_path):
    n_fg = len(FOREGROUND)
    fig, axes = plt.subplots(1, n_fg, figsize=(3.5 * n_fg, 3.5))
    fig.suptitle(
        f"{modality.upper()} — TTA uncertainty vs Dice (per sequence, n={len(unc_matrix)})",
        fontsize=12
    )

    for i, c in enumerate(FOREGROUND):
        ax   = axes[i]
        unc  = unc_matrix[:, c]
        dc   = dice_matrix[:, c]
        rho  = rho_arr[i]
        pval = pval_arr[i]

        ax.scatter(unc, dc, s=20, alpha=0.5, color="#1f77b4")

        if len(unc) > 2:
            m, b = np.polyfit(unc, dc, 1)
            x_line = np.linspace(unc.min(), unc.max(), 50)
            ax.plot(x_line, m * x_line + b, color="#d62728",
                    linewidth=1.2, linestyle="--")

        sig = ("***" if pval < 0.001 else
               "**"  if pval < 0.01  else
               "*"   if pval < 0.05  else "ns")
        ax.set_title(f"{CLASS_NAMES[c]}\nρ={rho:.3f}  p={pval:.3f} {sig}",
                     fontsize=8)
        ax.set_xlabel("TTA uncertainty", fontsize=7)
        ax.set_ylabel("Dice",            fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved scatter plot → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Spearman correlation: TTA uncertainty vs Dice per sequence")
    parser.add_argument("--ckpt",        required=True,
                        help="Single best-epoch checkpoint path")
    parser.add_argument("--test_dir",    required=True,
                        help="Test data directory")
    parser.add_argument("--out_dir",     required=True,
                        help="Output directory for plot and CSV")
    parser.add_argument("--modality",    default="ct", choices=["ct", "mr"])
    parser.add_argument("--crop_d",      type=int, default=18)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps"  if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Loading checkpoint: {os.path.basename(args.ckpt)}")
    model = load_model(args.ckpt, device)

    # batch_size=1 so each loader item = exactly one 18-slice sequence
    dataset = WHSDataset_2D_scale_partSeries([args.test_dir], crop_d=args.crop_d)
    loader  = DataLoader(dataset, batch_size=1, shuffle=False,
                         num_workers=args.num_workers, pin_memory=True)
    print(f"Test sequences: {len(dataset)}\n")

    print("Running TTA inference (this may take a while) ...")
    dice_matrix, unc_matrix = run_inference(model, loader, device)
    # shapes: (N_sequences, NUM_CLASS)

    # Spearman correlation per foreground class
    rho_arr  = np.zeros(len(FOREGROUND))
    pval_arr = np.zeros(len(FOREGROUND))

    col = 22
    print(f"\n{'Class':<{col}} {'ρ':>8} {'p-value':>10} {'Sig':>5}")
    print("-" * (col + 26))
    for i, c in enumerate(FOREGROUND):
        rho, pval = stats.spearmanr(unc_matrix[:, c], dice_matrix[:, c])
        rho_arr[i]  = rho
        pval_arr[i] = pval
        sig = ("***" if pval < 0.001 else
               "**"  if pval < 0.01  else
               "*"   if pval < 0.05  else "ns")
        print(f"{CLASS_NAMES[c]:<{col}} {rho:>8.4f} {pval:>10.4f} {sig:>5}")

    # Scatter plot
    scatter_path = os.path.join(args.out_dir, f"spearman_{args.modality}.png")
    plot_scatter(unc_matrix, dice_matrix, rho_arr, pval_arr,
                 args.modality, scatter_path)

    # Per-sequence CSV
    rows = []
    for j in range(len(dice_matrix)):
        row = {"sequence_idx": j}
        for c in range(NUM_CLASS):
            cname = CLASS_NAMES[c].replace(" ", "_")
            row[f"dice_{cname}"] = dice_matrix[j, c]
            row[f"unc_{cname}"]  = unc_matrix[j, c]
        rows.append(row)

    csv_path = os.path.join(args.out_dir, f"spearman_{args.modality}.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"\nSaved per-sequence data → {csv_path}")

    print(f"\nMean |ρ| across foreground classes : {np.abs(rho_arr).mean():.4f}")
    strongest_i = int(np.argmin(rho_arr))
    print(f"Strongest negative ρ               : "
          f"{CLASS_NAMES[FOREGROUND[strongest_i]]}  "
          f"ρ={rho_arr[strongest_i]:.4f}")


if __name__ == "__main__":
    main()
