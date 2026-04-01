"""
Step 3: Evaluate calibration on the test set.

Computes ECE (Expected Calibration Error) for three configurations:
  1. Single best checkpoint, T=1     (uncalibrated baseline)
  2. Checkpoint ensemble,   T=1     (uncalibrated ensemble)
  3. Checkpoint ensemble,   T=T*    (calibrated, T* from calibrate_temperature.py)

Outputs:
  - Terminal: 3-row ECE comparison table (global + per class)
  - <out_dir>/reliability_<modality>.png   reliability diagrams (ensemble uncal vs cal)
  - <out_dir>/ece_<modality>.csv           all ECE values

Usage (CT):
    python eval_calibration.py \
        --ckpt_dir    /content/drive/MyDrive/cardiac-project/output_aug_ensemble_ct/checkpoints/ResUNet_LSTM \
        --single_ckpt /content/drive/MyDrive/cardiac-project/output_aug_ensemble_ct/statistics/ResUNet_LSTM/ResUNet_LSTM_best_epoch_56.pth \
        --test_dir    /content/drive/MyDrive/cardiac-project/data/converted/ct/test \
        --temperature_json /content/drive/MyDrive/cardiac-project/calibration/ct_temperature.json \
        --out_dir     /content/drive/MyDrive/cardiac-project/calibration \
        --modality    ct  --n_ckpts 10

Usage (MR):
    python eval_calibration.py \
        --ckpt_dir    /content/drive/MyDrive/cardiac-project/output_aug_ensemble_mr/checkpoints/ResUNet_LSTM \
        --single_ckpt /content/drive/MyDrive/cardiac-project/output_aug_ensemble_mr/statistics/ResUNet_LSTM/ResUNet_LSTM_best_epoch_58.pth \
        --test_dir    /content/drive/MyDrive/cardiac-project/data/converted/mr/test \
        --temperature_json /content/drive/MyDrive/cardiac-project/calibration/mr_temperature.json \
        --out_dir     /content/drive/MyDrive/cardiac-project/calibration \
        --modality    mr  --n_ckpts 10
"""

import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import WHSDataset_2D_scale_partSeries
from models.reslstmunet import ResLSTMUNet

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


# ---------------------------------------------------------------------------
# Model helpers
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


def find_checkpoints(ckpt_dir, n_ckpts, min_epoch=None):
    paths = sorted(
        glob.glob(os.path.join(ckpt_dir, "*epoch_*.pth")),
        key=lambda p: int(p.split("epoch_")[-1].replace(".pth", "")),
        reverse=True,
    )
    if not paths:
        raise FileNotFoundError(f"No epoch_*.pth files found in {ckpt_dir}")
    if min_epoch is not None:
        paths = [p for p in paths
                 if int(p.split("epoch_")[-1].replace(".pth", "")) >= min_epoch]
    if n_ckpts:
        paths = paths[:n_ckpts]
    return paths


# ---------------------------------------------------------------------------
# Logit collection (same approach as calibrate_temperature.py)
# ---------------------------------------------------------------------------

def collect_logits(models, loader, device, n_samples_per_batch, rng):
    """
    Average raw logits across models (no softmax), subsample voxels per batch.

    Returns:
        logits : (N, NUM_CLASS) float32 CPU tensor — mean logits across models
        labels : (N,)           int64   CPU tensor
    """
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for image_serial, label_serial in loader:
            image_serial = [img.to(device) for img in image_serial]

            logit_sum = None
            for model in models:
                pred_serial, *_ = model(image_serial)
                logits_t = torch.stack(pred_serial, dim=0)   # (T, B, C, H, W)
                logit_sum = logits_t if logit_sum is None else logit_sum + logits_t
            mean_logits = logit_sum / len(models)             # (T, B, C, H, W)

            labels = torch.stack(
                [lb.squeeze(1).long() for lb in label_serial], dim=0
            )                                                 # (T, B, H, W)

            T, B, C, H, W = mean_logits.shape
            flat_logits = mean_logits.permute(0, 1, 3, 4, 2).reshape(-1, C).cpu()
            flat_labels = labels.reshape(-1).cpu()

            n = flat_logits.shape[0]
            idx = torch.from_numpy(
                rng.choice(n, size=min(n_samples_per_batch, n), replace=False)
            )
            all_logits.append(flat_logits[idx])
            all_labels.append(flat_labels[idx])

    return torch.cat(all_logits, dim=0), torch.cat(all_labels, dim=0)


# ---------------------------------------------------------------------------
# ECE computation
# ---------------------------------------------------------------------------

def compute_ece_global(probs_np, labels_np, n_bins=15):
    """
    Top-label ECE: confidence = max(p), accuracy = argmax(p) == label.
    Returns scalar ECE.
    """
    confidences = probs_np.max(axis=1)
    predictions = probs_np.argmax(axis=1)
    accuracies  = (predictions == labels_np).astype(float)

    bins = np.linspace(0, 1, n_bins + 1)
    ece  = 0.0
    N    = len(confidences)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (confidences >= lo) & (confidences <= hi if i == n_bins - 1 else confidences < hi)
        if mask.sum() > 0:
            ece += (mask.sum() / N) * abs(accuracies[mask].mean() - confidences[mask].mean())
    return ece


def compute_ece_classwise(probs_np, labels_np, n_bins=15):
    """
    Per-class ECE (CW-ECE, Nixon et al. 2019).
    For class k: confidence = p_k, positive = (label == k).

    Returns:
        ece_per_class : (NUM_CLASS,) array
        bin_stats     : list of NUM_CLASS tuples (bin_conf, bin_acc), each length n_bins
                        NaN for empty bins — used for reliability diagrams.
    """
    bins = np.linspace(0, 1, n_bins + 1)
    N    = probs_np.shape[0]

    ece_per_class = np.zeros(NUM_CLASS)
    bin_stats     = []

    for k in range(NUM_CLASS):
        conf_k = probs_np[:, k]
        pos_k  = (labels_np == k).astype(float)

        bin_conf_arr = np.full(n_bins, np.nan)
        bin_acc_arr  = np.full(n_bins, np.nan)
        ece_k = 0.0

        for i in range(n_bins):
            lo, hi = bins[i], bins[i + 1]
            mask = (conf_k >= lo) & (conf_k <= hi if i == n_bins - 1 else conf_k < hi)
            if mask.sum() > 0:
                bin_conf_arr[i] = conf_k[mask].mean()
                bin_acc_arr[i]  = pos_k[mask].mean()
                ece_k += (mask.sum() / N) * abs(bin_acc_arr[i] - bin_conf_arr[i])

        ece_per_class[k] = ece_k
        bin_stats.append((bin_conf_arr, bin_acc_arr))

    return ece_per_class, bin_stats


# ---------------------------------------------------------------------------
# Reliability diagram
# ---------------------------------------------------------------------------

def plot_reliability_diagrams(bin_stats_uncal, ece_uncal,
                               bin_stats_cal,   ece_cal,
                               modality, out_path, n_bins=15):
    """
    2-row × 8-col figure.
    Row 0: ensemble uncalibrated, Row 1: ensemble calibrated.
    Each cell shows the reliability curve for one class.
    """
    fig, axes = plt.subplots(2, NUM_CLASS, figsize=(20, 5), sharey=False)
    fig.suptitle(f"{modality.upper()} — Reliability diagrams: ensemble (uncalibrated vs calibrated)",
                 fontsize=12, y=1.01)

    bin_centers = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_centers[:-1] + bin_centers[1:]) / 2

    for row, (bin_stats, ece_arr, label) in enumerate([
        (bin_stats_uncal, ece_uncal, "Uncal."),
        (bin_stats_cal,   ece_cal,   f"Cal. T*"),
    ]):
        for k in range(NUM_CLASS):
            ax = axes[row, k]
            conf, acc = bin_stats[k]

            # Gap bars (miscalibration)
            gap = acc - conf
            colors = ["#d62728" if g < 0 else "#1f77b4" for g in
                      np.where(np.isnan(gap), 0, gap)]
            ax.bar(bin_centers, np.where(np.isnan(acc), 0, acc),
                   width=1 / n_bins, color=colors, alpha=0.6, align="center")

            # Perfect calibration diagonal
            ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.5)

            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect("equal")

            ece_val = ece_arr[k]
            title   = CLASS_NAMES[k].replace(" ", "\n")
            ax.set_title(f"{title}\nECE={ece_val:.4f}", fontsize=7)

            if k == 0:
                ax.set_ylabel(label, fontsize=8)
            ax.tick_params(labelsize=6)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved reliability diagram → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Calibration evaluation on test set")
    parser.add_argument("--ckpt_dir",          required=True,
                        help="Directory with epoch_*.pth ensemble checkpoints")
    parser.add_argument("--single_ckpt",       required=True,
                        help="Path to single best-epoch checkpoint (row 1 of table)")
    parser.add_argument("--test_dir",          required=True,
                        help="Test data directory")
    parser.add_argument("--temperature_json",  required=True,
                        help="JSON produced by calibrate_temperature.py")
    parser.add_argument("--out_dir",           required=True,
                        help="Directory for output PNG and CSV")
    parser.add_argument("--modality",          default="ct", choices=["ct", "mr"])
    parser.add_argument("--n_ckpts",           type=int, default=10)
    parser.add_argument("--min_epoch",         type=int, default=None)
    parser.add_argument("--crop_d",            type=int, default=18)
    parser.add_argument("--batch_size",        type=int, default=4)
    parser.add_argument("--num_workers",       type=int, default=4)
    parser.add_argument("--n_samples_per_batch", type=int, default=5000,
                        help="Voxels randomly sampled per batch for ECE")
    parser.add_argument("--n_bins",            type=int, default=15)
    parser.add_argument("--seed",              type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps"  if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    # Load temperature
    with open(args.temperature_json) as f:
        cal_json = json.load(f)
    T_star = cal_json["temperature"]
    print(f"Loaded temperature T* = {T_star:.4f}  ({args.temperature_json})")

    # Build loaders
    dataset = WHSDataset_2D_scale_partSeries([args.test_dir], crop_d=args.crop_d)
    loader  = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                         num_workers=args.num_workers, pin_memory=True)
    print(f"Test sequences: {len(dataset)}\n")

    rng = np.random.default_rng(args.seed)

    # ------------------------------------------------------------------
    # Run 1: Single model
    # ------------------------------------------------------------------
    print("--- Run 1/2: Single model ---")
    single_model = [load_model(args.single_ckpt, device)]
    rng_single   = np.random.default_rng(args.seed)
    logits_single, labels_single = collect_logits(
        single_model, loader, device, args.n_samples_per_batch, rng_single
    )
    print(f"  Collected {logits_single.shape[0]:,} voxels\n")

    # ------------------------------------------------------------------
    # Run 2: Ensemble
    # ------------------------------------------------------------------
    print("--- Run 2/2: Ensemble ---")
    ckpt_paths = find_checkpoints(args.ckpt_dir, args.n_ckpts, args.min_epoch)
    print(f"  Loading {len(ckpt_paths)} checkpoints:")
    for p in ckpt_paths:
        print(f"    {os.path.basename(p)}")
    ensemble_models = [load_model(p, device) for p in ckpt_paths]

    rng_ens = np.random.default_rng(args.seed)
    logits_ens, labels_ens = collect_logits(
        ensemble_models, loader, device, args.n_samples_per_batch, rng_ens
    )
    print(f"  Collected {logits_ens.shape[0]:,} voxels\n")

    # ------------------------------------------------------------------
    # ECE for all three configurations
    # ------------------------------------------------------------------
    configs = [
        ("Single model  (T=1.00)",          logits_single, labels_single, 1.0),
        ("Ensemble      (T=1.00)",           logits_ens,    labels_ens,    1.0),
        (f"Ensemble      (T={T_star:.4f})",  logits_ens,    labels_ens,    T_star),
    ]

    all_ece_global     = []
    all_ece_classwise  = []
    all_bin_stats      = []

    for cfg_name, logits, labels, T in configs:
        probs_np  = F.softmax(logits / T, dim=1).numpy()
        labels_np = labels.numpy()

        ece_g              = compute_ece_global(probs_np, labels_np, args.n_bins)
        ece_cw, bin_stats  = compute_ece_classwise(probs_np, labels_np, args.n_bins)

        all_ece_global.append(ece_g)
        all_ece_classwise.append(ece_cw)
        all_bin_stats.append(bin_stats)

    # ------------------------------------------------------------------
    # Print 3-row comparison table
    # ------------------------------------------------------------------
    col_w   = 30
    cls_w   = 7
    header  = f"{'Configuration':<{col_w}} {'ECE_global':>{cls_w}}"
    for name in CLASS_NAMES:
        header += f"  {name[:cls_w]:>{cls_w}}"
    sep = "-" * len(header)

    print(sep)
    print(header)
    print(sep)
    for i, (cfg_name, _, _, T) in enumerate(configs):
        row = f"{cfg_name:<{col_w}} {all_ece_global[i]:>{cls_w}.4f}"
        for k in range(NUM_CLASS):
            row += f"  {all_ece_classwise[i][k]:>{cls_w}.4f}"
        print(row)
    print(sep)

    # ------------------------------------------------------------------
    # Reliability diagram (ensemble uncal vs cal)
    # ------------------------------------------------------------------
    diagram_path = os.path.join(args.out_dir, f"reliability_{args.modality}.png")
    plot_reliability_diagrams(
        bin_stats_uncal=all_bin_stats[1], ece_uncal=all_ece_classwise[1],
        bin_stats_cal=all_bin_stats[2],   ece_cal=all_ece_classwise[2],
        modality=args.modality,
        out_path=diagram_path,
        n_bins=args.n_bins,
    )

    # ------------------------------------------------------------------
    # Save CSV
    # ------------------------------------------------------------------
    rows = []
    for i, (cfg_name, _, _, T) in enumerate(configs):
        row = {"configuration": cfg_name, "temperature": T,
               "ECE_global": all_ece_global[i]}
        for k in range(NUM_CLASS):
            row[f"ECE_{CLASS_NAMES[k].replace(' ', '_')}"] = all_ece_classwise[i][k]
        rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = os.path.join(args.out_dir, f"ece_{args.modality}.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved ECE table → {csv_path}")


if __name__ == "__main__":
    main()
