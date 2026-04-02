"""
Reliability diagrams for LV and PA — uncalibrated vs calibrated.
Covers both CT and MRI in a single publication-ready figure.

Layout: 2 rows (CT, MRI) × 4 cols (LV uncal | LV cal | PA uncal | PA cal)

Usage:
    python plot_reliability_pa_lv.py \
        --ct_ckpt_dir   /content/drive/MyDrive/cardiac-project/output_aug_ensemble_ct/checkpoints/ResUNet_LSTM \
        --ct_test_dir   /content/drive/MyDrive/cardiac-project/data/converted/ct/test \
        --ct_temp_json  /content/drive/MyDrive/cardiac-project/calibration/ct_temperature.json \
        --mr_ckpt_dir   /content/drive/MyDrive/cardiac-project/output_aug_ensemble_mr/checkpoints/ResUNet_LSTM \
        --mr_test_dir   /content/drive/MyDrive/cardiac-project/data/converted/mr/test \
        --mr_temp_json  /content/drive/MyDrive/cardiac-project/calibration/mr_temperature.json \
        --out_dir       /content/drive/MyDrive/cardiac-project/calibration \
        --n_ckpts       10
"""

import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import WHSDataset_2D_scale_partSeries
from models.reslstmunet import ResLSTMUNet

NUM_CLASS   = 8
CLASS_NAMES = [
    "background", "myocardium", "left atrium", "left ventricle",
    "right atrium", "right ventricle", "ascending aorta", "pulmonary artery",
]
IDX_LV = 3   # left ventricle
IDX_PA = 7   # pulmonary artery


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


def find_checkpoints(ckpt_dir, n_ckpts):
    paths = sorted(
        glob.glob(os.path.join(ckpt_dir, "*epoch_*.pth")),
        key=lambda p: int(p.split("epoch_")[-1].replace(".pth", "")),
        reverse=True,
    )
    if not paths:
        raise FileNotFoundError(f"No epoch_*.pth files found in {ckpt_dir}")
    return paths[:n_ckpts]


# ---------------------------------------------------------------------------
# Logit collection
# ---------------------------------------------------------------------------

def collect_logits(models, loader, device, n_samples_per_batch, rng):
    """Average ensemble logits and subsample voxels per batch."""
    all_logits, all_labels = [], []

    with torch.no_grad():
        for image_serial, label_serial in loader:
            image_serial = [img.to(device) for img in image_serial]

            logit_sum = None
            for model in models:
                pred_serial, *_ = model(image_serial)
                logits_t = torch.stack(pred_serial, dim=0)        # (T, B, C, H, W)
                logit_sum = logits_t if logit_sum is None else logit_sum + logits_t
            mean_logits = logit_sum / len(models)

            labels = torch.stack(
                [lb.squeeze(1).long() for lb in label_serial], dim=0
            )                                                      # (T, B, H, W)

            T, B, C, H, W = mean_logits.shape
            flat_logits = mean_logits.permute(0, 1, 3, 4, 2).reshape(-1, C).cpu()
            flat_labels = labels.reshape(-1).cpu()

            n = flat_logits.shape[0]
            idx = torch.from_numpy(
                rng.choice(n, size=min(n_samples_per_batch, n), replace=False)
            )
            all_logits.append(flat_logits[idx])
            all_labels.append(flat_labels[idx])

    return torch.cat(all_logits), torch.cat(all_labels)


# ---------------------------------------------------------------------------
# Per-class bin statistics (for reliability diagram)
# ---------------------------------------------------------------------------

def bin_stats_for_class(probs_np, labels_np, class_idx, n_bins=15):
    """
    Returns (bin_conf, bin_acc, ece) arrays of length n_bins.
    NaN for empty bins.
    """
    conf_k = probs_np[:, class_idx]
    pos_k  = (labels_np == class_idx).astype(float)
    bins   = np.linspace(0, 1, n_bins + 1)
    N      = probs_np.shape[0]

    bin_conf = np.full(n_bins, np.nan)
    bin_acc  = np.full(n_bins, np.nan)
    ece      = 0.0

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (conf_k >= lo) & (conf_k <= hi if i == n_bins - 1 else conf_k < hi)
        if mask.sum() > 0:
            bin_conf[i] = conf_k[mask].mean()
            bin_acc[i]  = pos_k[mask].mean()
            ece += (mask.sum() / N) * abs(bin_acc[i] - bin_conf[i])

    return bin_conf, bin_acc, ece


# ---------------------------------------------------------------------------
# Per-modality inference
# ---------------------------------------------------------------------------

def run_modality(ckpt_dir, test_dir, temp_json, n_ckpts, crop_d,
                 batch_size, num_workers, n_samples_per_batch, seed, device):
    """
    Load ensemble, collect logits, return bin_stats for LV and PA
    at T=1 (uncal) and T=T* (cal).

    Returns dict with keys: T_star, lv_uncal, lv_cal, pa_uncal, pa_cal
    Each value is (bin_conf, bin_acc, ece).
    """
    with open(temp_json) as f:
        T_star = json.load(f)["temperature"]

    ckpt_paths = find_checkpoints(ckpt_dir, n_ckpts)
    print(f"  Loading {len(ckpt_paths)} checkpoints ...")
    models = [load_model(p, device) for p in ckpt_paths]

    dataset = WHSDataset_2D_scale_partSeries([test_dir], crop_d=crop_d)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                         num_workers=num_workers, pin_memory=True)
    print(f"  Test sequences: {len(dataset)}")

    rng = np.random.default_rng(seed)
    print(f"  Collecting logits ...")
    logits, labels = collect_logits(models, loader, device, n_samples_per_batch, rng)
    print(f"  Collected {logits.shape[0]:,} voxels")

    probs_np    = F.softmax(logits,          dim=1).numpy()
    probs_np_cal = F.softmax(logits / T_star, dim=1).numpy()
    labels_np   = labels.numpy()

    return {
        "T_star":   T_star,
        "lv_uncal": bin_stats_for_class(probs_np,     labels_np, IDX_LV),
        "lv_cal":   bin_stats_for_class(probs_np_cal, labels_np, IDX_LV),
        "pa_uncal": bin_stats_for_class(probs_np,     labels_np, IDX_PA),
        "pa_cal":   bin_stats_for_class(probs_np_cal, labels_np, IDX_PA),
    }


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def _draw_cell(ax, bin_conf, bin_acc, ece, title, n_bins=15):
    """Draw a single reliability diagram cell."""
    bin_centers = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_centers[:-1] + bin_centers[1:]) / 2

    gap    = np.where(np.isnan(bin_acc), 0.0, bin_acc - bin_conf)
    colors = ["#d62728" if g < 0 else "#1f77b4" for g in gap]
    heights = np.where(np.isnan(bin_acc), 0.0, bin_acc)

    ax.bar(bin_centers, heights, width=1 / n_bins,
           color=colors, alpha=0.65, align="center", zorder=2)
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.9, alpha=0.5, zorder=3)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_title(f"{title}\nECE = {ece:.4f}", fontsize=8)
    ax.tick_params(labelsize=6)
    ax.set_xlabel("Confidence", fontsize=7)


def plot_figure(ct_stats, mr_stats, out_path, n_bins=15):
    """
    2 rows (CT, MRI) × 4 cols (LV uncal | LV cal | PA uncal | PA cal).
    """
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.5))
    fig.suptitle(
        "Reliability diagrams — LV and PA, uncalibrated vs calibrated",
        fontsize=11, y=1.02
    )

    col_titles = [
        "LV — Uncalibrated (T=1)",
        f"LV — Calibrated (T={ct_stats['T_star']:.2f} / {mr_stats['T_star']:.2f})",
        "PA — Uncalibrated (T=1)",
        f"PA — Calibrated (T={ct_stats['T_star']:.2f} / {mr_stats['T_star']:.2f})",
    ]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=8, fontweight="bold", pad=14)

    row_data = [
        ("CT", ct_stats),
        ("MRI", mr_stats),
    ]
    keys = ["lv_uncal", "lv_cal", "pa_uncal", "pa_cal"]
    subtitles = ["LV uncal", "LV cal", "PA uncal", "PA cal"]

    for row, (modality, stats) in enumerate(row_data):
        axes[row, 0].set_ylabel(modality, fontsize=9, fontweight="bold", labelpad=10)
        for col, (key, subtitle) in enumerate(zip(keys, subtitles)):
            bin_conf, bin_acc, ece = stats[key]
            _draw_cell(axes[row, col], bin_conf, bin_acc, ece,
                       title=subtitle, n_bins=n_bins)

    # Shared legend for bar colours
    from matplotlib.patches import Patch
    legend_elems = [
        Patch(facecolor="#1f77b4", alpha=0.65, label="Over-confident (acc > conf)"),
        Patch(facecolor="#d62728", alpha=0.65, label="Under-confident (acc < conf)"),
    ]
    fig.legend(handles=legend_elems, loc="lower center", ncol=2,
               fontsize=8, bbox_to_anchor=(0.5, -0.04))

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Reliability diagrams for LV and PA (CT + MRI, uncal vs cal)")

    # CT args
    parser.add_argument("--ct_ckpt_dir",  required=True)
    parser.add_argument("--ct_test_dir",  required=True)
    parser.add_argument("--ct_temp_json", required=True,
                        help="JSON from calibrate_temperature.py for CT")

    # MRI args
    parser.add_argument("--mr_ckpt_dir",  required=True)
    parser.add_argument("--mr_test_dir",  required=True)
    parser.add_argument("--mr_temp_json", required=True,
                        help="JSON from calibrate_temperature.py for MRI")

    parser.add_argument("--out_dir",             required=True)
    parser.add_argument("--n_ckpts",             type=int, default=10)
    parser.add_argument("--crop_d",              type=int, default=18)
    parser.add_argument("--batch_size",          type=int, default=4)
    parser.add_argument("--num_workers",         type=int, default=4)
    parser.add_argument("--n_samples_per_batch", type=int, default=5000)
    parser.add_argument("--n_bins",              type=int, default=15)
    parser.add_argument("--seed",                type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps"  if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}\n")

    shared = dict(n_ckpts=args.n_ckpts, crop_d=args.crop_d,
                  batch_size=args.batch_size, num_workers=args.num_workers,
                  n_samples_per_batch=args.n_samples_per_batch,
                  seed=args.seed, device=device)

    print("=== CT ===")
    ct_stats = run_modality(args.ct_ckpt_dir, args.ct_test_dir,
                             args.ct_temp_json, **shared)

    print("\n=== MRI ===")
    mr_stats = run_modality(args.mr_ckpt_dir, args.mr_test_dir,
                             args.mr_temp_json, **shared)

    out_path = os.path.join(args.out_dir, "reliability_pa_lv.png")
    plot_figure(ct_stats, mr_stats, out_path, n_bins=args.n_bins)


if __name__ == "__main__":
    main()
