"""
Step 1+2: Calibrate temperature scaling on the validation set.

Loads an ensemble of checkpoints, collects averaged logits on the val set
(random voxel subsample), sweeps T over [t_min, t_max], picks the T that
minimises NLL, and saves it to a JSON file.

Usage (CT):
    python calibrate_temperature.py \
        --ckpt_dir /content/drive/MyDrive/cardiac-project/output_aug_ensemble_ct/checkpoints/ResUNet_LSTM \
        --val_dir  /content/drive/MyDrive/cardiac-project/data/converted/ct/val \
        --out      /content/drive/MyDrive/cardiac-project/calibration/ct_temperature.json \
        --n_ckpts  10

Usage (MR):
    python calibrate_temperature.py \
        --ckpt_dir /content/drive/MyDrive/cardiac-project/output_aug_ensemble_mr/checkpoints/ResUNet_LSTM \
        --val_dir  /content/drive/MyDrive/cardiac-project/data/converted/mr/val \
        --out      /content/drive/MyDrive/cardiac-project/calibration/mr_temperature.json \
        --n_ckpts  10
"""

import argparse
import glob
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import WHSDataset_2D_scale_partSeries
from models.reslstmunet import ResLSTMUNet

NUM_CLASS = 8


# ---------------------------------------------------------------------------
# Model helpers (same pattern as eval_ensemble.py)
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
        if not paths:
            raise ValueError(f"No checkpoints at or after epoch {min_epoch}")
    if n_ckpts:
        paths = paths[:n_ckpts]
    return paths


# ---------------------------------------------------------------------------
# Logit collection
# ---------------------------------------------------------------------------

def collect_logits(models, loader, device, n_samples_per_batch, rng):
    """
    Run ensemble inference and collect randomly subsampled (logit, label) pairs.

    Averages raw logits across checkpoints (no softmax), then subsamples
    n_samples_per_batch voxels per batch to keep memory bounded.

    Returns:
        all_logits : (N_total, NUM_CLASS) float32 CPU tensor
        all_labels : (N_total,)           int64   CPU tensor
    """
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for image_serial, label_serial in loader:
            image_serial = [img.to(device) for img in image_serial]

            # Average raw logits across checkpoints (no softmax yet)
            logit_sum = None
            for model in models:
                pred_serial, *_ = model(image_serial)          # list of T tensors (B, C, H, W)
                logits_t = torch.stack(pred_serial, dim=0)     # (T, B, C, H, W)
                logit_sum = logits_t if logit_sum is None else logit_sum + logits_t
            mean_logits = logit_sum / len(models)              # (T, B, C, H, W)

            # Labels: list of T tensors (B, 1, H, W) → (T, B, H, W)
            labels = torch.stack(
                [lb.squeeze(1).long() for lb in label_serial], dim=0
            )

            # Flatten to voxel list
            T, B, C, H, W = mean_logits.shape
            flat_logits = mean_logits.permute(0, 1, 3, 4, 2).reshape(-1, C).cpu()  # (T*B*H*W, C)
            flat_labels = labels.reshape(-1).cpu()                                  # (T*B*H*W,)

            # Random subsample
            n = flat_logits.shape[0]
            idx = torch.from_numpy(
                rng.choice(n, size=min(n_samples_per_batch, n), replace=False)
            )
            all_logits.append(flat_logits[idx])
            all_labels.append(flat_labels[idx])

    return torch.cat(all_logits, dim=0), torch.cat(all_labels, dim=0)


# ---------------------------------------------------------------------------
# Temperature sweep
# ---------------------------------------------------------------------------

def sweep_temperature(logits, labels, t_min, t_max, t_steps):
    """
    Sweep T over a uniform grid and return (best_T, nll_values, T_grid).
    NLL = mean cross-entropy(logits / T, labels).
    """
    T_grid = np.linspace(t_min, t_max, t_steps)
    nll_values = []
    for T in T_grid:
        nll = F.cross_entropy(logits / float(T), labels).item()
        nll_values.append(nll)
    best_idx = int(np.argmin(nll_values))
    return float(T_grid[best_idx]), nll_values, T_grid.tolist()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Temperature scaling calibration")
    parser.add_argument("--ckpt_dir",            required=True,
                        help="Directory with epoch_*.pth checkpoints")
    parser.add_argument("--val_dir",             required=True,
                        help="Validation data directory (used to optimise T)")
    parser.add_argument("--out",                 required=True,
                        help="Output JSON path, e.g. calibration/ct_temperature.json")
    parser.add_argument("--n_ckpts",             type=int, default=10)
    parser.add_argument("--min_epoch",           type=int, default=None)
    parser.add_argument("--crop_d",              type=int, default=18)
    parser.add_argument("--batch_size",          type=int, default=4)
    parser.add_argument("--num_workers",         type=int, default=4)
    parser.add_argument("--n_samples_per_batch", type=int, default=2000,
                        help="Voxels randomly sampled per batch (controls memory use)")
    parser.add_argument("--t_min",               type=float, default=0.5)
    parser.add_argument("--t_max",               type=float, default=3.0)
    parser.add_argument("--t_steps",             type=int,   default=100)
    parser.add_argument("--seed",                type=int,   default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps"  if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    # Load checkpoints
    ckpt_paths = find_checkpoints(args.ckpt_dir, args.n_ckpts, args.min_epoch)
    print(f"\nLoading {len(ckpt_paths)} checkpoints:")
    for p in ckpt_paths:
        print(f"  {os.path.basename(p)}")
    models = [load_model(p, device) for p in ckpt_paths]

    # Validation loader
    dataset = WHSDataset_2D_scale_partSeries([args.val_dir], crop_d=args.crop_d)
    loader  = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                         num_workers=args.num_workers, pin_memory=True)
    print(f"\nVal sequences: {len(dataset)}")

    # Collect logits
    rng = np.random.default_rng(args.seed)
    print(f"\nCollecting logits ({args.n_samples_per_batch} voxels/batch) ...")
    logits, labels = collect_logits(models, loader, device,
                                    args.n_samples_per_batch, rng)
    print(f"  Total voxel samples collected: {logits.shape[0]:,}")

    # Baseline NLL at T=1 (uncalibrated)
    nll_uncal = F.cross_entropy(logits, labels).item()

    # Sweep T
    print(f"\nSweeping T in [{args.t_min}, {args.t_max}] ({args.t_steps} steps) ...")
    best_T, nll_values, T_grid = sweep_temperature(
        logits, labels, args.t_min, args.t_max, args.t_steps
    )
    nll_cal = F.cross_entropy(logits / best_T, labels).item()

    print(f"\n  NLL  T=1.00 (uncalibrated) : {nll_uncal:.4f}")
    print(f"  NLL  T={best_T:.4f} (calibrated)   : {nll_cal:.4f}")
    print(f"\n  Optimal temperature T = {best_T:.4f}")

    # Save result
    result = {
        "temperature":       best_T,
        "nll_uncalibrated":  nll_uncal,
        "nll_calibrated":    nll_cal,
        "t_grid":            T_grid,
        "nll_curve":         nll_values,
        "n_voxel_samples":   int(logits.shape[0]),
        "n_checkpoints":     len(ckpt_paths),
        "checkpoints":       [os.path.basename(p) for p in ckpt_paths],
        "val_dir":           args.val_dir,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved → {args.out}")


if __name__ == "__main__":
    main()