"""
Inference timing benchmark for ResLSTMUNet.

Measures and compares wall-clock inference time per single forward pass
(one crop_d-slice window, batch_size=1) across four modes:
  1. Baseline       — single model, no TTA
  2. TTA            — single model, 8 affine TTA transforms + identity
  3. Ensemble       — N checkpoints averaged, no TTA
  4. Ensemble + TTA — N checkpoints × (8 TTA transforms + identity)

This matches how inference time is reported in the paper: one call to the
model for one sequence window (batch_size=1, T=crop_d slices).

Timing covers only the model forward pass(es). Data is pre-loaded onto
the device so disk I/O does not contaminate the measurements.

Usage:
    python3 eval_timing.py \
        --ckpt      /path/to/best.pth \
        --ckpt_dir  /path/to/checkpoints \
        --test_dir  /path/to/converted/ct/test \
        --n_ckpts   10 \
        --n_tta     8 \
        --n_warmup  10 \
        --n_reps    50 \
        --out_csv   timing_ct.csv
"""

import argparse
import glob
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import WHSDataset_2D_scale_partSeries
from models.reslstmunet import ResLSTMUNet
from uncertainty import tta_predict, ensemble_predict, ensemble_tta_predict

NUM_CLASS = 8


# ---------------------------------------------------------------------------
# Helpers
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


def sync(device):
    """Block until all pending device operations are complete."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    # MPS: no public synchronise; wall-clock is sufficiently accurate.


def time_mode(fn, windows, device, n_warmup, n_reps):
    """
    Time fn(window) over n_reps calls using batch_size=1 windows.

    Args:
        fn:       callable(image_serial) → (ignored)
        windows:  list of pre-loaded windows, each a list of T tensors
                  (1, 1, H, W) already on device. Cycled if n_reps > len(windows).
        device:   torch device
        n_warmup: number of discarded warm-up calls
        n_reps:   number of timed calls

    Returns:
        elapsed: np.array of per-call seconds, shape (n_reps,)
    """
    n = len(windows)

    # Warm-up
    for i in range(n_warmup):
        with torch.no_grad():
            fn(windows[i % n])
    sync(device)

    # Timed runs
    elapsed = []
    for i in range(n_reps):
        win = windows[i % n]
        sync(device)
        t0 = time.perf_counter()
        with torch.no_grad():
            fn(win)
        sync(device)
        elapsed.append(time.perf_counter() - t0)

    return np.array(elapsed)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Per-forward-pass inference timing benchmark (batch_size=1)")
    parser.add_argument("--ckpt",        required=False, default=None,
                        help="Single checkpoint for baseline & TTA modes")
    parser.add_argument("--ckpt_dir",    required=False, default=None,
                        help="Directory of epoch_*.pth checkpoints (ensemble modes)")
    parser.add_argument("--test_dir",    required=True,
                        help="Path to converted test directory")
    parser.add_argument("--crop_d",      type=int, default=18,
                        help="Sequence length in slices (default: 18)")
    parser.add_argument("--n_ckpts",     type=int, default=10,
                        help="Ensemble checkpoints to load (default: 10)")
    parser.add_argument("--n_tta",       type=int, default=8,
                        help="TTA transforms, identity excluded (default: 8)")
    parser.add_argument("--n_warmup",    type=int, default=10,
                        help="Warm-up calls per mode, discarded (default: 10)")
    parser.add_argument("--n_reps",      type=int, default=50,
                        help="Timed calls per mode (default: 50)")
    parser.add_argument("--min_epoch",   type=int, default=None)
    parser.add_argument("--out_csv",     default=None,
                        help="Optional path to save CSV results")
    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else
        "cpu"
    )
    print(f"Device        : {device}")
    print(f"Input         : batch_size=1, T={args.crop_d} slices")
    print(f"Warm-up calls : {args.n_warmup} per mode (discarded)")
    print(f"Timed calls   : {args.n_reps} per mode")
    print(f"TTA           : {args.n_tta} transforms + identity = {args.n_tta + 1} passes")
    print(f"Ensemble      : {args.n_ckpts} checkpoints\n")

    # ------------------------------------------------------------------ #
    # Load models
    # ------------------------------------------------------------------ #
    if args.ckpt is None and args.ckpt_dir is None:
        raise ValueError("Provide at least one of --ckpt or --ckpt_dir")

    if args.ckpt:
        baseline_ckpt = args.ckpt
    else:
        baseline_ckpt = find_checkpoints(args.ckpt_dir, 1, args.min_epoch)[0]

    print(f"Baseline checkpoint : {os.path.basename(baseline_ckpt)}")
    baseline_model = load_model(baseline_ckpt, device)

    if args.ckpt_dir:
        ckpt_paths = find_checkpoints(args.ckpt_dir, args.n_ckpts, args.min_epoch)
        print(f"Loading {len(ckpt_paths)} ensemble checkpoints …")
        ensemble_models = [load_model(p, device) for p in ckpt_paths]
        n_ckpts_loaded  = len(ensemble_models)
    else:
        print("No --ckpt_dir provided; ensemble modes will reuse baseline model.")
        ensemble_models = [baseline_model]
        n_ckpts_loaded  = 1

    n_tta_total = args.n_tta + 1
    print(f"\nForward passes per call:")
    print(f"  Baseline       : 1")
    print(f"  TTA            : {n_tta_total}  (identity + {args.n_tta})")
    print(f"  Ensemble       : {n_ckpts_loaded}")
    print(f"  Ensemble + TTA : {n_ckpts_loaded * n_tta_total}\n")

    # ------------------------------------------------------------------ #
    # Pre-load windows onto device (eliminates I/O variance)
    # ------------------------------------------------------------------ #
    print("Pre-loading windows …", flush=True)
    dataset = WHSDataset_2D_scale_partSeries(
        [args.test_dir], crop_d=args.crop_d, augment=False
    )
    loader  = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    windows = []
    for image_serial, _ in loader:
        windows.append([img.to(device) for img in image_serial])
        if len(windows) >= max(args.n_warmup, args.n_reps):
            break  # no need to load more than we'll use

    print(f"Windows loaded : {len(windows)}\n")

    # ------------------------------------------------------------------ #
    # Time each mode
    # ------------------------------------------------------------------ #
    results = {}

    print("Timing: Baseline …", flush=True)
    results["Baseline"] = time_mode(
        lambda w: baseline_model(w),
        windows, device, args.n_warmup, args.n_reps
    )

    print("Timing: TTA …", flush=True)
    results["TTA"] = time_mode(
        lambda w: tta_predict(baseline_model, w, n_transforms=args.n_tta),
        windows, device, args.n_warmup, args.n_reps
    )

    print("Timing: Ensemble …", flush=True)
    results["Ensemble"] = time_mode(
        lambda w: ensemble_predict(ensemble_models, w),
        windows, device, args.n_warmup, args.n_reps
    )

    print("Timing: Ensemble + TTA …", flush=True)
    results["Ensemble+TTA"] = time_mode(
        lambda w: ensemble_tta_predict(ensemble_models, w, n_transforms=args.n_tta),
        windows, device, args.n_warmup, args.n_reps
    )

    # ------------------------------------------------------------------ #
    # Report
    # ------------------------------------------------------------------ #
    header = (f"\n{'Mode':<18} {'Mean (s)':>10} {'Std':>8} "
              f"{'Min':>8} {'Max':>8}  {'×baseline':>10}")
    sep = "-" * len(header)
    print(header)
    print(sep)

    baseline_mean = results["Baseline"].mean()
    rows = []
    for mode, times in results.items():
        mean_t = times.mean()
        std_t  = times.std()
        min_t  = times.min()
        max_t  = times.max()
        ratio  = mean_t / baseline_mean
        print(f"{mode:<18} {mean_t:>10.4f} {std_t:>8.4f} "
              f"{min_t:>8.4f} {max_t:>8.4f}  {ratio:>9.2f}×")
        rows.append({
            "mode":       mode,
            "mean_s":     round(mean_t, 6),
            "std_s":      round(std_t,  6),
            "min_s":      round(min_t,  6),
            "max_s":      round(max_t,  6),
            "x_baseline": round(ratio,  4),
            "n_reps":     args.n_reps,
            "n_ckpts":    n_ckpts_loaded,
            "n_tta":      args.n_tta,
            "crop_d":     args.crop_d,
            "device":     str(device),
        })
    print(sep)

    # ------------------------------------------------------------------ #
    # Save CSV
    # ------------------------------------------------------------------ #
    if args.out_csv:
        import csv
        fieldnames = list(rows[0].keys())
        with open(args.out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nResults saved to: {args.out_csv}")


if __name__ == "__main__":
    main()
