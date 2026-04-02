"""
Visualise Ensemble+TTA prediction + predictive uncertainty map for a single patient.

Produces a figure with four columns per selected slice:
  Col 0 – CT/MR image
  Col 1 – Ground truth
  Col 2 – Ensemble+TTA prediction (mean softmax probs across all checkpoint × augmentation combinations)
  Col 3 – Predictive uncertainty map (sum of per-class variance across all combinations)

The uncertainty colormap is normalised globally across all slices of the patient
so that colour values are directly comparable across rows.

Usage (CT):
    python visualize_uncertainty.py \
        --ckpt_dir    /content/drive/MyDrive/cardiac-project/output_aug_ensemble_ct/checkpoints/ResUNet_LSTM \
        --patient_dir /content/drive/MyDrive/cardiac-project/data/converted/ct/test/1019 \
        --out_dir     /content/drive/MyDrive/cardiac-project/viz_uncertainty \
        --n_slices    6  --n_ckpts 10  --modality ct

Usage (MR):
    python visualize_uncertainty.py \
        --ckpt_dir    /content/drive/MyDrive/cardiac-project/output_aug_ensemble_mr/checkpoints/ResUNet_LSTM \
        --patient_dir /content/drive/MyDrive/cardiac-project/data/converted/mr/test/1019 \
        --out_dir     /content/drive/MyDrive/cardiac-project/viz_uncertainty \
        --n_slices    6  --n_ckpts 10  --modality mr
"""

import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import numpy as np
import torch
import torch.nn.functional as F

from models.reslstmunet import ResLSTMUNet
from uncertainty import ensemble_tta_predict

NUM_CLASS  = 8
CLASS_NAMES = [
    "background", "myocardium", "left atrium", "left ventricle",
    "right atrium", "right ventricle", "ascending aorta", "pulmonary artery",
]
COLOURS = [
    "#000000",  # background
    "#FF6B6B",  # myocardium
    "#4ECDC4",  # left atrium
    "#45B7D1",  # left ventricle
    "#96CEB4",  # right atrium
    "#FFEAA7",  # right ventricle
    "#DDA0DD",  # ascending aorta
    "#FF8C42",  # pulmonary artery
]
CMAP = ListedColormap(COLOURS)


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
# Patient slice loading
# ---------------------------------------------------------------------------

def load_patient_slices(patient_dir):
    files = sorted(
        [f for f in os.listdir(patient_dir) if f.startswith("image")],
        key=lambda x: int(x[-8:-4]),
    )
    images, labels = [], []
    for f in files:
        images.append(np.load(os.path.join(patient_dir, f)))
        lbl_name = f.replace("image", "label")
        labels.append(np.load(os.path.join(patient_dir, lbl_name)))
    return images, labels


# ---------------------------------------------------------------------------
# Ensemble+TTA inference — returns predictions and per-voxel uncertainty
# ---------------------------------------------------------------------------

def predict_ensemble(models, images, device, crop_d=18, n_tta=None):
    """
    Sliding-window Ensemble+TTA inference over all slices of a patient.

    For each window, runs all N models × all TTA transforms and computes:
      - mean prediction  : argmax of mean softmax probabilities
      - uncertainty map  : sum of per-class variance across all
                           (checkpoint × augmentation) combinations

    Returns:
        preds : (N_slices, 224, 224) int   — predicted class index
        unc   : (N_slices, 224, 224) float — predictive uncertainty
    """
    n = len(images)
    preds    = [None] * n
    unc_maps = [None] * n

    # Build sliding windows (same logic as dataset.py)
    windows = []
    if n < crop_d:
        window = list(range(n)) + [n - 1] * (crop_d - n)
        windows.append((0, window))
    else:
        stride, start = 5, 0
        while start + crop_d <= n:
            windows.append((start, list(range(start, start + crop_d))))
            start += stride
        windows.append((n - crop_d, list(range(n - crop_d, n))))

    for _, indices in windows:
        # Build input sequence
        seq = []
        for i in indices:
            img = torch.from_numpy(images[i].astype("float32")).unsqueeze(0).unsqueeze(0)
            img = F.interpolate(img, [224, 224], mode="bilinear", align_corners=False)
            seq.append(img.to(device))

        mean_probs_t, unc_t = ensemble_tta_predict(models, seq, n_transforms=n_tta)
        # mean_probs_t: list of T tensors (1, C, H, W)
        # unc_t:        list of T tensors (1, H, W)

        for t, real_idx in enumerate(indices):
            if real_idx < n:
                preds[real_idx]    = mean_probs_t[t][0].argmax(dim=0).cpu().numpy()
                unc_maps[real_idx] = unc_t[t][0].cpu().numpy()

    return np.stack(preds), np.stack(unc_maps)


# ---------------------------------------------------------------------------
# Slice selection
# ---------------------------------------------------------------------------

def select_slices(labels, n_slices):
    foreground = [np.sum(lbl > 0) for lbl in labels]
    sorted_idx = np.argsort(foreground)[::-1]
    return sorted(sorted_idx[:n_slices].tolist())


# ---------------------------------------------------------------------------
# Main figure
# ---------------------------------------------------------------------------

def visualize(patient_dir, ckpt_dir, n_ckpts, out_dir, n_slices, crop_d,
              modality, device, n_tta=None):
    os.makedirs(out_dir, exist_ok=True)
    patient_id = os.path.basename(patient_dir)

    print(f"Loading slices for patient {patient_id} ...")
    images, labels = load_patient_slices(patient_dir)
    print(f"  {len(images)} slices found")

    ckpt_paths = find_checkpoints(ckpt_dir, n_ckpts)
    print(f"Loading {len(ckpt_paths)} checkpoints ...")
    models = [load_model(p, device) for p in ckpt_paths]

    print("Running ensemble+TTA inference ...")
    preds, unc_maps = predict_ensemble(models, images, device, crop_d, n_tta=n_tta)

    chosen = select_slices(labels, n_slices)
    print(f"Visualising slices: {chosen}")

    # Global uncertainty range across all selected slices (consistent colorbar)
    unc_vmax = max(unc_maps[i].max() for i in chosen)
    unc_vmin = 0.0

    fig, axes = plt.subplots(n_slices, 4, figsize=(16, 4 * n_slices))
    if n_slices == 1:
        axes = axes[np.newaxis, :]

    col_titles = ["Image", "Ground truth", "Ensemble+TTA prediction", "Uncertainty (predictive)"]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=10, fontweight="bold")

    im_unc = None
    for row, idx in enumerate(chosen):
        # Resize GT to 224×224
        lbl_t   = torch.from_numpy(labels[idx].astype("float32")).unsqueeze(0).unsqueeze(0)
        lbl_224 = F.interpolate(lbl_t, [224, 224], mode="nearest").squeeze().numpy()
        img_t   = torch.from_numpy(images[idx].astype("float32")).unsqueeze(0).unsqueeze(0)
        img_224 = F.interpolate(img_t, [224, 224], mode="bilinear",
                                align_corners=False).squeeze().numpy()

        # Col 0 — image
        axes[row, 0].imshow(img_224, cmap="gray", vmin=0, vmax=1)
        axes[row, 0].set_ylabel(f"Slice {idx:04d}", fontsize=8)

        # Col 1 — ground truth
        axes[row, 1].imshow(lbl_224, cmap=CMAP, vmin=0,
                            vmax=NUM_CLASS - 1, interpolation="nearest")

        # Col 2 — ensemble+TTA prediction
        axes[row, 2].imshow(preds[idx], cmap=CMAP, vmin=0,
                            vmax=NUM_CLASS - 1, interpolation="nearest")

        # Col 3 — uncertainty heat map
        im_unc = axes[row, 3].imshow(
            unc_maps[idx], cmap="plasma",
            vmin=unc_vmin, vmax=unc_vmax, interpolation="bilinear"
        )

        for ax in axes[row]:
            ax.axis("off")
        axes[row, 0].axis("on")
        axes[row, 0].set_xticks([])
        axes[row, 0].set_yticks([])

    # Shared colorbar for uncertainty column
    cbar_ax = fig.add_axes([0.76, 0.05, 0.01, 0.88])
    fig.colorbar(im_unc, cax=cbar_ax, label="Predictive uncertainty")

    # Legend for segmentation classes
    patches = [mpatches.Patch(color=COLOURS[c], label=CLASS_NAMES[c])
               for c in range(1, NUM_CLASS)]
    fig.legend(handles=patches, loc="lower center", ncol=4,
               bbox_to_anchor=(0.38, -0.01), fontsize=8)

    n_tta_display = n_tta if n_tta is not None else 8
    fig.suptitle(
        f"{modality.upper()} patient {patient_id} — Ensemble+TTA prediction & predictive uncertainty "
        f"({len(ckpt_paths)} checkpoints × {n_tta_display + 1} transforms)",
        fontsize=12, y=1.01
    )
    plt.tight_layout(rect=[0, 0, 0.75, 1])

    out_path = os.path.join(out_dir, f"uncertainty_{modality}_{patient_id}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Visualise Ensemble+TTA prediction + predictive uncertainty map")
    parser.add_argument("--ckpt_dir",    required=True,
                        help="Directory with epoch_*.pth ensemble checkpoints")
    parser.add_argument("--patient_dir", required=True,
                        help="Path to a single converted patient directory")
    parser.add_argument("--out_dir",     default="./viz_uncertainty")
    parser.add_argument("--n_slices",    type=int, default=6)
    parser.add_argument("--n_ckpts",     type=int, default=10)
    parser.add_argument("--crop_d",      type=int, default=18)
    parser.add_argument("--n_tta",       type=int, default=None,
                        help="TTA transforms per checkpoint (default: all 8)")
    parser.add_argument("--modality",    default="ct", choices=["ct", "mr"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps"  if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    visualize(args.patient_dir, args.ckpt_dir, args.n_ckpts,
              args.out_dir, args.n_slices, args.crop_d, args.modality, device,
              n_tta=args.n_tta)


if __name__ == "__main__":
    main()
