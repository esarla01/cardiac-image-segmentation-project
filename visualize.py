"""
Visualize model predictions vs ground truth for a single patient.

Usage:
    python visualize.py \
        --ckpt /path/to/checkpoint.pth \
        --patient_dir /path/to/converted/ct/test/1019 \
        --out_dir ./viz_output \
        --n_slices 6
"""

import argparse
import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  # headless; works on Colab and servers
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

from models.reslstmunet import ResLSTMUNet

# ── colour palette (one colour per class, background=black) ────────────────
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

COLOURS = [
    "#000000",  # 0  background     – black
    "#FF6B6B",  # 1  myocardium     – red
    "#4ECDC4",  # 2  left atrium    – teal
    "#45B7D1",  # 3  left ventricle – blue
    "#96CEB4",  # 4  right atrium   – green
    "#FFEAA7",  # 5  right ventricle– yellow
    "#DDA0DD",  # 6  ascending aorta– plum
    "#FF8C42",  # 7  pulmonary artery– orange
]

CMAP = ListedColormap(COLOURS)
NUM_CLASS = 8


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


def load_patient_slices(patient_dir):
    """Return sorted lists of (image_array, label_array) for every slice."""
    files = sorted(
        [f for f in os.listdir(patient_dir) if f.startswith("image")],
        key=lambda x: int(x[-8:-4])
    )
    images, labels = [], []
    for f in files:
        img = np.load(os.path.join(patient_dir, f))          # (H, W)
        lbl_name = f.replace("image", "label")
        lbl = np.load(os.path.join(patient_dir, lbl_name))   # (H, W)
        images.append(img)
        labels.append(lbl)
    return images, labels


def predict_slices(model, images, device, crop_d=18):
    """
    Run inference slice-by-slice using a sliding window of length crop_d.
    For each slice index, use the window that centres (or ends at) that slice.
    Returns a numpy array of shape (N_slices, H, W) with predicted class indices.
    """
    import torch.nn.functional as F

    n = len(images)
    preds = [None] * n

    # Build windows the same way the dataset does
    windows = []
    if n < crop_d:
        window = list(range(n)) + [n - 1] * (crop_d - n)
        windows.append((0, window))
    else:
        stride = 5
        start = 0
        while start + crop_d <= n:
            windows.append((start, list(range(start, start + crop_d))))
            start += stride
        # last window
        windows.append((n - crop_d, list(range(n - crop_d, n))))

    with torch.no_grad():
        for window_start, indices in windows:
            seq = []
            for i in indices:
                img = torch.from_numpy(images[i].astype("float32")).unsqueeze(0).unsqueeze(0)
                img = F.interpolate(img, [224, 224], mode="bilinear", align_corners=False)
                seq.append(img.to(device))  # keep (1, 1, H, W)

            pred_serial, *_ = model(seq)  # list of T tensors (1, C, H, W)

            for t, real_idx in enumerate(indices):
                if real_idx < n:
                    p = pred_serial[t].argmax(dim=1)[0].cpu().numpy()  # (224, 224)
                    preds[real_idx] = p  # last write wins for overlapping windows

    return np.stack(preds)  # (N, 224, 224)


def select_slices(labels, n_slices):
    """Pick slices that have the most foreground content (most interesting)."""
    foreground = [np.sum(lbl > 0) for lbl in labels]
    sorted_idx = np.argsort(foreground)[::-1]
    chosen = sorted(sorted_idx[:n_slices].tolist())
    return chosen


def make_legend():
    patches = [
        mpatches.Patch(color=COLOURS[c], label=CLASS_NAMES[c])
        for c in range(1, NUM_CLASS)
    ]
    return patches


def visualize(patient_dir, ckpt_path, out_dir, n_slices, crop_d, device):
    os.makedirs(out_dir, exist_ok=True)
    patient_id = os.path.basename(patient_dir)

    print(f"Loading slices for patient {patient_id} ...")
    images, labels = load_patient_slices(patient_dir)
    print(f"  {len(images)} slices found")

    model = load_model(ckpt_path, device)
    print("Running inference ...")
    preds = predict_slices(model, images, device, crop_d)

    chosen = select_slices(labels, n_slices)
    print(f"Visualising slices: {chosen}")

    fig, axes = plt.subplots(n_slices, 3, figsize=(12, 4 * n_slices))
    if n_slices == 1:
        axes = axes[np.newaxis, :]

    for row, idx in enumerate(chosen):
        # Resize GT label to 224×224 to match model output
        import torch.nn.functional as F
        lbl_t = torch.from_numpy(labels[idx].astype("float32")).unsqueeze(0).unsqueeze(0)
        lbl_224 = F.interpolate(lbl_t, [224, 224], mode="nearest").squeeze().numpy()

        img_t = torch.from_numpy(images[idx].astype("float32")).unsqueeze(0).unsqueeze(0)
        img_224 = F.interpolate(img_t, [224, 224], mode="bilinear", align_corners=False).squeeze().numpy()

        # Column 0 – CT image
        axes[row, 0].imshow(img_224, cmap="gray", vmin=0, vmax=1)
        axes[row, 0].set_title(f"Slice {idx:04d} – CT image")

        # Column 1 – Ground truth
        axes[row, 1].imshow(lbl_224, cmap=CMAP, vmin=0, vmax=NUM_CLASS - 1, interpolation="nearest")
        axes[row, 1].set_title("Ground truth")

        # Column 2 – Prediction
        axes[row, 2].imshow(preds[idx], cmap=CMAP, vmin=0, vmax=NUM_CLASS - 1, interpolation="nearest")
        axes[row, 2].set_title("Prediction")

        for ax in axes[row]:
            ax.axis("off")

    fig.legend(handles=make_legend(), loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.01), fontsize=9)
    fig.suptitle(f"Patient {patient_id}", fontsize=14, y=1.01)
    plt.tight_layout()

    out_path = os.path.join(out_dir, f"patient_{patient_id}_predictions.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out_path}")

    # ── per-class overlay: difference map ──────────────────────────────────
    fig2, axes2 = plt.subplots(n_slices, NUM_CLASS - 1,
                               figsize=(3 * (NUM_CLASS - 1), 3 * n_slices))
    if n_slices == 1:
        axes2 = axes2[np.newaxis, :]

    for row, idx in enumerate(chosen):
        lbl_t = torch.from_numpy(labels[idx].astype("float32")).unsqueeze(0).unsqueeze(0)
        lbl_224 = F.interpolate(lbl_t, [224, 224], mode="nearest").squeeze().numpy()

        for col, c in enumerate(range(1, NUM_CLASS)):
            gt_mask   = (lbl_224   == c).astype(float)
            pred_mask = (preds[idx] == c).astype(float)

            # Overlap image: TP=green, FN=red, FP=blue
            overlay = np.zeros((224, 224, 3))
            overlay[(gt_mask == 1) & (pred_mask == 1)] = [0, 0.8, 0]   # TP green
            overlay[(gt_mask == 1) & (pred_mask == 0)] = [0.9, 0, 0]   # FN red
            overlay[(gt_mask == 0) & (pred_mask == 1)] = [0, 0, 0.9]   # FP blue

            axes2[row, col].imshow(overlay, interpolation="nearest")
            axes2[row, col].set_title(CLASS_NAMES[c], fontsize=8)
            axes2[row, col].axis("off")

        axes2[row, 0].set_ylabel(f"Slice {chosen[row]:04d}", fontsize=8)

    tp_patch = mpatches.Patch(color=(0, 0.8, 0),  label="TP")
    fn_patch = mpatches.Patch(color=(0.9, 0, 0),  label="FN (missed)")
    fp_patch = mpatches.Patch(color=(0, 0, 0.9),  label="FP (over-seg)")
    fig2.legend(handles=[tp_patch, fn_patch, fp_patch], loc="lower center",
                ncol=3, bbox_to_anchor=(0.5, -0.01), fontsize=10)
    fig2.suptitle(f"Patient {patient_id} – Per-class overlap", fontsize=13, y=1.01)
    plt.tight_layout()

    out_path2 = os.path.join(out_dir, f"patient_{patient_id}_overlap.png")
    plt.savefig(out_path2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out_path2}")


def main():
    parser = argparse.ArgumentParser(description="Visualise predictions vs ground truth")
    parser.add_argument("--ckpt",        required=True, help="Path to .pth checkpoint")
    parser.add_argument("--patient_dir", required=True, help="Path to a single converted patient dir")
    parser.add_argument("--out_dir",     default="./viz_output", help="Where to save PNGs")
    parser.add_argument("--n_slices",    type=int, default=6,  help="Number of slices to show")
    parser.add_argument("--crop_d",      type=int, default=18, help="Sequence length used at training")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps"  if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    visualize(args.patient_dir, args.ckpt, args.out_dir,
              args.n_slices, args.crop_d, device)


if __name__ == "__main__":
    main()