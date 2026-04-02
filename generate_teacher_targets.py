"""
generate_teacher_targets.py

Runs ensemble_tta_predict() over all training slices and saves per-slice
teacher targets for knowledge distillation:

  teacher_dir/<patient_name>/soft_probs<NNNN>.npy  -- shape (8, 224, 224), float32
  teacher_dir/<patient_name>/uncertainty<NNNN>.npy -- shape (1, 224, 224), float32

Each slice is processed independently (T=1 sequence) to avoid sequence-length
alignment complexity during distillation training.

Usage:
    python generate_teacher_targets.py \\
        --ckpt_dir /path/to/checkpoints \\
        --data_dirs /path/to/ct/train /path/to/ct/train2 \\
        --teacher_dir /path/to/teacher_targets \\
        --min_epoch 50 --n_ckpts 10 --n_tta 8
"""

import argparse
import glob
import os

import numpy as np
import torch
from tqdm import tqdm

from models.reslstmunet import ResLSTMUNet
from uncertainty import ensemble_tta_predict

NUM_CLASS = 8


def find_checkpoints(ckpt_dir: str, n_ckpts: int, min_epoch: int) -> list:
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


def load_ensemble(ckpt_dir: str, n_ckpts: int, min_epoch: int,
                  device: torch.device) -> list:
    ckpt_paths = find_checkpoints(ckpt_dir, n_ckpts, min_epoch)
    print(f"Loading {len(ckpt_paths)} checkpoints:")
    models = []
    for p in ckpt_paths:
        print(f"  {os.path.basename(p)}")
        model = ResLSTMUNet(in_channels=1, out_channels=NUM_CLASS,
                            pretrained=False, deep_sup=True, multiscale_att=True)
        state = torch.load(p, map_location=device)
        if all(k.startswith("module.") for k in state.keys()):
            state = {k[len("module."):]: v for k, v in state.items()}
        model.load_state_dict(state)
        model.to(device)
        model.eval()
        models.append(model)
    return models


def run_patient(models: list, patient_dir: str, teacher_dir: str,
                device: torch.device, n_tta: int) -> None:
    """Process all slices for one patient directory."""
    patient_name = os.path.basename(patient_dir)
    out_dir = os.path.join(teacher_dir, patient_name)
    os.makedirs(out_dir, exist_ok=True)

    # Collect and sort image paths the same way the dataset does
    image_paths = sorted(
        [os.path.join(patient_dir, f) for f in os.listdir(patient_dir)
         if f.startswith("image") and f.endswith(".npy")],
        key=lambda x: int(x[-8:-4]),
    )

    if not image_paths:
        print(f"  WARNING: no image*.npy files found in {patient_dir}")
        return

    for image_path in image_paths:
        slice_num = os.path.basename(image_path)[-8:-4]  # e.g. '0042'
        sp_path  = os.path.join(out_dir, f"soft_probs{slice_num}.npy")
        unc_path = os.path.join(out_dir, f"uncertainty{slice_num}.npy")

        # Skip if already generated (allows resuming interrupted runs)
        if os.path.exists(sp_path) and os.path.exists(unc_path):
            continue

        # Load and preprocess slice (mirrors WHSDataset_2D_scale_partSeries)
        import torch.nn.functional as F
        raw = np.load(image_path)
        img = torch.from_numpy(raw.astype("float32")).unsqueeze(0).unsqueeze(0)
        img = F.interpolate(img, [224, 224], mode="bilinear", align_corners=False)
        img = img.to(device)  # (1, 1, 224, 224)

        # Wrap as T=1 serial for ensemble_tta_predict
        x_serial = [img]

        with torch.no_grad():
            mean_probs, uncertainty = ensemble_tta_predict(
                models, x_serial, n_transforms=n_tta)

        # mean_probs[0]: (1, 8, 224, 224), uncertainty[0]: (1, 224, 224)
        sp  = mean_probs[0].squeeze(0).cpu().float().numpy()    # (8, 224, 224)
        unc = uncertainty[0].unsqueeze(0).cpu().float().numpy() # (1, 224, 224)

        np.save(sp_path,  sp)
        np.save(unc_path, unc)

    torch.cuda.empty_cache()


def discover_patient_dirs(data_dirs: list) -> list:
    """Mirror the directory traversal in WHSDataset_2D_scale_partSeries.__init__."""
    patient_dirs = []
    for data_dir in data_dirs:
        for name in os.listdir(data_dir):
            full = os.path.join(data_dir, name)
            if os.path.isdir(full):
                patient_dirs.append(full)
    return sorted(patient_dirs)


def main():
    parser = argparse.ArgumentParser(
        description="Pre-generate teacher soft predictions and uncertainty maps.")
    parser.add_argument("--ckpt_dir", required=True,
                        help="Directory containing epoch_*.pth checkpoints.")
    parser.add_argument("--data_dirs", required=True, nargs="+",
                        help="Training data directories (same as train_paths in YAML).")
    parser.add_argument("--teacher_dir", required=True,
                        help="Output root for teacher targets.")
    parser.add_argument("--min_epoch", type=int, default=50,
                        help="Only use checkpoints from this epoch onwards.")
    parser.add_argument("--n_ckpts", type=int, default=10,
                        help="Maximum number of checkpoints to use.")
    parser.add_argument("--n_tta", type=int, default=8,
                        help="Number of TTA transforms (1-8; identity always included).")
    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    models = load_ensemble(args.ckpt_dir, args.n_ckpts, args.min_epoch, device)
    patient_dirs = discover_patient_dirs(args.data_dirs)
    print(f"Found {len(patient_dirs)} patient directories.")

    for patient_dir in tqdm(patient_dirs, desc="Patients"):
        run_patient(models, patient_dir, args.teacher_dir, device, args.n_tta)

    print("Done. Teacher targets saved to:", args.teacher_dir)


if __name__ == "__main__":
    main()
