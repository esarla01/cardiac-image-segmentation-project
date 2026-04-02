"""
train_distill.py

Distillation training for ResLSTMUNet with uncertainty head.

The student is warm-started from the teacher's best checkpoint.  Only the
new uncertainty_head parameters are randomly re-initialised.  Training uses
DistillationLoss at full resolution and plain WCEDCELoss at the deep-
supervision scales.

Usage:
    python train_distill.py --mode CT \\
        --teacher_ckpt /path/to/best_epoch_55.pth \\
        --teacher_dir  /path/to/teacher_targets
"""

import argparse
import copy
import os
import time
import yaml

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import WHSDataset_2D_scale_partSeries
from dataset_distill import WHSDataset_Distill
from loss import WCEDCELoss
from loss_distill import DistillationLoss
from models.reslstmunet import ResLSTMUNet
from utils import logger as make_logger

NUM_CLASS = 8


def warm_start_student(student: ResLSTMUNet, ckpt_path: str,
                       device: torch.device) -> None:
    """
    Load teacher checkpoint weights into student (strict=False).
    Asserts that only uncertainty_head.* keys are absent from the checkpoint.
    Re-initialises the uncertainty_head with Kaiming normal.
    """
    state = torch.load(ckpt_path, map_location=device)
    if all(k.startswith("module.") for k in state.keys()):
        state = {k[len("module."):]: v for k, v in state.items()}

    missing, unexpected = student.load_state_dict(state, strict=False)

    unexpected_missing = {k for k in missing if "uncertainty_head" not in k}
    if unexpected_missing:
        raise RuntimeError(
            "Unexpected missing keys when loading teacher checkpoint:\n"
            f"  {sorted(unexpected_missing)}\n"
            "This usually means the architecture flags (deep_sup, multiscale_att) "
            "do not match those used to train the teacher."
        )
    print(f"Teacher weights loaded. Missing (expected — uncertainty_head only):")
    for k in sorted(missing):
        print(f"  {k}")
    if unexpected:
        print(f"Unexpected keys in teacher ckpt (ignored): {sorted(unexpected)}")

    # Re-initialise only the uncertainty_head
    for m in student.uncertainty_head.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
    print("uncertainty_head re-initialised with Kaiming normal.")


def main():
    # ------------------------------------------------------------------ #
    # Arguments
    # ------------------------------------------------------------------ #
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["CT", "MR"], required=True)
    parser.add_argument("--teacher_ckpt", type=str, required=True,
                        help="Path to teacher best checkpoint (.pth).")
    parser.add_argument("--teacher_dir", type=str, required=True,
                        help="Root directory of pre-generated teacher targets.")
    parser.add_argument("--num_epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=2.0,
                        help="Distillation temperature.")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="KL divergence loss weight.")
    parser.add_argument("--beta", type=float, default=0.3,
                        help="Uncertainty Huber loss weight.")
    parser.add_argument("--gamma", type=float, default=0.2,
                        help="Hard-label WCEDCELoss weight.")
    args = parser.parse_args()

    # ------------------------------------------------------------------ #
    # Config
    # ------------------------------------------------------------------ #
    CFG_FILE = "train_info.yaml"
    with open(CFG_FILE, "r") as f:
        cfg = yaml.safe_load(f)

    MODE = args.mode
    dataset_cfg = cfg["WHS_datasets"][MODE.lower()]
    image_paths = dataset_cfg["train_paths"]
    val_paths   = [dataset_cfg["val_path"]]
    architecture = cfg["model_2D"][0]

    NUM_EPOCHS  = args.num_epochs
    BATCHSIZE   = args.batch_size
    NUM_WORKERS = 8
    crop_d      = 18
    model_freq  = 3

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    # ------------------------------------------------------------------ #
    # Data
    # ------------------------------------------------------------------ #
    train_set = WHSDataset_Distill(
        image_multidir=image_paths,
        teacher_dir=args.teacher_dir,
        crop_d=crop_d, augment=True,
    )
    val_set = WHSDataset_2D_scale_partSeries(
        image_multidir=val_paths, crop_d=crop_d, augment=False)

    train_loader = DataLoader(
        train_set, num_workers=NUM_WORKERS, batch_size=BATCHSIZE,
        shuffle=True, pin_memory=False)
    val_loader = DataLoader(
        val_set, num_workers=NUM_WORKERS, batch_size=BATCHSIZE,
        shuffle=False, pin_memory=False)

    # ------------------------------------------------------------------ #
    # Model
    # ------------------------------------------------------------------ #
    student = ResLSTMUNet(
        in_channels=1, out_channels=NUM_CLASS,
        pretrained=False,
        deep_sup=True,
        multiscale_att=True,
        predict_uncertainty=True,
    ).to(device)

    warm_start_student(student, args.teacher_ckpt, device)
    print(f"Student parameter count: {sum(p.numel() for p in student.parameters()):,}")

    # ------------------------------------------------------------------ #
    # Loss and optimiser
    # ------------------------------------------------------------------ #
    class_weights = torch.tensor([1., 3., 3., 3., 3., 3., 3., 3.]).to(device)

    distill_criterion = DistillationLoss(
        temperature=args.temperature,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        intra_weights=class_weights,
        device=str(device),
    )
    val_criterion = WCEDCELoss(
        intra_weights=class_weights, inter_weights=0.5)

    optimizer = optim.Adam(student.parameters(), lr=args.lr, betas=(0.9, 0.999))

    # ------------------------------------------------------------------ #
    # Output directories
    # ------------------------------------------------------------------ #
    model_save_dir = os.path.join(
        dataset_cfg["results_output"]["model_state_dict"], architecture + "_distill")
    stats_dir = os.path.join(
        dataset_cfg["results_output"]["statistics"], architecture + "_distill")
    os.makedirs(model_save_dir, exist_ok=True)
    os.makedirs(stats_dir, exist_ok=True)

    log = make_logger(os.path.join(stats_dir, f"train_distill_{architecture}.log"))
    log.info("Starting distillation training.")
    log.info(f"T={args.temperature}  alpha={args.alpha}  beta={args.beta}  gamma={args.gamma}")

    # ------------------------------------------------------------------ #
    # Training loop
    # ------------------------------------------------------------------ #
    results = {"loss": [], "val_loss": []}
    best_loss  = float("inf")
    best_epoch = 0
    best_wts   = copy.deepcopy(student.state_dict())
    since      = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_losses     = []
        epoch_val_losses = []

        # ---- train ----
        student.train()
        for it, (image_serial, label_serial,
                 soft_probs_serial, teacher_unc_serial) in enumerate(train_loader):

            image_serial       = [x.to(device) for x in image_serial]
            label_serial       = [x.to(device) for x in label_serial]
            soft_probs_serial  = [x.to(device) for x in soft_probs_serial]
            teacher_unc_serial = [x.to(device) for x in teacher_unc_serial]

            optimizer.zero_grad()

            (pred_serial, pred1_serial, pred2_serial,
             pred3_serial, pred4_serial, unc_serial) = student(image_serial)

            temporal = len(pred_serial)
            loss = 0.0
            for t in range(temporal):
                pred   = pred_serial[t]           # (B, 8, H, W)
                pred1  = pred1_serial[t]
                pred2  = pred2_serial[t]
                pred3  = pred3_serial[t]
                pred4  = pred4_serial[t]
                unc    = unc_serial[t]             # (B, 1, H, W)
                label  = label_serial[t]           # (B, 1, H, W)
                soft   = soft_probs_serial[t]      # (B, 8, H, W)
                t_unc  = teacher_unc_serial[t]     # (B, 1, H, W)

                # Full-resolution: distillation + hard-label
                loss0 = distill_criterion(
                    pred, unc, soft, t_unc, label.squeeze(1).long())

                # Deep supervision (sub-resolution): hard-label only
                loss1 = val_criterion(
                    pred1,
                    F.interpolate(label, scale_factor=0.5, mode="bilinear",
                                  align_corners=False).squeeze(1).long())
                loss2 = val_criterion(
                    pred2,
                    F.interpolate(label, scale_factor=0.25, mode="bilinear",
                                  align_corners=False).squeeze(1).long())
                loss3 = val_criterion(
                    pred3,
                    F.interpolate(label, scale_factor=0.125, mode="bilinear",
                                  align_corners=False).squeeze(1).long())
                loss4 = val_criterion(
                    pred4,
                    F.interpolate(label, scale_factor=1. / 16, mode="bilinear",
                                  align_corners=False).squeeze(1).long())

                loss += (0.4 * loss0 + 0.3 * loss1
                         + 0.2 * loss2 + 0.05 * loss3 + 0.05 * loss4)

            loss /= temporal
            loss.backward()
            optimizer.step()

            if it % max(1, len(train_loader) // 5) == 0:
                log.info(
                    f"Train: Epoch {epoch}/{NUM_EPOCHS}  "
                    f"Iter {it}/{len(train_loader)}  loss {loss.item():.4f}")
            epoch_losses.append(loss.item())

        results["loss"].append(np.mean(epoch_losses))

        # ---- validate ----
        student.eval()
        with torch.no_grad():
            for val_it, (val_img_serial, val_lbl_serial) in enumerate(val_loader):
                val_img_serial = [x.to(device) for x in val_img_serial]
                val_lbl_serial = [x.to(device) for x in val_lbl_serial]

                (val_pred_serial, val_pred1_serial, val_pred2_serial,
                 val_pred3_serial, val_pred4_serial, _) = student(val_img_serial)

                val_temporal = len(val_pred_serial)
                val_loss = 0.0
                for t in range(val_temporal):
                    vp  = val_pred_serial[t]
                    vp1 = val_pred1_serial[t]
                    vp2 = val_pred2_serial[t]
                    vp3 = val_pred3_serial[t]
                    vp4 = val_pred4_serial[t]
                    vl  = val_lbl_serial[t]

                    vl0 = val_criterion(vp,  vl.squeeze(1).long())
                    vl1 = val_criterion(
                        vp1, F.interpolate(vl, scale_factor=0.5, mode="bilinear",
                                           align_corners=False).squeeze(1).long())
                    vl2 = val_criterion(
                        vp2, F.interpolate(vl, scale_factor=0.25, mode="bilinear",
                                           align_corners=False).squeeze(1).long())
                    vl3 = val_criterion(
                        vp3, F.interpolate(vl, scale_factor=0.125, mode="bilinear",
                                           align_corners=False).squeeze(1).long())
                    vl4 = val_criterion(
                        vp4, F.interpolate(vl, scale_factor=1. / 16, mode="bilinear",
                                           align_corners=False).squeeze(1).long())

                    val_loss += (0.4 * vl0 + 0.3 * vl1
                                 + 0.2 * vl2 + 0.05 * vl3 + 0.05 * vl4)
                val_loss /= val_temporal

                if val_it % max(1, len(val_loader) // 5) == 0:
                    log.info(
                        f"Val:   Epoch {epoch}/{NUM_EPOCHS}  "
                        f"Iter {val_it}/{len(val_loader)}  val_loss {val_loss.item():.4f}")
                epoch_val_losses.append(val_loss.item())

        mean_val = np.mean(epoch_val_losses)
        results["val_loss"].append(mean_val)
        log.info(
            f"Average: Epoch {epoch}/{NUM_EPOCHS}  "
            f"train_loss {results['loss'][-1]:.4f}  val_loss {mean_val:.4f}\n")

        # Save best
        if mean_val < best_loss:
            best_loss  = mean_val
            best_epoch = epoch
            best_wts   = copy.deepcopy(student.state_dict())

        # Periodic checkpoint
        if epoch % model_freq == 0 or epoch == NUM_EPOCHS or epoch > NUM_EPOCHS - 25:
            ckpt_path = os.path.join(model_save_dir,
                                     f"distill_student_epoch_{epoch}.pth")
            torch.save(student.state_dict(), ckpt_path)

    # ------------------------------------------------------------------ #
    # Save best model and results
    # ------------------------------------------------------------------ #
    best_path = os.path.join(stats_dir,
                             f"distill_student_best_epoch_{best_epoch}.pth")
    torch.save(best_wts, best_path)
    log.info(f"Best model saved: {best_path}")

    elapsed = time.time() - since
    log.info(f"Training complete in {elapsed // 60:.0f}m {elapsed % 60:.0f}s")

    pd.DataFrame(
        {"loss": results["loss"], "val_loss": results["val_loss"]},
        index=range(1, NUM_EPOCHS + 1),
    ).to_csv(os.path.join(stats_dir, "distill_train_results.csv"),
             index_label="Epoch")


if __name__ == "__main__":
    main()
