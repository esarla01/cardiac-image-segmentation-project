"""
Uncertainty quantification for ResLSTMUNet via Checkpoint Ensemble and TTA.

Checkpoint Ensemble:
  - Train the model and save checkpoints at multiple epochs (e.g. every 3 epochs).
  - At inference, load N checkpoints and average their softmax probabilities.
  - Uncertainty = variance across checkpoints (captures model uncertainty from
    different points in the loss landscape without requiring dropout).

Test-Time Augmentation (TTA):
  - Model stays in eval() mode.
  - TTA transforms are restricted to the affine family (rotation, translation,
    scale) to stay within the training distribution. Flips and 90°/270°
    rotations are deliberately excluded: the model was trained without them and
    cardiac anatomy has a fixed orientation (left != right, apex points down),
    so those transforms would produce out-of-distribution inputs.
  - Each transform is inverted on the output before averaging, so all
    predictions are back in the original image space.
  - Returns mean probabilities + variance across augmentations.

Combined Ensemble + TTA:
  - For each checkpoint, apply all TTA transforms and collect predictions.
  - Average across all (checkpoint, transform) combinations.
  - Uncertainty = variance across the full set of predictions.
"""

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF


# ---------------------------------------------------------------------------
# TTA transform definitions
# Requires torchvision >= 0.11 (supports batched tensors in TF.affine).
# Each entry: (forward_fn, inverse_fn) on (..., H, W) tensors.
# Parameters stay within the training distribution:
#   rotation +-20 deg, translation +-22 px (10% of 224), scale 0.83-1.2x.
# ---------------------------------------------------------------------------

def _aff(x, angle=0.0, translate=(0, 0), scale=1.0):
    return TF.affine(x, angle=angle, translate=list(translate), scale=scale,
                     shear=0, interpolation=TF.InterpolationMode.BILINEAR)

TTA_TRANSFORMS = [
    # (forward_fn, inverse_fn) -- each pair is exactly invertible
    (lambda x: _aff(x, angle= 10),           lambda x: _aff(x, angle=-10)),
    (lambda x: _aff(x, angle=-10),           lambda x: _aff(x, angle= 10)),
    (lambda x: _aff(x, angle= 15),           lambda x: _aff(x, angle=-15)),
    (lambda x: _aff(x, angle=-15),           lambda x: _aff(x, angle= 15)),
    (lambda x: _aff(x, scale=1.10),          lambda x: _aff(x, scale=1/1.10)),
    (lambda x: _aff(x, scale=0.90),          lambda x: _aff(x, scale=1/0.90)),
    (lambda x: _aff(x, translate=( 15, 0)),  lambda x: _aff(x, translate=(-15,  0))),
    (lambda x: _aff(x, translate=(-15, 0)),  lambda x: _aff(x, translate=( 15,  0))),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _stack_temporal(samples):
    """
    samples: list of N passes, each a list of T tensors (B, C, H, W).
    Returns: list of T tensors (N, B, C, H, W).
    """
    T = len(samples[0])
    return [
        torch.stack([samples[s][t] for s in range(len(samples))])
        for t in range(T)
    ]


# ---------------------------------------------------------------------------
# Test-Time Augmentation
# ---------------------------------------------------------------------------

def tta_predict(model, x_serial, n_transforms: int = None):
    """
    Run inference with affine TTA on a single model.

    Args:
        model:         ResLSTMUNet in eval() mode.
        x_serial:      List of T tensors (B, 1, H, W).
        n_transforms:  How many TTA transforms to use (default: all 8).
                       Identity is always included as the first pass.

    Returns:
        mean_probs:  List of T tensors (B, C, H, W).
        uncertainty: List of T tensors (B, H, W) -- variance across augmentations.
    """
    model.eval()
    transforms = [(lambda x: x, lambda x: x)] + (
        TTA_TRANSFORMS[:n_transforms] if n_transforms else TTA_TRANSFORMS
    )
    all_samples = []

    with torch.no_grad():
        for fwd, inv in transforms:
            aug_serial = [fwd(x) for x in x_serial]
            pred_serial, *_ = model(aug_serial)
            probs = [inv(torch.softmax(p, dim=1)) for p in pred_serial]
            all_samples.append(probs)

    mean_probs, uncertainty_list = [], []

    for stacked in _stack_temporal(all_samples):   # (A, B, C, H, W)
        mean_p = stacked.mean(dim=0)
        unc    = stacked.var(dim=0).sum(dim=1)
        mean_probs.append(mean_p)
        uncertainty_list.append(unc)

    return mean_probs, uncertainty_list


# ---------------------------------------------------------------------------
# Checkpoint Ensemble
# ---------------------------------------------------------------------------

def ensemble_predict(models, x_serial):
    """
    Average softmax probabilities across multiple checkpoints.

    Args:
        models:    List of ResLSTMUNet models (each loaded from a different
                   checkpoint and placed in eval() mode).
        x_serial:  List of T tensors (B, 1, H, W).

    Returns:
        mean_probs:  List of T tensors (B, C, H, W) -- mean softmax probs.
        uncertainty: List of T tensors (B, H, W)    -- variance across checkpoints.
    """
    all_samples = []

    with torch.no_grad():
        for model in models:
            model.eval()
            pred_serial, *_ = model(x_serial)
            probs = [torch.softmax(p, dim=1) for p in pred_serial]
            all_samples.append(probs)

    mean_probs, uncertainty_list = [], []

    for stacked in _stack_temporal(all_samples):   # (M, B, C, H, W)
        mean_p = stacked.mean(dim=0)
        unc    = stacked.var(dim=0).sum(dim=1)
        mean_probs.append(mean_p)
        uncertainty_list.append(unc)

    return mean_probs, uncertainty_list


# ---------------------------------------------------------------------------
# Combined Checkpoint Ensemble + TTA
# ---------------------------------------------------------------------------

def ensemble_tta_predict(models, x_serial, n_transforms: int = None):
    """
    Combine checkpoint ensemble with TTA.

    For each checkpoint, all TTA transforms are applied and predictions are
    collected. The final mean and uncertainty are computed across all
    (checkpoint × transform) combinations.

    Args:
        models:        List of ResLSTMUNet models in eval() mode.
        x_serial:      List of T tensors (B, 1, H, W).
        n_transforms:  TTA transforms to use per checkpoint (default: all 8).
                       Identity is always included.

    Returns:
        mean_probs:  List of T tensors (B, C, H, W).
        uncertainty: List of T tensors (B, H, W) -- variance across all combinations.
    """
    transforms = [(lambda x: x, lambda x: x)] + (
        TTA_TRANSFORMS[:n_transforms] if n_transforms else TTA_TRANSFORMS
    )
    all_samples = []

    with torch.no_grad():
        for model in models:
            model.eval()
            for fwd, inv in transforms:
                aug_serial = [fwd(x) for x in x_serial]
                pred_serial, *_ = model(aug_serial)
                probs = [inv(torch.softmax(p, dim=1)) for p in pred_serial]
                all_samples.append(probs)

    mean_probs, uncertainty_list = [], []

    for stacked in _stack_temporal(all_samples):   # (M*A, B, C, H, W)
        mean_p = stacked.mean(dim=0)
        unc    = stacked.var(dim=0).sum(dim=1)
        mean_probs.append(mean_p)
        uncertainty_list.append(unc)

    return mean_probs, uncertainty_list
