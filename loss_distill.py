"""
loss_distill.py

DistillationLoss for uncertainty-aware knowledge distillation.

Combines three terms:
  1. KL divergence   -- soft prediction distillation at temperature T.
  2. Huber loss      -- uncertainty map regression in log1p space.
  3. WCEDCELoss      -- hard-label segmentation quality (same as train.py).

Total = alpha * L_KL + beta * L_Huber + gamma * L_hard
      (alpha + beta + gamma must equal 1.0)

KL notes:
  - Teacher soft probabilities are temperature-scaled by raising each class
    probability to 1/T and renormalising.  This sharpens (T<1) or softens
    (T>1) the teacher distribution before computing KL.
  - Student log-softmax is computed at temperature T.
  - The T^2 correction (Hinton et al. 2015) compensates for the reduced
    gradient magnitude caused by dividing logits by T.

Uncertainty notes:
  - Teacher uncertainty is the raw variance sum across 8 softmax classes,
    typically in [0, ~1.5].
  - log1p normalisation maps this to [0, ~0.92], giving a well-conditioned
    regression target.
  - The student's Softplus output is inherently positive and covers the
    same range, so no further rescaling is needed.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from loss import WCEDCELoss


class DistillationLoss(nn.Module):
    def __init__(
        self,
        temperature: float = 2.0,
        alpha: float = 0.5,    # KL divergence weight
        beta: float = 0.3,     # Huber loss weight
        gamma: float = 0.2,    # hard-label WCEDCELoss weight
        huber_delta: float = 1.0,
        intra_weights=None,    # per-class weights for WCEDCELoss
        device: str = "cuda",
    ):
        super().__init__()
        assert abs(alpha + beta + gamma - 1.0) < 1e-5, (
            f"alpha + beta + gamma must equal 1.0, got {alpha + beta + gamma:.4f}"
        )
        self.T     = temperature
        self.alpha = alpha
        self.beta  = beta
        self.gamma = gamma

        self.kl_loss = nn.KLDivLoss(reduction="batchmean")
        self.huber   = nn.SmoothL1Loss(reduction="mean", beta=huber_delta)
        self.hard_criterion = WCEDCELoss(
            intra_weights=intra_weights,
            inter_weights=0.5,
            device=device,
        )

    def forward(
        self,
        student_logits: torch.Tensor,   # (B, 8, H, W)
        student_unc: torch.Tensor,      # (B, 1, H, W)  Softplus output
        teacher_soft: torch.Tensor,     # (B, 8, H, W)  soft probs (sum to 1)
        teacher_unc: torch.Tensor,      # (B, 1, H, W)  raw variance
        hard_label: torch.Tensor,       # (B, H, W)     long class indices
    ) -> torch.Tensor:
        """Returns scalar distillation loss."""

        # ---- 1. KL divergence (soft prediction distillation) ----
        # Temperature-scale the teacher by raising probs to 1/T and renormalising.
        # This avoids log(0) since teacher_soft is already a valid distribution.
        T = self.T
        teacher_T = teacher_soft ** (1.0 / T)
        teacher_T = teacher_T / teacher_T.sum(dim=1, keepdim=True)

        student_log = F.log_softmax(student_logits / T, dim=1)
        kl = self.kl_loss(student_log, teacher_T) * (T ** 2)

        # ---- 2. Huber loss (uncertainty map regression) ----
        teacher_unc_log = torch.log1p(teacher_unc)   # normalise to ~[0, 0.9]
        huber = self.huber(student_unc, teacher_unc_log)

        # ---- 3. Hard-label segmentation loss ----
        hard = self.hard_criterion(student_logits, hard_label)

        return self.alpha * kl + self.beta * huber + self.gamma * hard
