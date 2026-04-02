"""
dataset_distill.py

WHSDataset_Distill extends WHSDataset_2D_scale_partSeries to also load
pre-generated teacher soft predictions and uncertainty maps.

Returns a 4-tuple per sample:
    (image_serial, label_serial, soft_probs_serial, unc_serial)

where each is a list of T tensors with shapes:
    image:      (1, 224, 224)  float32
    label:      (1, 224, 224)  float32 (class indices)
    soft_probs: (8, 224, 224)  float32 (summing to 1 over dim 0)
    unc:        (1, 224, 224)  float32 (non-negative)

Augmentation contract:
    The same affine parameters (angle, translate, scale, shear) are applied to
    all four modalities identically, preserving spatial correspondence:
      image      -- BILINEAR
      label      -- NEAREST
      soft_probs -- BILINEAR
      unc        -- BILINEAR, then clamped to >= 0

Teacher target files are expected at:
    teacher_dir/<patient_name>/soft_probs<NNNN>.npy  (8, 224, 224)
    teacher_dir/<patient_name>/uncertainty<NNNN>.npy (1, 224, 224)

where <NNNN> = last four characters before .npy in the image filename,
e.g. image0042.npy -> soft_probs0042.npy, uncertainty0042.npy.
"""

import os
import random

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from dataset import WHSDataset_2D_scale_partSeries


class WHSDataset_Distill(WHSDataset_2D_scale_partSeries):
    def __init__(self, image_multidir: list, teacher_dir: str,
                 crop_d: int = 32, stride: int = 5, augment: bool = False):
        """
        Args:
            image_multidir: list of patient root directories (same as parent).
            teacher_dir:    root of teacher targets; contains subdirectories
                            named by patient_name matching those in image_multidir.
            crop_d:         sequence length (default 32).
            stride:         sliding-window stride (default 5).
            augment:        apply affine augmentation (default False).
        """
        super().__init__(image_multidir, crop_d, stride, augment)
        self.teacher_dir = teacher_dir

    # ------------------------------------------------------------------
    # Internal helper
    # ------------------------------------------------------------------

    def _get_teacher_paths(self, image_path: str):
        """
        Derive teacher file paths from an image slice path.

        image_path: .../patient_dir/image0042.npy
        Returns:
            (soft_probs_path, unc_path)
        """
        base = os.path.basename(image_path)   # image0042.npy
        slice_num = base[-8:-4]               # '0042'
        patient_name = os.path.basename(os.path.dirname(image_path))
        soft_path = os.path.join(self.teacher_dir, patient_name,
                                 f"soft_probs{slice_num}.npy")
        unc_path  = os.path.join(self.teacher_dir, patient_name,
                                 f"uncertainty{slice_num}.npy")
        return soft_path, unc_path

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __getitem__(self, index):
        image_serial      = []
        label_serial      = []
        soft_probs_serial = []
        unc_serial        = []

        # Sample augmentation parameters once per sequence (same as parent)
        if self.augment:
            angle     = random.uniform(-15, 15)
            translate = [random.uniform(-0.1, 0.1) * 224,
                         random.uniform(-0.1, 0.1) * 224]
            scale     = random.uniform(1 / 1.2, 1.2)
            shear     = random.uniform(-3, 3)

        for image_path in self.image_paths[index]:
            # ---- image ----
            image = np.load(image_path)
            image = torch.from_numpy(image.astype("float32")).unsqueeze(0).unsqueeze(0)
            image = F.interpolate(image, [224, 224], mode="bilinear",
                                  align_corners=False).squeeze(0)

            # ---- label ----
            label_path = os.path.join(
                os.path.dirname(image_path),
                os.path.basename(image_path).replace("image", "label"),
            )
            label = np.load(label_path)
            label = torch.from_numpy(label.astype("float32")).unsqueeze(0).unsqueeze(0)
            label = F.interpolate(label, [224, 224]).squeeze(0)

            # ---- teacher targets ----
            sp_path, unc_path = self._get_teacher_paths(image_path)
            soft_probs = torch.from_numpy(
                np.load(sp_path).astype("float32"))   # (8, 224, 224)
            unc = torch.from_numpy(
                np.load(unc_path).astype("float32"))  # (1, 224, 224)

            # ---- augmentation (identical params for all four modalities) ----
            if self.augment:
                image = TF.affine(
                    image, angle=angle, translate=translate, scale=scale, shear=shear,
                    interpolation=TF.InterpolationMode.BILINEAR)
                label = TF.affine(
                    label, angle=angle, translate=translate, scale=scale, shear=shear,
                    interpolation=TF.InterpolationMode.NEAREST)
                soft_probs = TF.affine(
                    soft_probs, angle=angle, translate=translate, scale=scale, shear=shear,
                    interpolation=TF.InterpolationMode.BILINEAR)
                unc = TF.affine(
                    unc, angle=angle, translate=translate, scale=scale, shear=shear,
                    interpolation=TF.InterpolationMode.BILINEAR)
                # Clamp uncertainty: bilinear interpolation may introduce tiny negatives
                unc = torch.clamp(unc, min=0.0)

            image_serial.append(image)
            label_serial.append(label)
            soft_probs_serial.append(soft_probs)
            unc_serial.append(unc)

        return image_serial, label_serial, soft_probs_serial, unc_serial
