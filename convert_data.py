import re
import numpy as np
from pathlib import Path

DATA_ROOT = Path("/Users/erinsarlak/Desktop/MedicalImagingProject/data")
SRC_ROOT  = DATA_ROOT / "processed_data"
DST_ROOT  = DATA_ROOT / "converted"

MODALITIES = ["ct", "mr"]
SPLITS     = ["train", "val", "test"]


def convert():
    total = 0
    for modality in MODALITIES:
        for split in SPLITS:
            src_dir = SRC_ROOT / f"{modality}_256" / split / "npz"
            if not src_dir.exists():
                print(f"Skipping {src_dir} (not found)")
                continue

            npz_files = sorted(src_dir.glob("*.npz"))
            print(f"Processing {modality}/{split}: {len(npz_files)} files ...")

            for npz_path in npz_files:
                # filename pattern: {modality}_{patient_id}_slice_{slice_num}.npz
                match = re.match(r".+_(\d+)_slice_(\d+)\.npz", npz_path.name)
                if not match:
                    print(f"  Skipping unrecognized filename: {npz_path.name}")
                    continue

                patient_id = match.group(1)
                slice_num  = int(match.group(2))

                patient_dir = DST_ROOT / modality / split / patient_id
                patient_dir.mkdir(parents=True, exist_ok=True)

                arrays = np.load(npz_path)
                # Images are already normalized to [0, 1] — save as float32
                np.save(patient_dir / f"image_{slice_num:04d}.npy", arrays["image"].astype("float32"))
                np.save(patient_dir / f"label_{slice_num:04d}.npy", arrays["label"].astype("float32"))
                total += 1

    print(f"\nDone. Converted {total} files → {DST_ROOT}")


if __name__ == "__main__":
    convert()
