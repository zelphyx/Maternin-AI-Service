"""
harmonize_guptajanavi.py
========================
Guptajanavi palpebral conjunctiva dataset harmonizer.

Layout (after Kaggle --unzip):
  guptajanavi/palpebral-conjunctiva-to-detect-anaemia/
    img_1_<index>.jpg   -> anaemic (class 1)
    img_2_<index>.jpg   -> non-anaemic (class 2)
    anaemicvsnonanaemic.h5  -> Keras dump (skip)

Output:
  datasets/anemia_real/canonical/{anemia,normal}/guptajanavi_<original_name>.jpg
"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path

logger = logging.getLogger("maternin.training.harmonize_guptajanavi")

DEFAULT_RAW_DIR = Path("datasets/anemia_real/raw/guptajanavi/palpebral-conjunctiva-to-detect-anaemia")
DEFAULT_OUT_DIR = Path("datasets/anemia_real/canonical")


def remap(raw_dir: Path, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "anemia").mkdir(exist_ok=True)
    (out_dir / "normal").mkdir(exist_ok=True)

    summary: dict = {"anemia": 0, "normal": 0}

    for img in sorted(raw_dir.iterdir()):
        if not img.is_file() or img.suffix.lower() != ".jpg":
            continue
        # Pattern: img_<class>_<idx>.jpg
        # class 1 = anaemic, class 2 = non-anaemic (per Kaggle description)
        parts = img.stem.split("_")
        if len(parts) < 3 or parts[0] != "img":
            continue
        try:
            cls_id = int(parts[1])
        except ValueError:
            continue
        if cls_id == 1:
            label = "anemia"
        elif cls_id == 2:
            label = "normal"
        else:
            continue
        target = out_dir / label / f"guptajanavi_{img.name}"
        shutil.copy2(img, target)
        summary[label] += 1

    return summary


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default=str(DEFAULT_RAW_DIR))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)

    if not raw_dir.exists():
        raise SystemExit(f"Raw dataset not found: {raw_dir}")

    summary = remap(raw_dir, out_dir)
    logger.info(f"guptajanavi harmonization summary: {summary}")


if __name__ == "__main__":
    main()