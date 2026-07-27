"""
harmonize_ramya125g.py
======================
Harmonize the ramya125g/balanced-conjunctiva-dataset (Kaggle).

Layout (after Kaggle --unzip):
  ramya125g/balanced-conjunctiva-dataset/balanced_eye_conjunctiva/
    Anemia/*.jpg
    Non Anemia/*.jpg

Output:
  datasets/anemia_real/canonical/{anemia,normal}/ramya_<original_name>.jpg
"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path

logger = logging.getLogger("maternin.training.harmonize_ramya")

DEFAULT_RAW_DIR = Path("datasets/anemia_real/raw/ramya125g/balanced-conjunctiva-dataset/balanced_eye_conjunctiva")
DEFAULT_OUT_DIR = Path("datasets/anemia_real/canonical")


def remap(raw_dir: Path, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "anemia").mkdir(exist_ok=True)
    (out_dir / "normal").mkdir(exist_ok=True)

    summary: dict = {"anemia": 0, "normal": 0, "skipped": 0}
    valid_ext = (".jpg", ".jpeg", ".png")

    for label_dir in sorted(raw_dir.iterdir()):
        if not label_dir.is_dir():
            continue
        lname = label_dir.name.strip().lower()
        # "Anemia" -> anemia, "Non Anemia" -> normal
        if lname == "anemia":
            target_label = "anemia"
        elif "non" in lname and "anemia" in lname:
            target_label = "normal"
        else:
            logger.warning(f"Unknown label folder: {label_dir.name}")
            continue

        for img in sorted(label_dir.iterdir()):
            if not img.is_file() or img.suffix.lower() not in valid_ext:
                continue
            target = out_dir / target_label / f"ramya_{img.name}"
            if target.exists():  # avoid double-copy from re-runs
                continue
            shutil.copy2(img, target)
            summary[target_label] += 1
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
    logger.info(f"ramya125g harmonization summary: {summary}")
    print(summary)


if __name__ == "__main__":
    main()
