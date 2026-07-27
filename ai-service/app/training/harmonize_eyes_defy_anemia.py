"""
harmonize_eyes_defy_anemia.py
=============================
EYES-DEFY-ANEMIA specific harmonizer.

Dataset layout (after Kaggle --unzip):
  dataset anemia/{Country}/{patient_id}/{timestamp}.jpg   <- the actual conjunctiva photo
  dataset anemia/{Country}/{Country}.xlsx                  <- labels: Number, Hgb, Gender, Age
  dataset anemia/{Country}/{patient_id}/..._forniceal.png  (mask - skip)
  dataset anemia/{Country}/{patient_id}/..._palpebral.png  (mask - skip)
  dataset anemia/{Country}/{patient_id}/..._forniceal_palpebral.png  (mask - skip)

WHO anemia thresholds (for conjunctival pallor screening):
  Female: Hgb < 12 g/dL
  Male:   Hgb < 13 g/dL

Output:
  datasets/anemia_real/canonical/{anemia,normal}/*.jpg
"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path

import openpyxl

logger = logging.getLogger("maternin.training.harmonize_eda")

DEFAULT_ROOT = Path("datasets/anemia_real/raw/harshwardhanfartale/eyes-defy-anemia/dataset anemia")
DEFAULT_OUTPUT = Path("datasets/anemia_real/canonical")

WHO_FEMALE_THRESHOLD = 12.0  # g/dL
WHO_MALE_THRESHOLD = 13.0    # g/dL


def _threshold_for(gender: str) -> float:
    if isinstance(gender, str):
        return WHO_FEMALE_THRESHOLD if gender.upper().startswith("F") else WHO_MALE_THRESHOLD
    return WHO_FEMALE_THRESHOLD


def load_labels_per_country(root: Path) -> dict[int, str]:
    """Build {patient_number: 'anemia'|'normal'} from each Country.xlsx."""
    labels: dict[int, str] = {}
    for country_dir in sorted(root.iterdir()):
        if not country_dir.is_dir():
            continue
        xlsx = country_dir / f"{country_dir.name}.xlsx"
        if not xlsx.exists():
            logger.warning(f"No xlsx in {country_dir}, skipping")
            continue
        wb = openpyxl.load_workbook(xlsx, data_only=True)
        ws = wb.active
        rows = list(ws.iter_rows(values_only=True))
        # Header row 0 is (Number, Hgb, Gender, Age, Note, ...)
        for r in rows[1:]:
            if not r or r[0] is None:
                continue
            try:
                num = int(r[0])
                hgb = float(r[1])
            except (ValueError, TypeError):
                continue
            threshold = _threshold_for(r[2])
            labels[num] = "anemia" if hgb < threshold else "normal"
        logger.info(f"{country_dir.name}: {len(labels)} patients loaded so far")
    return labels


def remap(root: Path, labels: dict[int, str], out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    (out / "anemia").mkdir(exist_ok=True)
    (out / "normal").mkdir(exist_ok=True)

    summary: dict = {"anemia": 0, "normal": 0, "skipped": 0}
    prov: list[tuple[str, str]] = []  # (filename, source_path)
    for country_dir in sorted(root.iterdir()):
        if not country_dir.is_dir():
            continue
        for patient_dir in sorted(country_dir.iterdir()):
            if not patient_dir.is_dir() or not patient_dir.name.isdigit():
                continue
            try:
                patient_id = int(patient_dir.name)
            except ValueError:
                continue
            label = labels.get(patient_id)
            if label is None:
                summary["skipped"] += 1
                continue
            # Take only the .jpg (not the _*.png masks)
            for img in patient_dir.iterdir():
                if img.suffix.lower() == ".jpg" and img.is_file():
                    target = out / label / f"{country_dir.name}_{patient_id}_{img.name}"
                    shutil.copy2(img, target)
                    prov.append((target.name, str(img)))
                    summary[label] += 1
                    break  # one jpg per patient
    return {"summary": summary, "provenance": prov}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", default=str(DEFAULT_ROOT))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    root = Path(args.raw_root)
    out = Path(args.out_dir)

    if not root.exists():
        raise SystemExit(f"Raw dataset not found: {root}")

    labels = load_labels_per_country(root)
    logger.info(f"Loaded labels for {len(labels)} patients total")

    out.mkdir(parents=True, exist_ok=True)
    result = remap(root, labels, out)
    logger.info(f"Harmonization summary: {result['summary']}")


if __name__ == "__main__":
    main()
