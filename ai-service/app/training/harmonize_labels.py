"""
harmonize_labels.py
===================
Walk raw dataset folders, map heterogeneous labels (Anemic, Yes, Normal, No, ...)
into canonical {anemia, normal}. Then split 80/10/10 into train/val/test.
"""
from __future__ import annotations

import logging
import random
import shutil
from pathlib import Path

logger = logging.getLogger("maternin.training.harmonize")

ANEMIA_SYNONYMS = {"anemia", "anemic", "yes", "1", "true", "positive", "pos"}
NORMAL_SYNONYMS = {"normal", "no", "0", "false", "negative", "neg", "healthy"}

LABEL_RULES = {
    "anemia": sorted(ANEMIA_SYNONYMS),
    "normal": sorted(NORMAL_SYNONYMS),
}


def is_anemia(label: str) -> bool:
    return label.strip().lower() in ANEMIA_SYNONYMS


def harmonize_label(raw_label: str) -> str:
    low = raw_label.strip().lower()
    if low in ANEMIA_SYNONYMS:
        return "anemia"
    if low in NORMAL_SYNONYMS:
        return "normal"
    raise ValueError(f"Unknown label: {raw_label!r}")


def remap_tree(src_root: str, dst_root: str) -> dict:
    """Walk src_root/<label>/<files>, copy to dst_root/<canonical_label>/<files>."""
    src = Path(src_root)
    dst = Path(dst_root)
    dst.mkdir(parents=True, exist_ok=True)
    mapping: dict = {}
    for label_dir in src.iterdir():
        if not label_dir.is_dir():
            continue
        for img in label_dir.iterdir():
            if not img.is_file():
                continue
            canonical = harmonize_label(label_dir.name)
            target_dir = dst / canonical
            target_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(img, target_dir / img.name)
            mapping[img.name] = canonical
    return mapping


def write_label_mapping_yaml(rules: dict, path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        f.write("# LABEL_MAPPING.yaml — auto-generated\n")
        f.write("# Canonical labels: anemia, normal\n\n")
        for canonical, synonyms in sorted(rules.items()):
            f.write(f"{canonical}:\n")
            for syn in sorted(synonyms):
                f.write(f"  - {syn}\n")


def split_train_val_test(
    src_root: str,
    dst_root: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> dict:
    """Split canonical dataset into train/val/test under dst_root/{split}/{class}/."""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    src = Path(src_root)
    dst = Path(dst_root)

    rng = random.Random(seed)
    counts: dict = {"train": {}, "val": {}, "test": {}}

    for cls in ("anemia", "normal"):
        files = sorted((src / cls).iterdir())
        rng.shuffle(files)
        n = len(files)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        train_files = files[:n_train]
        val_files = files[n_train : n_train + n_val]
        test_files = files[n_train + n_val :]

        for split, group in (("train", train_files), ("val", val_files), ("test", test_files)):
            split_dir = dst / split / cls
            split_dir.mkdir(parents=True, exist_ok=True)
            for f in group:
                shutil.copy2(f, split_dir / f.name)
            counts[split][cls] = len(group)

    return counts


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="datasets/anemia_real/raw")
    parser.add_argument("--canonical-dir", default="datasets/anemia_real/canonical")
    parser.add_argument("--split-dir", default="datasets/anemia_real")
    args = parser.parse_args()

    raw_root = Path(args.raw_dir)
    sources_md = ["# DATASET_SOURCES", "", "| File | Source |", "|------|--------|"]
    for source_dir in raw_root.iterdir():
        if not source_dir.is_dir() or source_dir.name in ("canonical", "MANIFEST.json"):
            continue
        target_inbox = Path(args.canonical_dir)
        target_inbox.mkdir(parents=True, exist_ok=True)
        mapping = remap_tree(str(source_dir), str(target_inbox))
        for fname, cls in mapping.items():
            sources_md.append(f"| {cls}/{fname} | {source_dir.name} |")

    Path(f"{args.canonical_dir}/DATASET_SOURCES.md").write_text("\n".join(sources_md))
    write_label_mapping_yaml(LABEL_RULES, f"{args.canonical_dir}/LABEL_MAPPING.yaml")

    counts = split_train_val_test(args.canonical_dir, args.split_dir)
    print(f"Split complete: {counts}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
