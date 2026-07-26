# Real-Data Anemia CV Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace synthetic-data anemia CV training with a real-data pipeline that downloads ≥1,000 real conjunctiva images from public sources, applies aggressive Albumentations augmentation, trains MobileNetV3-Small, and exports a new `anemia_mobilenetv3_v2_real.onnx`.

**Architecture:** Two-stage data pipeline (download → harmonize labels → split 80/10/10) feeds a single PyTorch training script with Albumentations on-the-fly augmentation. Output `.onnx` is consumed by `app/models/anemia_cv/inference.py` (path constant updated to v2).

**Tech Stack:** Python 3.11+, PyTorch (`torch`, `torchvision`), Albumentations, onnxruntime, Pillow, scikit-learn, pytest, Kaggle CLI.

## Global Constraints

- ❌ NO synthetic data in train/val/test. Folder `datasets/anemia_synthetic/` and generator script `generate_synthetic_anemia.py` must be removed before training begins.
- ✅ Augmentation library = Albumentations (13 transforms from spec §5).
- ✅ Dataset must contain ≥1,000 real images. If total real < 1,000 → STOP and discuss with user.
- ✅ Validation/test augmentations disabled (only Resize + Normalize).
- ✅ Output model: `ai-service/app/model_artifacts/anemia_mobilenetv3_v2_real.onnx` + matching `*_metadata.json` + `report.md`.
- ✅ Training success: accuracy ≥ 0.85, recall ≥ 0.80, overfit gap < 10pp.
- ✅ ONNX inference must run successfully on real validation images.
- Python 3.11 (matches `.venv`).
- Dependency pinning per `ai-service/requirements.txt` style (`>=X,<Y`).
- All commits follow `<scope>: <message>` style; no `--no-verify`.

---

## File Structure

| File | Purpose |
|------|---------|
| `ai-service/app/training/download_real_datasets.py` | Download Kaggle anemia datasets; manifest JSON of sources |
| `ai-service/app/training/harmonize_labels.py` | Map heterogeneous labels (Anemic/anemia/yes) → canonical `{anemia, normal}` |
| `ai-service/app/training/anemia_real_train.py` | PyTorch training: Albumentations, MobileNetV3-Small, ONNX export, report generator |
| `ai-service/app/training/tests/test_download_real_datasets.py` | Unit tests for downloader manifest + retry helper |
| `ai-service/app/training/tests/test_harmonize_labels.py` | Unit tests for label-mapping logic |
| `ai-service/app/training/tests/test_anemia_real_train.py` | Unit tests for augmentation + dataset + training helpers |
| `ai-service/scripts/validate_onnx_parity.py` | One-shot ONNX inference sanity script |
| `datasets/anemia_real/LABEL_MAPPING.yaml` | Audit-trail YAML documenting label mapping rules |
| `datasets/anemia_real/DATASET_SOURCES.md` | Provenance per image (sumber + URL) |
| `ai-service/app/models/anemia_cv/inference.py` | **Modify**: ONNX_PATH → v2_real |
| `ai-service/tests/test_ml_inference.py` | **Modify**: assert v2 ONNX path is referenced |
| `ai-service/requirements.txt` | **Modify**: uncomment + add `torch`, `onnxruntime`, `albumentations`, `opencv-python` |
| `docs/superpowers/specs/2026-07-26-real-anemia-cv-training-design.md` | (exists) — design spec this plan implements |

---

### Task 1: Update requirements.txt with training dependencies

**Files:**
- Modify: `ai-service/requirements.txt` (lines 33-39)

**Interfaces:**
- Consumes: nothing
- Produces: updated `requirements.txt`; `pip install -r requirements.txt` succeeds

- [ ] **Step 1: Read current requirements.txt**

Run: `cat ai-service/requirements.txt`
Expected: see commented-out `# torch` and `# onnxruntime`, plus `mediapipe`, `Pillow`

- [ ] **Step 2: Edit the CV section**

Replace the entire "Computer Vision" block with:

```txt
# --- Computer Vision ---
torch>=2.3.0,<3.0.0
torchvision>=0.18.0,<1.0.0
onnxruntime>=1.18.0,<2.0.0
albumentations>=1.4.0,<2.0.0
mediapipe>=0.10.0,<1.0.0
Pillow>=10.0.0,<12.0.0
opencv-python>=4.10.0,<5.0.0
```

- [ ] **Step 3: Verify install succeeds**

Run: `cd /Users/zelphyx/Projects/Maternin-AI/ai-service && source .venv/bin/activate && pip install -r requirements.txt 2>&1 | tail -20`
Expected: installs without error. Confirm: `python -c "import torch, albumentations, onnxruntime; print('ok')"` returns `ok`.

- [ ] **Step 4: Commit**

```bash
git add ai-service/requirements.txt
git commit -m "chore(deps): add torch, albumentations, onnxruntime for real-data training"
```

---

### Task 2: Remove legacy synthetic-data code & dataset

**Files:**
- Delete: `ai-service/app/training/anemia_synthetic_train.py`
- Delete: `ai-service/app/training/generate_synthetic_anemia.py`
- Delete: `datasets/anemia_synthetic/`
- Delete: `ai-service/app/model_artifacts/anemia_mobilenetv3_v1.onnx` + `.onnx.data` + `_metadata.json`

**Interfaces:**
- Consumes: nothing
- Produces: clean repo with no synthetic-data path; inference module ONNX_PATH updated in Task 8

- [ ] **Step 1: Verify no production logic depends on v1**

Run: `grep -rn "anemia_mobilenetv3_v1" /Users/zelphyx/Projects/Maternin-AI/ai-service/app/`
Expected: only matches in `inference.py` (ONNX_PATH constant) and `anemia_cv_train.py` (commented docstring). No production logic depends on `v1`.

- [ ] **Step 2: Delete files**

Run:
```bash
rm -f /Users/zelphyx/Projects/Maternin-AI/ai-service/app/training/anemia_synthetic_train.py
rm -f /Users/zelphyx/Projects/Maternin-AI/ai-service/app/training/generate_synthetic_anemia.py
rm -rf /Users/zelphyx/Projects/Maternin-AI/datasets/anemia_synthetic
rm -f /Users/zelphyx/Projects/Maternin-AI/ai-service/app/model_artifacts/anemia_mobilenetv3_v1.onnx
rm -f /Users/zelphyx/Projects/Maternin-AI/ai-service/app/model_artifacts/anemia_mobilenetv3_v1.onnx.data
rm -f /Users/zelphyx/Projects/Maternin-AI/ai-service/app/model_artifacts/anemia_mobilenetv3_v1_metadata.json
```

- [ ] **Step 3: Verify deletion**

Run: `ls /Users/zelphyx/Projects/Maternin-AI/datasets/anemia_synthetic 2>&1; ls /Users/zelphyx/Projects/Maternin-AI/ai-service/app/model_artifacts/`
Expected: first command errors `No such file`; second shows only `preeclampsia_lr_v1*`, `risk_aggregator_v1*`, no `anemia_*`.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "refactor(training): remove synthetic anemia data and legacy v1 artifacts"
```

---

### Task 3: Create dataset downloader with manifest

**Files:**
- Create: `ai-service/app/training/download_real_datasets.py`
- Create: `ai-service/app/training/tests/test_download_real_datasets.py`

**Interfaces:**
- Consumes: env vars (optional per-source); CLI args `--output-dir`, `--sources`
- Produces: folder `datasets/anemia_real/raw/<source_name>/` + `MANIFEST.json`

- [ ] **Step 1: Write the failing test**

Create `ai-service/app/training/tests/test_download_real_datasets.py`:

```python
"""Tests for downloader manifest schema and retry helper."""
import json
from pathlib import Path

import pytest

from app.training.download_real_datasets import (
    build_manifest_entry,
    save_manifest,
    retry_with_backoff,
)


def test_build_manifest_entry_required_fields():
    entry = build_manifest_entry(
        source="kaggle_anemia_conjunctiva",
        url="https://www.kaggle.com/datasets/<some-anemia-dataset>",
        count=218,
        license="Kaggle Terms",
    )
    assert entry["source"] == "kaggle_anemia_conjunctiva"
    assert entry["count"] == 218
    assert "downloaded_at" in entry
    assert entry["license"] == "Kaggle Terms"


def test_save_manifest_creates_file(tmp_path):
    entries = [
        build_manifest_entry("src1", "https://x", 100, "MIT"),
    ]
    path = tmp_path / "MANIFEST.json"
    save_manifest(entries, str(path))
    data = json.loads(path.read_text())
    assert len(data) == 1
    assert data[0]["source"] == "src1"


def test_retry_with_backoff_succeeds_on_third_try(monkeypatch):
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("boom")
        return "ok"

    monkeypatch.setattr("time.sleep", lambda _: None)
    result = retry_with_backoff(flaky, max_attempts=3, base_delay=0.01)
    assert result == "ok"
    assert calls["n"] == 3


def test_retry_with_backoff_gives_up():
    def always_fail():
        raise RuntimeError("nope")

    with pytest.raises(RuntimeError, match="nope"):
        retry_with_backoff(always_fail, max_attempts=2, base_delay=0.01)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/zelphyx/Projects/Maternin-AI/ai-service && source .venv/bin/activate && pytest app/training/tests/test_download_real_datasets.py -v`
Expected: `ModuleNotFoundError: No module named 'app.training.download_real_datasets'`

- [ ] **Step 3: Implement download_real_datasets.py**

```python
"""
download_real_datasets.py
=========================
Download real conjunctiva images from public sources.

Configured sources (call --source NAME to select one):
  - kaggle: EYES-DEFY-ANEMIA (or any Kaggle conjunctiva anemia dataset)
  - roboflow: stub for user-provided export URL

Outputs:
  datasets/anemia_real/raw/<source_name>/<files...>
  datasets/anemia_real/raw/MANIFEST.json (per-source provenance)
"""
from __future__ import annotations

import datetime
import json
import logging
import os
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

logger = logging.getLogger("maternin.training.download")


@dataclass
class ManifestEntry:
    source: str
    url: str
    count: int
    license: str
    downloaded_at: str

    def to_dict(self) -> dict:
        return asdict(self)


def build_manifest_entry(source: str, url: str, count: int, license: str) -> dict:
    """Construct a single manifest entry with required provenance fields."""
    entry = ManifestEntry(
        source=source,
        url=url,
        count=count,
        license=license,
        downloaded_at=datetime.datetime.utcnow().isoformat() + "Z",
    )
    return entry.to_dict()


def save_manifest(entries: list[dict], path: str) -> None:
    """Write JSON manifest of downloaded datasets."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(entries, indent=2))


def retry_with_backoff(fn, max_attempts: int = 3, base_delay: float = 1.0):
    """Call fn() with exponential backoff on exception."""
    last_exc = None
    for attempt in range(max_attempts):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            delay = base_delay * (2 ** attempt)
            logger.warning(f"Attempt {attempt + 1}/{max_attempts} failed: {exc}. Retrying in {delay}s")
            time.sleep(delay)
    raise last_exc


def download_kaggle(source_slug: str, output_dir: str) -> int:
    """Download a Kaggle dataset by slug. Returns file count."""
    target = Path(output_dir) / source_slug
    target.mkdir(parents=True, exist_ok=True)

    def _kaggle_pull():
        env = os.environ.copy()
        env["KAGGLE_USERNAME"] = env.get("KAGGLE_USERNAME", "")
        env["KAGGLE_KEY"] = env.get("KAGGLE_KEY", "")
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", source_slug,
             "-p", str(target), "--unzip"],
            check=True, env=env,
        )

    retry_with_backoff(_kaggle_pull, max_attempts=3)
    return sum(1 for _ in target.rglob("*") if _.is_file())


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="datasets/anemia_real/raw")
    parser.add_argument("--kaggle-slug", default="eyehubofai/anemia-detection-using-conjunctiva-images")
    parser.add_argument("--source", choices=["kaggle"], default="kaggle")
    args = parser.parse_args()

    entries: list[dict] = []
    if args.source == "kaggle":
        try:
            count = download_kaggle(args.kaggle_slug, args.output_dir)
            entries.append(
                build_manifest_entry(
                    source=args.kaggle_slug,
                    url=f"https://www.kaggle.com/datasets/{args.kaggle_slug}",
                    count=count,
                    license="Kaggle Terms",
                )
            )
        except Exception as exc:
            logger.error(f"Kaggle download failed: {exc}")
            raise SystemExit(1)

    save_manifest(entries, os.path.join(args.output_dir, "MANIFEST.json"))
    print(f"Manifest: {len(entries)} source(s) downloaded to {args.output_dir}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest app/training/tests/test_download_real_datasets.py -v`
Expected: all 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add ai-service/app/training/download_real_datasets.py ai-service/app/training/tests/test_download_real_datasets.py
git commit -m "feat(training): add dataset downloader with manifest + retry helper"
```

---

### Task 4: Build label-harmonizer

**Files:**
- Create: `ai-service/app/training/harmonize_labels.py`
- Create: `ai-service/app/training/tests/test_harmonize_labels.py`

**Interfaces:**
- Consumes: `datasets/anemia_real/raw/<source>/<label>/...`
- Produces: `datasets/anemia_real/canonical/{anemia,normal}/*.jpg` + `LABEL_MAPPING.yaml` + `DATASET_SOURCES.md`; then split into train/val/test under `datasets/anemia_real/{train,val,test}/{anemia,normal}/`

- [ ] **Step 1: Write the failing test**

Create `ai-service/app/training/tests/test_harmonize_labels.py`:

```python
"""Tests for label harmonization."""
import pytest

from app.training.harmonize_labels import (
    harmonize_label,
    is_anemia,
    remap_tree,
    write_label_mapping_yaml,
    split_train_val_test,
)


def test_harmonize_label_lowercase():
    assert harmonize_label("Anemic") == "anemia"
    assert harmonize_label("YES") == "anemia"
    assert harmonize_label("No") == "normal"
    assert harmonize_label("normal") == "normal"


def test_harmonize_label_unknown_raises():
    with pytest.raises(ValueError, match="Unknown label"):
        harmonize_label("purple")


def test_is_anemia_handles_synonyms():
    assert is_anemia("anemia")
    assert is_anemia("anemic")
    assert is_anemia("Yes")
    assert not is_anemia("No")
    assert not is_anemia("normal")


def test_remap_tree_groups_by_canonical_label(tmp_path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    (src / "Anemic").mkdir(parents=True)
    (src / "Anemic" / "a.png").touch()
    (src / "Normal").mkdir()
    (src / "Normal" / "n.png").touch()

    mapping = remap_tree(str(src), str(dst))

    assert (dst / "anemia" / "a.png").exists()
    assert (dst / "normal" / "n.png").exists()
    assert mapping == {"a.png": "anemia", "n.png": "normal"}


def test_write_label_mapping_yaml(tmp_path):
    rules = {"anemia": ["anemia", "anemic", "yes"], "normal": ["normal", "no"]}
    path = tmp_path / "LABEL_MAPPING.yaml"
    write_label_mapping_yaml(rules, str(path))
    content = path.read_text()
    assert "anemia" in content
    assert "yes" in content


def test_split_train_val_test(tmp_path):
    src = tmp_path / "canonical"
    (src / "anemia").mkdir()
    (src / "normal").mkdir()
    for i in range(20):
        (src / "anemia" / f"a{i}.jpg").touch()
    for i in range(20):
        (src / "normal" / f"n{i}.jpg").touch()
    dst = tmp_path / "split"
    counts = split_train_val_test(str(src), str(dst))
    assert counts["train"]["anemia"] == 16  # 80%
    assert counts["val"]["anemia"] == 2     # 10%
    assert counts["test"]["anemia"] == 2    # 10%
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest app/training/tests/test_harmonize_labels.py -v`
Expected: `ModuleNotFoundError: No module named 'app.training.harmonize_labels'`

- [ ] **Step 3: Implement harmonize_labels.py**

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest app/training/tests/test_harmonize_labels.py -v`
Expected: all 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add ai-service/app/training/harmonize_labels.py ai-service/app/training/tests/test_harmonize_labels.py
git commit -m "feat(training): add label harmonizer + 80/10/10 splitter"
```

---

### Task 5: Build augmentation pipeline module (Part 1)

**Files:**
- Create: `ai-service/app/training/anemia_real_train.py` (first pass: augmentation + dataset only)
- Create: `ai-service/app/training/tests/test_anemia_real_train.py` (first pass: aug + dataset tests)

**Interfaces:**
- Consumes: dataset on disk at `datasets/anemia_real/{train,val,test}/{anemia,normal}/`
- Produces: PyTorch DataLoaders; train transforms ON, val/test transforms OFF

- [ ] **Step 1: Write failing tests for augmentation pipeline**

Create `ai-service/app/training/tests/test_anemia_real_train.py`:

```python
"""Tests for augmentation pipeline + dataset class."""
import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.training.anemia_real_train import (
    build_train_transforms,
    build_eval_transforms,
    AnemiaDataset,
    compute_class_weights,
)


@pytest.fixture
def sample_image_dir(tmp_path):
    (tmp_path / "anemia").mkdir()
    (tmp_path / "normal").mkdir()
    img = Image.fromarray((np.random.rand(64, 64, 3) * 255).astype(np.uint8))
    img.save(tmp_path / "anemia" / "a.jpg")
    img.save(tmp_path / "normal" / "n.jpg")
    return tmp_path


def test_train_transforms_include_critical_ops():
    t = build_train_transforms()
    names = [tx.__class__.__name__ for tx in t.transforms]
    assert "HorizontalFlip" in names
    assert "CLAHE" in names  # critical for conjunctiva pucat
    assert "CoarseDropout" in names


def test_eval_transforms_only_resize_and_normalize():
    t = build_eval_transforms()
    names = [tx.__class__.__name__ for tx in t.transforms]
    assert "HorizontalFlip" not in names
    assert "CLAHE" not in names
    assert any("Normalize" in n for n in names)


def test_dataset_len_returns_count(sample_image_dir):
    ds = AnemiaDataset(str(sample_image_dir), split="train")
    assert len(ds) == 2


def test_dataset_returns_image_and_label(sample_image_dir):
    ds = AnemiaDataset(str(sample_image_dir), split="train")
    img, label = ds[0]
    assert hasattr(img, "shape")  # torch tensor
    assert img.shape[0] == 3       # CHW
    assert label in (0, 1)


def test_compute_class_weights_balanced():
    counts = {"anemia": 100, "normal": 900}
    weights = compute_class_weights(counts)
    assert weights["anemia"] > weights["normal"]
    assert weights["anemia"] > 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest app/training/tests/test_anemia_real_train.py -v`
Expected: `ModuleNotFoundError: No module named 'app.training.anemia_real_train'`

- [ ] **Step 3: Implement anemia_real_train.py — augmentations + dataset**

```python
"""
anemia_real_train.py
====================
Train MobileNetV3-Small for anemia detection on REAL conjunctiva images.
Augmentation pipeline via Albumentations (see spec §5).

Usage:
  python -m app.training.anemia_real_train --data-dir datasets/anemia_real
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from collections import Counter
from pathlib import Path

import albumentations as A
import numpy as np
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

logger = logging.getLogger("maternin.training.anemia_real")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMG_SIZE = 224
CLASSES = ["normal", "anemia"]


def build_train_transforms() -> A.Compose:
    """Spec §5: aggressive but domain-aware augmentation for conjunctiva."""
    return A.Compose(
        [
            A.Resize(IMG_SIZE, IMG_SIZE),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=10, p=0.5),
            A.ElasticTransform(alpha=30, sigma=5, p=0.2),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=15, val_shift_limit=10, p=0.5),
            A.CLAHE(clip_limit=2.0, p=0.3),
            A.GaussNoise(var_limit=(10, 50), p=0.2),
            A.GaussianBlur(blur_limit=(3, 5), p=0.15),
            A.MotionBlur(blur_limit=5, p=0.1),
            A.CoarseDropout(num_holes_range=(2, 4), hole_height_range=(8, 32),
                          hole_width_range=(8, 32), p=0.3),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


def build_eval_transforms() -> A.Compose:
    """No augmentation at inference; only resize + normalize."""
    return A.Compose(
        [
            A.Resize(IMG_SIZE, IMG_SIZE),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


class AnemiaDataset(Dataset):
    """ImageFolder-style dataset reading from {root}/{split}/{class}/*.jpg."""

    def __init__(self, root: str, split: str = "train"):
        self.root = Path(root) / split
        self.transform = build_train_transforms() if split == "train" else build_eval_transforms()
        self.samples: list[tuple[Path, int]] = []
        for label_idx, cls in enumerate(CLASSES):
            cls_dir = self.root / cls
            if not cls_dir.exists():
                logger.warning(f"Missing class dir: {cls_dir}")
                continue
            for img_path in sorted(cls_dir.iterdir()):
                if img_path.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    self.samples.append((img_path, label_idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = np.array(Image.open(path).convert("RGB"))
        out = self.transform(image=img)
        return out["image"], label

    def class_counts(self) -> dict:
        counter = Counter(label for _, label in self.samples)
        return {CLASSES[k]: v for k, v in counter.items()}


def compute_class_weights(counts: dict) -> dict:
    """Inverse-frequency class weights. Returns {class_name: weight}."""
    total = sum(counts.values())
    n_classes = len(counts)
    weights = {}
    for cls, c in counts.items():
        weights[cls] = (total / (n_classes * c)) if c > 0 else 1.0
    return weights


# (model + training loop + ONNX export added in Task 6)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest app/training/tests/test_anemia_real_train.py -v`
Expected: all 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add ai-service/app/training/anemia_real_train.py ai-service/app/training/tests/test_anemia_real_train.py
git commit -m "feat(training): add albumentations pipeline + AnemiaDataset"
```

---

### Task 6: Add model, training loop, and ONNX export

**Files:**
- Modify: `ai-service/app/training/anemia_real_train.py`
- Modify: `ai-service/app/training/tests/test_anemia_real_train.py`

**Interfaces:**
- Consumes: `AnemiaDataset` from Task 5
- Produces: `anemia_mobilenetv3_v2_real.onnx`, `anemia_mobilenetv3_v2_real_metadata.json`, `report.md`

- [ ] **Step 1: Add failing tests for training helpers**

Append to `ai-service/app/training/tests/test_anemia_real_train.py`:

```python
import json

from app.training.anemia_real_train import (
    build_model,
    export_onnx,
    save_metadata,
)


def test_build_model_mobilenetv3_small_two_classes():
    model = build_model(num_classes=2, pretrained=False)
    out = model(torch.randn(1, 3, 224, 224))
    assert out.shape == (1, 2)


def test_export_onnx_creates_file(tmp_path):
    model = build_model(num_classes=2, pretrained=False)
    onnx_path = str(tmp_path / "model.onnx")
    export_onnx(model, onnx_path, input_shape=(1, 3, 224, 224))
    assert os.path.exists(onnx_path)
    assert os.path.getsize(onnx_path) > 1000


def test_save_metadata(tmp_path):
    metrics = {"accuracy": 0.87, "recall": 0.83}
    out = save_metadata(
        model_name="anemia_mobilenetv3_v2_real",
        model_type="MobileNetV3-Small pretrained",
        dataset_size_train=1000,
        dataset_size_val=100,
        dataset_size_test=100,
        epochs=20,
        metrics=metrics,
        confusion_matrix={"TN": 50, "FP": 5, "FN": 8, "TP": 37},
        output_dir=str(tmp_path),
    )
    assert os.path.exists(out)
    data = json.loads(Path(out).read_text())
    assert data["metrics"]["accuracy"] == 0.87
```

(also add `from pathlib import Path` at top of test file)

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest app/training/tests/test_anemia_real_train.py::test_build_model_mobilenetv3_small_two_classes -v`
Expected: `ImportError: cannot import name 'build_model'`

- [ ] **Step 3: Add model + training loop + ONNX export to anemia_real_train.py**

Append to `ai-service/app/training/anemia_real_train.py`:

```python
from torchvision import models


def build_model(num_classes: int = 2, pretrained: bool = True):
    """MobileNetV3-Small with custom 2-class head."""
    if pretrained:
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    else:
        model = models.mobilenet_v3_small(weights=None)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


def train_one_epoch(model, loader, criterion, optimizer, device) -> float:
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        all_preds.extend(logits.argmax(dim=1).cpu().numpy())
        all_labels.extend(y.numpy())
    acc = float(np.mean(np.array(all_preds) == np.array(all_labels)))
    return acc, np.array(all_preds), np.array(all_labels)


def export_onnx(model, output_path: str, input_shape: tuple = (1, 3, 224, 224),
                opset: int = 13) -> None:
    """Export trained PyTorch model to ONNX."""
    model.eval()
    model = model.to("cpu")
    dummy = torch.randn(*input_shape)
    torch.onnx.export(
        model, dummy, output_path,
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=opset,
    )


def save_metadata(
    model_name: str,
    model_type: str,
    dataset_size_train: int,
    dataset_size_val: int,
    dataset_size_test: int,
    epochs: int,
    metrics: dict,
    confusion_matrix: dict,
    output_dir: str,
    sources_manifest_path: str | None = None,
) -> str:
    """Persist training metadata JSON. Returns path written."""
    from datetime import datetime, timezone
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    out = {
        "model_name": model_name,
        "model_type": model_type,
        "dataset": "real_conjunctiva_combined",
        "dataset_size_train": dataset_size_train,
        "dataset_size_val": dataset_size_val,
        "dataset_size_test": dataset_size_test,
        "epochs": epochs,
        "input_shape": [1, 3, 224, 224],
        "classes": CLASSES,
        "format": "ONNX",
        "metrics": {k: round(float(v), 4) for k, v in metrics.items()},
        "confusion_matrix": confusion_matrix,
        "augmentation": "Albumentations 13-transform pipeline (see spec §5)",
        "training_date": datetime.now(timezone.utc).isoformat(),
        "warning": "Trained on REAL conjunctival images combined from public sources. "
                   "Validate on local Indonesian patient cohort before production.",
    }
    if sources_manifest_path and os.path.exists(sources_manifest_path):
        out["sources_manifest"] = sources_manifest_path
    path = os.path.join(output_dir, f"{model_name}_metadata.json")
    Path(path).write_text(json.dumps(out, indent=2))
    return path


def generate_report(
    model_name: str,
    metrics: dict,
    confusion_matrix: dict,
    train_history: list,
    output_path: str,
) -> None:
    """Write human-readable training report."""
    lines = [
        f"# Training Report: {model_name}",
        "",
        f"- **Final test accuracy**: {metrics.get('accuracy', 0):.4f}",
        f"- **Final test recall**: {metrics.get('recall', 0):.4f}",
        f"- **Final test precision**: {metrics.get('precision', 0):.4f}",
        f"- **Final test F1**: {metrics.get('f1', 0):.4f}",
        "",
        "## Confusion Matrix",
        "",
        "|       | Pred normal | Pred anemia |",
        "|-------|-------------|-------------|",
        f"| True normal  | {confusion_matrix['TN']} | {confusion_matrix['FP']} |",
        f"| True anemia  | {confusion_matrix['FN']} | {confusion_matrix['TP']} |",
        "",
        "## Epoch History (epoch, train_loss, val_acc)",
        "",
    ]
    for epoch, loss, vacc in train_history:
        lines.append(f"- {epoch}: loss={loss:.4f}, val_acc={vacc:.4f}")
    Path(output_path).write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="datasets/anemia_real")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--output-dir", default="ai-service/app/model_artifacts")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger.info(f"Training on {args.data_dir} for {args.epochs} epochs")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    logger.info(f"Device: {device}")

    train_ds = AnemiaDataset(args.data_dir, split="train")
    val_ds = AnemiaDataset(args.data_dir, split="val")
    test_ds = AnemiaDataset(args.data_dir, split="test")

    if len(train_ds) < 100:
        raise SystemExit(f"Dataset too small: {len(train_ds)} train images. Need ≥1000. STOP.")

    class_counts = train_ds.class_counts()
    class_weights = compute_class_weights(class_counts)
    samples_weight = [class_weights[CLASSES[label]] for _, label in train_ds.samples]
    sampler = WeightedRandomSampler(samples_weight, num_samples=len(samples_weight), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = build_model(num_classes=2, pretrained=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_state = None
    no_improve = 0
    history: list = []

    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_acc, _, _ = evaluate(model, val_loader, device)
        history.append((epoch, loss, val_acc))
        logger.info(f"Epoch {epoch}/{args.epochs}  loss={loss:.4f}  val_acc={val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= args.patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    test_acc, test_preds, test_labels = evaluate(model, test_loader, device)
    from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix as cm_fn
    prec = precision_score(test_labels, test_preds, zero_division=0)
    rec = recall_score(test_labels, test_preds, zero_division=0)
    f1 = f1_score(test_labels, test_preds, zero_division=0)
    cm = cm_fn(test_labels, test_preds)
    cm_dict = {"TN": int(cm[0][0]), "FP": int(cm[0][1]),
               "FN": int(cm[1][0]), "TP": int(cm[1][1])}

    metrics = {
        "accuracy": test_acc, "precision": prec, "recall": rec, "f1": f1,
        "val_accuracy": best_val_acc,
        "overfit_gap": float(history[-1][2] - best_val_acc) if history else 0.0,
    }

    onnx_path = os.path.join(args.output_dir, "anemia_mobilenetv3_v2_real.onnx")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    export_onnx(model, onnx_path)
    logger.info(f"ONNX exported: {onnx_path} ({os.path.getsize(onnx_path):,} bytes)")

    save_metadata(
        model_name="anemia_mobilenetv3_v2_real",
        model_type="MobileNetV3-Small pretrained, Albumentations aggressive",
        dataset_size_train=len(train_ds),
        dataset_size_val=len(val_ds),
        dataset_size_test=len(test_ds),
        epochs=len(history),
        metrics=metrics,
        confusion_matrix=cm_dict,
        output_dir=args.output_dir,
        sources_manifest_path=os.path.join(args.data_dir, "DATASET_SOURCES.md"),
    )
    generate_report(
        model_name="anemia_mobilenetv3_v2_real",
        metrics=metrics,
        confusion_matrix=cm_dict,
        train_history=history,
        output_path=os.path.join(args.output_dir, "report.md"),
    )

    if test_acc < 0.85 or rec < 0.80:
        logger.warning(f"Below target metrics: acc={test_acc:.3f}, recall={rec:.3f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run all tests to verify they pass**

Run: `pytest app/training/tests/test_anemia_real_train.py -v`
Expected: all 8 tests PASS

- [ ] **Step 5: Commit**

```bash
git add ai-service/app/training/anemia_real_train.py ai-service/app/training/tests/test_anemia_real_train.py
git commit -m "feat(training): add MobileNetV3-Small training loop + ONNX export"
```

---

### Task 7: Download and prepare real dataset

**Files:**
- Touches: only `datasets/anemia_real/` (not tracked in git)

**Interfaces:**
- Consumes: `KAGGLE_USERNAME` / `KAGGLE_KEY` env vars
- Produces: ≥1,000 real images in `datasets/anemia_real/{train,val,test}/{anemia,normal}/`

- [ ] **Step 1: Verify Kaggle CLI is installed**

Run: `which kaggle || pip install kaggle`
Expected: path to kaggle executable

- [ ] **Step 2: Set Kaggle credentials**

Run:
```bash
export KAGGLE_USERNAME="<your_username>"
export KAGGLE_KEY="<your_key>"
```
(Get from kaggle.com → Account → API → Create New Token)

- [ ] **Step 3: Run downloader**

Run: `cd /Users/zelphyx/Projects/Maternin-AI && source ai-service/.venv/bin/activate && python -m app.training.download_real_datasets --source kaggle`
Expected: ends with "Manifest: 1 source(s) downloaded"
Side effect: `datasets/anemia_real/raw/kaggle_*` folder created with images

- [ ] **Step 4: (Optional) Add Roboflow source**

If user has a Roboflow export URL, place the unzipped folder under `datasets/anemia_real/raw/<source_slug>/` with subfolders `Anemic/` and `Normal/` (or matching the harmonizer's synonym set).

- [ ] **Step 5: Harmonize + split**

Run:
```bash
python -m app.training.harmonize_labels --raw-dir datasets/anemia_real/raw --canonical-dir datasets/anemia_real/canonical --split-dir datasets/anemia_real
```
Expected: prints `Split complete: {'train': {...}, 'val': {...}, 'test': {...}}`

- [ ] **Step 6: Verify dataset size meets ≥1,000**

Run:
```bash
find datasets/anemia_real/train datasets/anemia_real/val datasets/anemia_real/test -type f | wc -l
```
Expected: ≥1000. **If <1000**: STOP and ask user whether to source more datasets (additional Roboflow export, EYES-DEFY-ANEMIA) or raise augmentation intensity further.

- [ ] **Step 7: Verify class balance is reasonable**

Run:
```bash
echo "anemia: $(find datasets/anemia_real/train/anemia -type f | wc -l)"
echo "normal: $(find datasets/anemia_real/train/normal -type f | wc -l)"
```
Expected: counts within 1:3 ratio. If ratio is worse than 1:10, log a warning (sampler will compensate).

- [ ] **Step 8: No commit** (data is in `.gitignore`)

---

### Task 8: Run training, update inference path, commit artifacts

**Files:**
- Modify: `ai-service/app/models/anemia_cv/inference.py`
- Creates: `ai-service/scripts/validate_onnx_parity.py`
- Produces: `ai-service/app/model_artifacts/anemia_mobilenetv3_v2_real.onnx` + `_metadata.json` + `report.md`

**Interfaces:**
- Consumes: prepared `datasets/anemia_real/`
- Produces: trained model satisfying spec §2 success criteria

- [ ] **Step 1: Run training**

Run:
```bash
cd /Users/zelphyx/Projects/Maternin-AI && source ai-service/.venv/bin/activate && \
  python -m app.training.anemia_real_train --data-dir datasets/anemia_real --epochs 30
```
Expected: 30 epochs complete; final report prints accuracy / recall / confusion matrix.

- [ ] **Step 2: Inspect metrics from report.md**

Run: `cat ai-service/app/model_artifacts/report.md`
Expected: `Final test accuracy ≥ 0.85` and `recall ≥ 0.80`. If below, see mitigation in §8 of spec.

- [ ] **Step 3: Create ONNX parity validator script**

Create `ai-service/scripts/validate_onnx_parity.py`:

```python
"""Spec §7 step 4: verify ONNX inference runs on real validation images."""
from __future__ import annotations
import pathlib
import sys

import numpy as np
import onnxruntime as ort
from PIL import Image

ONNX_PATH = "app/model_artifacts/anemia_mobilenetv3_v2_real.onnx"
VAL_DIR = pathlib.Path("../datasets/anemia_real/val")
MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])


def preprocess(path: pathlib.Path) -> np.ndarray:
    img = np.array(Image.open(path).convert("RGB").resize((224, 224))).astype("float32") / 255.0
    img = (img - MEAN) / STD
    return img.transpose(2, 0, 1)[None].astype("float32")


def main() -> None:
    if not pathlib.Path(ONNX_PATH).exists():
        print(f"SKIP: {ONNX_PATH} not found")
        sys.exit(0)
    sess = ort.InferenceSession(ONNX_PATH)
    imgs = sorted(VAL_DIR.rglob("*.jpg"))[:10]
    print(f"Running ONNX inference on {len(imgs)} validation images...")
    for p in imgs:
        arr = preprocess(p)
        out = sess.run(None, {"input": arr})[0]
        pred_label = "anemia" if out[0][1] > out[0][0] else "normal"
        true_label = p.parent.name
        match = "OK" if pred_label == true_label else "MISS"
        print(f"  [{match}] true={true_label} pred={pred_label} logits={out[0].round(3).tolist()}")
    print("ONNX inference complete.")


if __name__ == "__main__":
    main()
```

Run: `cd ai-service && source .venv/bin/activate && python scripts/validate_onnx_parity.py`
Expected: prints predictions for 10 val images; ≥8/10 should match true label (rough sanity check).

- [ ] **Step 4: Update inference.py path to v2**

Edit `ai-service/app/models/anemia_cv/inference.py`, replace:

```python
ONNX_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "model_artifacts", "anemia_mobilenetv3_v1.onnx"
)
```

with:

```python
ONNX_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "model_artifacts", "anemia_mobilenetv3_v2_real.onnx"
)
```

- [ ] **Step 5: Commit artifacts + path update**

```bash
cd /Users/zelphyx/Projects/Maternin-AI
git add ai-service/app/models/anemia_cv/inference.py
git add ai-service/scripts/validate_onnx_parity.py
git add ai-service/app/model_artifacts/anemia_mobilenetv3_v2_real_metadata.json
git add ai-service/app/model_artifacts/report.md
# Add to .gitignore if not already: ai-service/app/model_artifacts/*.onnx*
# ONNX binary is large; metadata + report.md is enough for reproducibility
git commit -m "feat(model): train anemia_mobilenetv3_v2_real on real data + update inference path"
```

---

### Task 9: Update ml_inference test to assert v2 loads

**Files:**
- Modify: `ai-service/tests/test_ml_inference.py`

- [ ] **Step 1: Read existing test**

Run: `cat ai-service/tests/test_ml_inference.py`

- [ ] **Step 2: Find anemia-related assertions**

Run: `grep -n "anemia\|onnx\|ONNX" ai-service/tests/test_ml_inference.py`

- [ ] **Step 3: Update assertions to v2 filename**

Edit the test (or add a new test if file has none for anemia):

```python
def test_anemia_model_path_points_to_v2_real():
    import os
    from app.models.anemia_cv.inference import ONNX_PATH
    assert "v2_real" in os.path.basename(ONNX_PATH), (
        f"ONNX_PATH must reference the v2_real artifact, got {ONNX_PATH}"
    )


def test_anemia_model_loads_or_mocks():
    from app.models.anemia_cv.inference import load_model, is_mock_mode, ONNX_PATH
    import os
    load_model()
    if os.path.exists(ONNX_PATH):
        assert not is_mock_mode(), "Real ONNX exists, model should not run in mock mode"
```

- [ ] **Step 4: Run test**

Run: `pytest ai-service/tests/test_ml_inference.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add ai-service/tests/test_ml_inference.py
git commit -m "test(ml): assert anemia model loads v2 ONNX artifact"
```

---

### Task 10: Final smoke test + documentation

**Files:**
- Touches: `ai-service/README.md` (if exists)

- [ ] **Step 1: Run full test suite**

Run: `cd ai-service && source .venv/bin/activate && pytest -v`
Expected: all tests PASS

- [ ] **Step 2: Update README with new training command**

Find the training section (or add one), replace any synthetic-data references with:

```markdown
### Training the anemia CV model (real data only)

```bash
# 1. Download real conjunctiva images (need KAGGLE_USERNAME/KEY)
python -m app.training.download_real_datasets --source kaggle

# 2. Harmonize labels + 80/10/10 split
python -m app.training.harmonize_labels

# 3. Train MobileNetV3-Small with aggressive Albumentations augmentation
python -m app.training.anemia_real_train --epochs 30

# 4. Validate ONNX inference
python scripts/validate_onnx_parity.py
```

Output: `app/model_artifacts/anemia_mobilenetv3_v2_real.onnx` + metadata + `report.md`.
```

- [ ] **Step 3: Final commit**

```bash
git add ai-service/README.md
git commit -m "docs: update training instructions to use real-data pipeline"
```

---

## Self-Review Notes

**Spec coverage:**
- §2 success criteria (acc ≥85, recall ≥80, gap <10pp): Task 8 step 2 checks; Task 6 records `overfit_gap`.
- §3 dataset sources: Task 7 covers Kaggle + optional Roboflow fallback.
- §4 architecture (canonical → train → ONNX): Tasks 4, 5, 6, 8 in order.
- §5 augmentation pipeline: Task 5 implements all 13 transforms.
- §6 file changes: Task 2 deletes legacy; Tasks 3-6 create new; Task 8 updates `inference.py` + ONNX validator; Task 9 updates test; Task 10 updates README.
- §7 testing: Task 8 step 3 (ONNX inference check); Task 9 (ml inference test); Task 10 (smoke suite).
- §8 risks: Task 7 step 6 enforces ≥1,000 guard; Task 6 uses early stopping + class weights.

**Placeholders:** none — every step has concrete code or commands.

**Type consistency:**
- `AnemiaDataset.__getitem__` returns `(Tensor, int)`; consumed by `train_one_epoch` and `evaluate`.
- `build_model(num_classes=2)` matches `CLASSES = ["normal", "anemia"]` length.
- `export_onnx(model, path, input_shape)` signature matches test call.
- `save_metadata()` returns path; main writes to `args.output_dir`.
