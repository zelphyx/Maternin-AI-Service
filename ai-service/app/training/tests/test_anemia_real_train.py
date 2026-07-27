"""Tests for augmentation pipeline + dataset class."""
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from app.training.anemia_real_train import (
    build_train_transforms,
    build_eval_transforms,
    AnemiaDataset,
    compute_class_weights,
    build_model,
    export_onnx,
    save_metadata,
)


@pytest.fixture
def sample_image_dir(tmp_path):
    (tmp_path / "train" / "anemia").mkdir(parents=True)
    (tmp_path / "train" / "normal").mkdir(parents=True)
    img = Image.fromarray((np.random.rand(64, 64, 3) * 255).astype(np.uint8))
    img.save(tmp_path / "train" / "anemia" / "a.jpg")
    img.save(tmp_path / "train" / "normal" / "n.jpg")
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
