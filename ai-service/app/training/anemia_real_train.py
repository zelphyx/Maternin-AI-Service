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
from torchvision import models

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


def build_model(num_classes: int = 2, pretrained: bool = True, dropout: float = 0.3,
                arch: str = "mobilenetv3_small"):
    """Build classification backbone. arch ∈ {mobilenetv3_small, efficientnet_b0, efficientnet_b3, convnext_tiny}."""
    if arch == "mobilenetv3_small":
        if pretrained:
            model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        else:
            model = models.mobilenet_v3_small(weights=None)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )
        return model
    elif arch == "efficientnet_b0":
        if pretrained:
            model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        else:
            model = models.efficientnet_b0(weights=None)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )
        return model
    elif arch == "efficientnet_b3":
        if pretrained:
            model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)
        else:
            model = models.efficientnet_b3(weights=None)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )
        return model
    elif arch == "convnext_tiny":
        if pretrained:
            model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
        else:
            model = models.convnext_tiny(weights=None)
        in_features = model.classifier[-1].in_features
        # ConvNeXt classifier is Sequential(LayerNorm, Flatten, Linear) — keep LayerNorm, replace Linear
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model
    else:
        raise ValueError(f"Unknown arch: {arch}")


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
        "augmentation": "Albumentations 13-transform pipeline (see spec section 5)",
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
    parser.add_argument("--arch", default="mobilenetv3_small",
                        choices=["mobilenetv3_small", "efficientnet_b0", "efficientnet_b3", "convnext_tiny"])
    parser.add_argument("--output-dir", default="ai-service/app/model_artifacts")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger.info(f"Training on {args.data_dir} for {args.epochs} epochs")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    logger.info(f"Device: {device}")

    train_ds = AnemiaDataset(args.data_dir, split="train")
    val_ds = AnemiaDataset(args.data_dir, split="val")
    test_ds = AnemiaDataset(args.data_dir, split="test")

    if len(train_ds) < 10:
        raise SystemExit(f"Dataset too small: {len(train_ds)} train images. Need >=10. STOP.")

    class_counts = train_ds.class_counts()
    class_weights = compute_class_weights(class_counts)
    samples_weight = [class_weights[CLASSES[label]] for _, label in train_ds.samples]
    sampler = WeightedRandomSampler(samples_weight, num_samples=len(samples_weight), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = build_model(num_classes=2, pretrained=True, arch=args.arch).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    # CosineAnnealing LR schedule — helps converge better on small datasets
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_state = None
    no_improve = 0
    history: list = []

    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        scheduler.step()
        val_acc, _, _ = evaluate(model, val_loader, device)
        history.append((epoch, loss, val_acc))
        logger.info(f"Epoch {epoch}/{args.epochs}  loss={loss:.4f}  val_acc={val_acc:.4f}  lr={scheduler.get_last_lr()[0]:.2e}")
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

    model_name = f"anemia_{args.arch.replace('_', '_')}_v2_real"
    onnx_path = os.path.join(args.output_dir, f"{model_name}.onnx")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    export_onnx(model, onnx_path)
    logger.info(f"ONNX exported: {onnx_path} ({os.path.getsize(onnx_path):,} bytes)")

    save_metadata(
        model_name=model_name,
        model_type=f"{args.arch} pretrained, Albumentations aggressive",
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
        model_name=model_name,
        metrics=metrics,
        confusion_matrix=cm_dict,
        train_history=history,
        output_path=os.path.join(args.output_dir, "report.md"),
    )

    if test_acc < 0.85 or rec < 0.80:
        logger.warning(f"Below target metrics: acc={test_acc:.3f}, recall={rec:.3f}")


if __name__ == "__main__":
    main()
