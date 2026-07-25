"""
MaternIn AI — Training Script: MobileNetV3-Small Anemia Detection (CV)
========================================================================
Fine-tune MobileNetV3-Small pada dataset citra konjungtiva untuk deteksi anemia.

Karena dataset HemaVision yang ada berupa repo referensi (bukan raw images),
script ini menyiapkan pipeline training lengkap yang bisa dijalankan ketika
dataset gambar konjungtiva tersedia, atau menggunakan model .keras yang sudah
ada dari HemaVision sebagai baseline.

Dataset yang dibutuhkan:
  - Folder gambar konjungtiva terklasifikasi:
    datasets/anemia_conjunctiva/images/anemia/
    datasets/anemia_conjunctiva/images/normal/

Model pre-trained tersedia:
  - datasets/anemia_conjunctiva/HemaVision-Anemia-Triage/models/mobilenet_final.keras

Output: app/model_artifacts/anemia_mobilenetv3_v1.onnx

Jalankan:
    cd /Users/zelphyx/Projects/Maternin-AI/ai-service
    source .venv/bin/activate
    python app/training/anemia_cv_train.py
"""

import os
import sys
import json
import shutil
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("maternin.training.anemia_cv")

# ── Paths ────────────────────────────────────────────────────────────────
ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), "..", "model_artifacts")
PRETRAINED_KERAS = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "datasets",
    "anemia_conjunctiva", "HemaVision-Anemia-Triage", "models", "mobilenet_final.keras"
)
CUSTOM_DATASET_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "datasets",
    "anemia_conjunctiva", "images"
)
OUTPUT_ONNX = os.path.join(ARTIFACT_DIR, "anemia_mobilenetv3_v1.onnx")
OUTPUT_META = os.path.join(ARTIFACT_DIR, "anemia_mobilenetv3_v1_metadata.json")


def convert_keras_to_onnx():
    """
    Konversi model .keras yang sudah ada ke format ONNX untuk inferensi cepat.
    """
    print("=" * 70)
    print("MaternIn — Anemia CV Model (Keras → ONNX Conversion)")
    print("=" * 70)

    if not os.path.exists(PRETRAINED_KERAS):
        print(f"\n❌ Pre-trained model not found: {PRETRAINED_KERAS}")
        print("   Jalankan mode training manual dengan dataset gambar konjungtiva.")
        return False

    try:
        import tensorflow as tf
        import tf2onnx
        import numpy as np

        print(f"\n[1/3] Loading pre-trained Keras model...")
        print(f"  Path: {PRETRAINED_KERAS}")
        print(f"  Size: {os.path.getsize(PRETRAINED_KERAS):,} bytes")

        model = tf.keras.models.load_model(PRETRAINED_KERAS)
        model.summary()

        print(f"\n[2/3] Converting to ONNX format...")
        input_shape = model.input_shape
        print(f"  Input shape: {input_shape}")

        # Convert
        spec = (tf.TensorSpec(input_shape, tf.float32, name="input"),)
        model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec)

        os.makedirs(ARTIFACT_DIR, exist_ok=True)
        with open(OUTPUT_ONNX, "wb") as f:
            f.write(model_proto.SerializeToString())

        print(f"\n[3/3] ONNX model saved!")
        print(f"  Output: {OUTPUT_ONNX}")
        print(f"  Size: {os.path.getsize(OUTPUT_ONNX):,} bytes")

        # Save metadata
        metadata = {
            "model_name": "anemia_mobilenetv3_v1",
            "model_type": "MobileNetV2 (HemaVision pre-trained)",
            "source": "HemaVision-Anemia-Triage/models/mobilenet_final.keras",
            "input_shape": list(input_shape) if input_shape else "unknown",
            "output": "binary classification (anemia / normal)",
            "format": "ONNX",
            "note": "Converted from pre-trained HemaVision model. Fine-tune with local conjunctiva dataset for higher accuracy.",
        }
        with open(OUTPUT_META, "w") as f:
            json.dump(metadata, f, indent=2)

        return True

    except ImportError as e:
        print(f"\n⚠️ TensorFlow/tf2onnx not installed: {e}")
        print("   Install with: pip install tensorflow tf2onnx")
        print("   Falling back to mock model setup...")
        return False


def setup_mock_model():
    """
    Buat mock ONNX placeholder jika konversi tidak bisa dilakukan.
    Model asli akan digunakan saat TF/ONNX runtime tersedia.
    """
    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    metadata = {
        "model_name": "anemia_mobilenetv3_v1",
        "model_type": "MobileNetV3-Small (Mock Placeholder)",
        "source": "Placeholder — model asli tersedia di HemaVision-Anemia-Triage/models/mobilenet_final.keras",
        "status": "mock",
        "instructions": [
            "1. Install TensorFlow: pip install tensorflow tf2onnx",
            "2. Jalankan ulang: python app/training/anemia_cv_train.py",
            "3. Atau gunakan model .keras langsung via TFLite/Keras inference",
        ],
        "pretrained_keras_path": PRETRAINED_KERAS,
        "pretrained_keras_exists": os.path.exists(PRETRAINED_KERAS),
    }
    with open(OUTPUT_META, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n📋 Mock metadata saved: {OUTPUT_META}")
    print(f"   Pre-trained .keras exists: {os.path.exists(PRETRAINED_KERAS)}")

    if os.path.exists(PRETRAINED_KERAS):
        print(f"\n💡 TIP: Model .keras dari HemaVision tersedia!")
        print(f"   Path: {PRETRAINED_KERAS}")
        print(f"   Untuk konversi ke ONNX, install: pip install tensorflow tf2onnx")
        print(f"   Atau wrapper inference bisa langsung load .keras via TensorFlow.")


def train_from_scratch():
    """
    Training dari nol jika ada dataset gambar konjungtiva lokal.
    Butuh folder: datasets/anemia_conjunctiva/images/{anemia,normal}/
    """
    print("\n" + "=" * 70)
    print("MaternIn — Anemia CV Model (Training From Scratch)")
    print("=" * 70)

    if not os.path.exists(CUSTOM_DATASET_DIR):
        print(f"\n❌ Custom dataset not found: {CUSTOM_DATASET_DIR}")
        print("   Buat folder struktur:")
        print("     datasets/anemia_conjunctiva/images/anemia/   (foto konjungtiva pucat)")
        print("     datasets/anemia_conjunctiva/images/normal/    (foto konjungtiva normal)")
        return False

    try:
        import torch
        import torch.nn as nn
        from torchvision import models, transforms, datasets
        from torch.utils.data import DataLoader, random_split

        print(f"\n[1/5] Loading dataset from {CUSTOM_DATASET_DIR}...")

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        dataset = datasets.ImageFolder(CUSTOM_DATASET_DIR, transform=transform)
        print(f"  Total images: {len(dataset)}")
        print(f"  Classes: {dataset.classes}")

        # Split 80:20
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

        print(f"  Train: {train_size}, Validation: {val_size}")

        # Use MPS (Apple Silicon GPU) if available
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"\n[2/5] Device: {device}")

        # MobileNetV3-Small with custom head
        print(f"\n[3/5] Building MobileNetV3-Small...")
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 2)
        model = model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        # Training loop
        print(f"\n[4/5] Training for 20 epochs...")
        best_val_acc = 0.0
        for epoch in range(20):
            model.train()
            train_loss = 0.0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            # Validation
            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            val_acc = correct / total
            print(f"  Epoch {epoch+1:2d}/20 | Loss: {train_loss/len(train_loader):.4f} | Val Acc: {val_acc:.4f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), os.path.join(ARTIFACT_DIR, "anemia_mobilenetv3_best.pt"))

        # Export to ONNX
        print(f"\n[5/5] Exporting to ONNX...")
        model.eval()
        model = model.to("cpu")
        dummy_input = torch.randn(1, 3, 224, 224)
        torch.onnx.export(
            model, dummy_input, OUTPUT_ONNX,
            input_names=["input"], output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        )

        print(f"  Best Validation Accuracy: {best_val_acc:.4f}")
        print(f"  ONNX saved: {OUTPUT_ONNX}")

        metadata = {
            "model_name": "anemia_mobilenetv3_v1",
            "model_type": "MobileNetV3-Small (Fine-tuned)",
            "input_shape": [1, 3, 224, 224],
            "classes": dataset.classes,
            "dataset_size": len(dataset),
            "best_val_accuracy": round(best_val_acc, 4),
            "epochs": 20,
            "device": device,
            "format": "ONNX",
        }
        with open(OUTPUT_META, "w") as f:
            json.dump(metadata, f, indent=2)

        return True

    except ImportError as e:
        print(f"\n❌ PyTorch not installed: {e}")
        print("   Install with: pip install torch torchvision")
        return False


def main():
    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    # Priority 1: Train from scratch if custom dataset exists
    if os.path.exists(CUSTOM_DATASET_DIR):
        if train_from_scratch():
            return

    # Priority 2: Convert pre-trained .keras to ONNX
    if os.path.exists(PRETRAINED_KERAS):
        if convert_keras_to_onnx():
            return

    # Priority 3: Setup mock placeholder
    print("\n⚠️ No model training/conversion possible — setting up mock placeholder.")
    setup_mock_model()

    print(f"\n{'=' * 70}")
    print(f"✅ Anemia CV setup complete!")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
