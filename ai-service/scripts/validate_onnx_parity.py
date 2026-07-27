"""Spec §7 step 4: verify ONNX inference runs on real validation images.

Run from ai-service/:
    python scripts/validate_onnx_parity.py

Exits 0 if model artifact is missing (so CI doesn't fail before training).
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
from PIL import Image

ONNX_PATH = pathlib.Path(__file__).resolve().parent.parent / "app" / "model_artifacts" / "anemia_convnext_tiny_v2_real.onnx"
VAL_DIR = pathlib.Path(__file__).resolve().parent.parent.parent / "datasets" / "anemia_real" / "val"
MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])


def preprocess(path: pathlib.Path) -> np.ndarray:
    img = np.array(Image.open(path).convert("RGB").resize((224, 224))).astype("float32") / 255.0
    img = (img - MEAN) / STD
    return img.transpose(2, 0, 1)[None].astype("float32")


def main() -> int:
    if not ONNX_PATH.exists():
        print(f"SKIP: {ONNX_PATH.name} not found. Train first: python -m app.training.anemia_real_train")
        return 0

    import onnxruntime as ort
    sess = ort.InferenceSession(str(ONNX_PATH))

    if not VAL_DIR.exists():
        print(f"SKIP: val dir not found: {VAL_DIR}")
        return 0

    imgs = sorted(VAL_DIR.rglob("*.jpg"))[:10]
    if not imgs:
        print(f"SKIP: no validation images in {VAL_DIR}")
        return 0

    print(f"Running ONNX inference on {len(imgs)} validation images...")
    matches = 0
    for p in imgs:
        arr = preprocess(p)
        out = sess.run(None, {"input": arr})[0]
        pred_label = "anemia" if out[0][1] > out[0][0] else "normal"
        true_label = p.parent.name
        match = "OK" if pred_label == true_label else "MISS"
        if match == "OK":
            matches += 1
        print(f"  [{match}] true={true_label} pred={pred_label} logits={out[0].round(3).tolist()}")

    print(f"ONNX inference complete: {matches}/{len(imgs)} matched.")
    return 0


if __name__ == "__main__":
    sys.exit(main())