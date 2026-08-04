"""
MaternIn AI Service — Upload Artifacts to Hugging Face Hub
===========================================================
One-time setup (or run ulang kalau model di-retrain) untuk upload:

  1. Model artifacts (~125 MB) → HF Hub Model repo `maternin-models`
     - 2 .pkl (preeclampsia LR + risk aggregator XGBoost)
     - 3 .onnx + 3 .onnx.data (ConvNeXt, EfficientNet-B0, MobileNetV3)
     - 6 _metadata.json
  2. Runtime data (~5 MB) → HF Hub Dataset repo `maternin-data`
     - kb/maternal_health_qa_kemenkes_500.json (chatbot KB)
     - nutrition/tkpi_indonesian_food_master_300.csv (nutrition DB)

Prerequisites:
  pip install huggingface_hub
  export HF_TOKEN="hf_xxx..."  # write token
  export HF_USERNAME="zelphyx"  # ganti ke username HF lo

Usage:
  python scripts/upload_artifacts.py
  python scripts/upload_artifacts.py --only models
  python scripts/upload_artifacts.py --only data
  python scripts/upload_artifacts.py --models-repo myuser/custom-name

Default username + repo names pakai env vars, fallback ke placeholder
supaya lo sadar perlu setting.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Resolve repo root = ../../ from this file
REPO_ROOT = Path(__file__).resolve().parents[2]
AI_SERVICE = REPO_ROOT / "ai-service"
MODEL_ARTIFACTS = AI_SERVICE / "app" / "model_artifacts"
DATASETS_ROOT = REPO_ROOT / "datasets"


def upload_models(hf_api, username: str, repo_id: str | None = None) -> str:
    """Upload model artifacts to HF Hub Model repo. Return final repo_id."""
    repo_id = repo_id or f"{username}/maternin-models"
    print(f"\n[1/2] Uploading model artifacts → https://huggingface.co/{repo_id}")
    print(f"      Source: {MODEL_ARTIFACTS}")

    if not MODEL_ARTIFACTS.exists():
        print(f"❌ Model artifacts dir not found: {MODEL_ARTIFACTS}")
        sys.exit(1)

    # Create repo kalau belum ada (idempotent)
    hf_api.create_repo(
        repo_id=repo_id,
        repo_type="model",
        private=True,
        exist_ok=True,
    )

    # Whitelist files — skip report.md (training notes, bukan model)
    files_to_upload = sorted([
        f.name for f in MODEL_ARTIFACTS.iterdir()
        if f.is_file()
        and not f.name.endswith(".md")
        and not f.name.startswith(".")
    ])

    print(f"      Files ({len(files_to_upload)}):")
    for fn in files_to_upload:
        size_mb = (MODEL_ARTIFACTS / fn).stat().st_size / (1024 * 1024)
        print(f"        - {fn} ({size_mb:.2f} MB)")

    # Upload per file — kasih progress per item
    for fn in files_to_upload:
        local_path = MODEL_ARTIFACTS / fn
        size_mb = local_path.stat().st_size / (1024 * 1024)
        print(f"      ⬆ {fn} ({size_mb:.2f} MB)...", end=" ", flush=True)
        hf_api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=fn,
            repo_id=repo_id,
            repo_type="model",
        )
        print("ok")

    print(f"✅ Models uploaded → https://huggingface.co/{repo_id}")
    return repo_id


def upload_data(hf_api, username: str, repo_id: str | None = None) -> str:
    """Upload runtime data to HF Hub Dataset repo. Return final repo_id."""
    repo_id = repo_id or f"{username}/maternin-data"
    print(f"\n[2/2] Uploading runtime data → https://huggingface.co/datasets/{repo_id}")

    # Create repo kalau belum ada
    hf_api.create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        private=True,
        exist_ok=True,
    )

    # 1. Chatbot KB
    kb_src = DATASETS_ROOT / "buku_kia_kemenkes" / "maternal_health_qa_kemenkes_500.json"
    if not kb_src.exists():
        print(f"❌ KB file not found: {kb_src}")
        sys.exit(1)
    kb_size = kb_src.stat().st_size / 1024
    print(f"      ⬆ kb/{kb_src.name} ({kb_size:.1f} KB)...", end=" ", flush=True)
    hf_api.upload_file(
        path_or_fileobj=str(kb_src),
        path_in_repo=f"kb/{kb_src.name}",
        repo_id=repo_id,
        repo_type="dataset",
    )
    print("ok")

    # 2. Nutrition DB
    nut_src = DATASETS_ROOT / "tkpi_nutrition" / "tkpi_indonesian_food_master_300.csv"
    if not nut_src.exists():
        print(f"❌ Nutrition CSV not found: {nut_src}")
        sys.exit(1)
    nut_size = nut_src.stat().st_size / 1024
    print(f"      ⬆ nutrition/{nut_src.name} ({nut_size:.1f} KB)...", end=" ", flush=True)
    hf_api.upload_file(
        path_or_fileobj=str(nut_src),
        path_in_repo=f"nutrition/{nut_src.name}",
        repo_id=repo_id,
        repo_type="dataset",
    )
    print("ok")

    print(f"✅ Data uploaded → https://huggingface.co/datasets/{repo_id}")
    return repo_id


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload MaternIn artifacts to HF Hub")
    parser.add_argument(
        "--only", choices=["models", "data", "all"], default="all",
        help="Which artifacts to upload (default: all)",
    )
    parser.add_argument(
        "--models-repo", help="Override models repo (default: <username>/maternin-models)",
    )
    parser.add_argument(
        "--data-repo", help="Override data repo (default: <username>/maternin-data)",
    )
    parser.add_argument(
        "--public", action="store_true",
        help="Make repos public (default: private)",
    )
    args = parser.parse_args()

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("❌ Set HF_TOKEN env var dulu (write token dari https://huggingface.co/settings/tokens)")
        sys.exit(1)

    username = os.environ.get("HF_USERNAME")
    if not username:
        print("❌ Set HF_USERNAME env var (HF username lo)")
        sys.exit(1)

    from huggingface_hub import HfApi
    hf_api = HfApi(token=hf_token)

    if args.public:
        # Force public visibility — set via env agar create_repo() honor
        os.environ["HF_HUB_PRIVATE"] = "0"

    print(f"🔑 HF user: {username}")
    print(f"📦 Token:   hf_{'*' * (len(hf_token) - 6)}{hf_token[-4:]}")

    if args.only in ("models", "all"):
        models_repo = upload_models(hf_api, username, args.models_repo)
        print(f"\n   Set Space env: HF_HUB_REPO={models_repo}")

    if args.only in ("data", "all"):
        data_repo = upload_data(hf_api, username, args.data_repo)
        print(f"\n   Set Space env: HF_HUB_DATA_REPO={data_repo}")

    print("\n🎉 Done. Next step: setup Space secrets — see scripts/setup_space_secrets.md")


if __name__ == "__main__":
    main()