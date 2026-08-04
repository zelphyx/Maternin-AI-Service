"""
MaternIn AI Service — Artifact Loader
======================================
Centralized helper untuk download model artifacts + runtime data
dari Hugging Face Hub saat startup.

Behavior:
  - Production (HF Space):
      * `HF_HUB_REPO` di-set → download model artifacts dari HF Hub Model repo
      * `HF_HUB_DATA_REPO` di-set → download runtime data (KB, nutrition DB) dari HF Hub Dataset repo
      * Cache di `/app/.cache/hub` (atau sesuai `HF_HUB_CACHE`)
  - Development (local):
      * Kedua env var kosong → pakai path lokal (`app/model_artifacts/` + `datasets/` di root)
      * Tidak ada download — tetap cepat untuk iterasi

Env vars:
  HF_TOKEN                — Hub token (read-only cukup untuk Space runtime)
  HF_HUB_REPO             — model repo, mis. "zelphyx/maternin-models"
  HF_HUB_DATA_REPO        — data repo, mis. "zelphyx/maternin-data"
  HF_HUB_CACHE            — local cache dir (default "/app/.cache/hub" di HF Space)
"""
from __future__ import annotations

import logging
import os
import shutil
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger("maternin.ai.artifact_loader")

_CACHE_ROOT = Path(os.environ.get("HF_HUB_CACHE", "/app/.cache/hub"))

# Project root in dev = ../../.. from this file (ai-service/app/core/)
_LOCAL_DATA_ROOT = Path(__file__).resolve().parents[3] / "datasets"

_models_dir: Path | None = None
_data_dir: Path | None = None


def ensure_model_artifacts() -> Path:
    """
    Pastiin model artifacts (.pkl + .onnx + .onnx.data) tersedia di disk.
    Return path ke folder yang berisi semua artifact model.

    Dipanggil saat lifespan startup, SEBELUM `load_lr() / load_xgb() / load_cv()`.
    """
    global _models_dir
    if _models_dir is not None:
        return _models_dir

    repo_id = os.environ.get("HF_HUB_REPO")

    if not repo_id:
        # Dev mode — pakai folder lokal
        local = Path(__file__).resolve().parent.parent / "model_artifacts"
        if not local.exists():
            logger.warning(
                f"[models] Local artifacts dir not found: {local}. "
                f"Loaders will fall back to heuristic/mock mode."
            )
        logger.info(f"[models] Using local path: {local}")
        _models_dir = local
        return _models_dir

    # Production mode — download dari Hub
    from huggingface_hub import snapshot_download

    logger.info(f"[models] Downloading from HF Hub: {repo_id}")
    _models_dir = Path(
        snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            cache_dir=str(_CACHE_ROOT),
            token=os.environ.get("HF_TOKEN"),
            allow_patterns=[
                "*.pkl",
                "*.onnx",
                "*.onnx.data",
                "*_metadata.json",
            ],
        )
    )
    logger.info(f"[models] Cached at: {_models_dir}")
    return _models_dir


def ensure_runtime_data() -> Path:
    """
    Pastiin runtime data (chatbot KB JSON + nutrition DB CSV) tersedia di disk.

    Di dev: return path ke folder `datasets/` di project root.
    Di deploy: download dari HF Hub Dataset repo ke `/app/data/{kb,nutrition}/...`.

    Dipanggil saat lifespan startup. Setelah return, set env var `MATERIN_DATA_DIR`
    supaya agents (chatbot, nutrition_parser) bisa resolve path-nya.
    """
    global _data_dir
    if _data_dir is not None:
        return _data_dir

    repo_id = os.environ.get("HF_HUB_DATA_REPO")

    if not repo_id:
        # Dev mode
        if not _LOCAL_DATA_ROOT.exists():
            logger.warning(
                f"[data] Local datasets dir not found: {_LOCAL_DATA_ROOT}. "
                f"Chatbot akan fallback ke static text, nutrition akan fallback ke parser rule."
            )
        logger.info(f"[data] Using local path: {_LOCAL_DATA_ROOT}")
        _data_dir = _LOCAL_DATA_ROOT
        return _data_dir

    # Production mode — download dari Hub, repackage ke flat layout
    from huggingface_hub import snapshot_download

    logger.info(f"[data] Downloading from HF Hub: {repo_id}")
    tmp = Path(
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            cache_dir=str(_CACHE_ROOT),
            token=os.environ.get("HF_TOKEN"),
            allow_patterns=["**/*.json", "**/*.csv"],
        )
    )

    _data_dir = Path(os.environ.get("MATERIN_DATA_DIR", "/app/data"))
    _data_dir.mkdir(parents=True, exist_ok=True)

    # Repackage: <repo>/kb/*.json → <data_dir>/kb/*.json, dst.
    _copy_data_files(tmp, _data_dir)
    logger.info(f"[data] Repackaged to: {_data_dir}")
    return _data_dir


def _copy_data_files(src_root: Path, dst_root: Path) -> None:
    """
    Copy data files dari HF Hub cache ke flat layout di dst_root.

    Asumsi struktur upload:
      <repo>/kb/<file>.json
      <repo>/nutrition/<file>.csv
    """
    for sub in ("kb", "nutrition"):
        out_dir = dst_root / sub
        out_dir.mkdir(exist_ok=True)
        for f in src_root.glob(f"{sub}/*"):
            if f.is_file():
                shutil.copy(f, out_dir / f.name)
                logger.info(f"[data]   {sub}/{f.name}")


def reset_cache() -> None:
    """
    Reset cached path. Untuk testing only — paksa re-download di pemanggilan berikutnya.
    """
    global _models_dir, _data_dir
    _models_dir = None
    _data_dir = None