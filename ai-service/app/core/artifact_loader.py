"""
MaternIn AI Service — Artifact Loader
======================================
Resolves paths to model artifacts and runtime data.

All files are bundled directly in the Docker image:
  - Model artifacts: /app/app/model_artifacts/
  - Runtime data:   /app/datasets/

In dev (local): falls back to repo-relative paths automatically.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger("maternin.ai.artifact_loader")

# Resolved once, cached for lifetime of process
_models_dir: Path | None = None
_data_dir: Path | None = None


def _is_container() -> bool:
    """True when running inside the Docker container (WORKDIR /app)."""
    return Path("/app").exists()


def ensure_model_artifacts() -> Path:
    """
    Return path to the model artifacts directory.

    Container: /app/app/model_artifacts/
    Dev:       <repo>/ai-service/app/model_artifacts/
    """
    global _models_dir
    if _models_dir is not None:
        return _models_dir

    if _is_container():
        _models_dir = Path("/app/app/model_artifacts")
    else:
        # Dev: relative to this file (ai-service/app/core/)
        _models_dir = Path(__file__).resolve().parent.parent / "model_artifacts"

    if not _models_dir.exists():
        logger.warning(
            f"[models] Artifacts not found: {_models_dir}. "
            "Using fallback heuristic / mock mode."
        )
    else:
        logger.info(f"[models] Using: {_models_dir}")
    return _models_dir


def ensure_runtime_data() -> Path:
    """
    Return path to the datasets directory (KB + nutrition CSV).

    Container: /app/datasets/
    Dev:       <repo>/datasets/
    """
    global _data_dir
    if _data_dir is not None:
        return _data_dir

    if _is_container():
        _data_dir = Path("/app/datasets")
    else:
        # Dev: ../../.. from ai-service/app/core/ → repo root → datasets/
        _data_dir = Path(__file__).resolve().parents[2] / "datasets"

    if not _data_dir.exists():
        logger.warning(
            f"[data] Datasets not found: {_data_dir}. "
            "Chatbot will use static fallback, nutrition parser will degrade."
        )
    else:
        logger.info(f"[data] Using: {_data_dir}")
    return _data_dir
