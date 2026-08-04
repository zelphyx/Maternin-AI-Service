# ──────────────────────────────────────────────────────────────────────
# MaternIn AI Service — Dockerfile (Hugging Face Space, Docker SDK)
# ──────────────────────────────────────────────────────────────────────
# Multi-stage plan:
#   1. python:3.11-slim base + system deps (libgl, ffmpeg for OpenCV/MediaPipe)
#   2. Install Python deps in cached layer (only re-run if requirements.txt changes)
#   3. Copy source code (model artifacts are downloaded from HF Hub at runtime)
#
# Build context = ai-service/ (the Dockerfile sits at the ai-service root).
# Push mechanism: `git push hf main` from ai-service/ → builds Space automatically.
# ──────────────────────────────────────────────────────────────────────

FROM python:3.11-slim AS base

# System dependencies for OpenCV, MediaPipe, ONNX Runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        ffmpeg \
        git \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Python dependencies (cached layer) ──────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir "huggingface_hub>=0.24.0,<1.0.0"

# ── Application code ────────────────────────────────────────────────────
COPY app/ ./app/
COPY conftest.py ./

# ── Runtime configuration ───────────────────────────────────────────────
ENV OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2 \
    PYTHONUNBUFFERED=1 \
    LOG_LEVEL=info \
    HF_HUB_CACHE=/app/.cache/hub \
    MATERIN_DATA_DIR=/app/data

# Hugging Face Spaces expects port 7860
EXPOSE 7860

# Lightweight health check — uvicorn process stays responsive
HEALTHCHECK --interval=60s --timeout=5s --retries=3 \
    CMD curl -fsS http://localhost:7860/health || exit 1

# Single worker is intentional: HF Space CPU basic = 2 vCPU shared, multiple
# workers + GIL contention hurts more than it helps. concurrency comes from
# async/await inside uvicorn event loop (FastAPI handlers).
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port 7860 --workers 1 --timeout-keep-alive 60"]
