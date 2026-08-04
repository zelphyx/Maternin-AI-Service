# ──────────────────────────────────────────────────────────────────────
# MaternIn AI Service — Dockerfile (Dokploy / generic Docker)
# ──────────────────────────────────────────────────────────────────────
# Context path = repo root (Maternin-AI/).
# Build trigger: `docker build -f Dockerfile .`
#
# Structure assumed:
#   .
#   ├── Dockerfile           ← you are here
#   ├── .dockerignore
#   ├── ai-service/
#   │   ├── requirements.txt
#   │   ├── conftest.py
#   │   └── app/             ← application source
#   └── datasets/            ← runtime data (chatbot KB + nutrition CSV)
#
# Note: model artifacts (~125 MB) are NOT bundled. They download from
# Hugging Face Hub on first container start via `huggingface_hub`.
# ──────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# System deps for OpenCV, MediaPipe, ONNX Runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        ffmpeg \
        git \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Python dependencies (cached layer) ──────────────────────────────────
COPY ai-service/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir "huggingface_hub>=0.24.0,<1.0.0"

# ── Application source ──────────────────────────────────────────────────
COPY ai-service/app/ ./app/
COPY ai-service/conftest.py ./conftest.py

# ── Runtime configuration ───────────────────────────────────────────────
ENV OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2 \
    PYTHONUNBUFFERED=1 \
    LOG_LEVEL=info \
    HF_HUB_CACHE=/app/.cache/hub \
    MATERIN_DATA_DIR=/app/data

EXPOSE 7860

# Health check — Dokploy / load balancer pings this
HEALTHCHECK --interval=60s --timeout=5s --retries=3 \
    CMD curl -fsS http://localhost:7860/health || exit 1

# Single worker is intentional: avoids GIL contention on small VPS.
# ${PORT:-7860} → Dokploy sets $PORT; HF Space / local uses 7860.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860} --workers 1 --timeout-keep-alive 60"]