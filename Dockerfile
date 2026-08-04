# ──────────────────────────────────────────────────────────────────────
# MaternIn AI Service — Dockerfile (Dokploy / generic Docker)
# ──────────────────────────────────────────────────────────────────────
# Context path = repo root (Maternin-AI/).
# Build: `docker build -f Dockerfile .`
#
# Structure:
#   .
#   ├── Dockerfile
#   ├── .dockerignore
#   ├── ai-service/
#   │   ├── requirements.txt
#   │   ├── conftest.py
#   │   └── app/
#   │       └── model_artifacts/   ← bundled (~125 MB)
#   └── datasets/                   ← bundled (KB + nutrition CSV)
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

# Python dependencies
COPY ai-service/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Application + bundled artifacts
COPY ai-service/app/ ./app/
COPY ai-service/conftest.py ./conftest.py
COPY datasets/ ./datasets/

ENV OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2 \
    PYTHONUNBUFFERED=1 \
    LOG_LEVEL=info

EXPOSE 7860

HEALTHCHECK --interval=60s --timeout=5s --retries=3 \
    CMD curl -fsS http://localhost:7860/health || exit 1

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860} --workers 1 --timeout-keep-alive 60"]
