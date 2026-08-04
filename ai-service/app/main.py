"""
MaternIn AI Service — Main Entrypoint
=======================================
Register routers, middleware, dan lifespan events.
"""

import logging
import os
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.config import settings

# ── Logging ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("maternin.ai")


# ── Lifespan: startup & shutdown events ──────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load model artifacts ke memori pada startup.
    Cleanup saat shutdown.
    """
    import time

    logger.info("🚀 MaternIn AI Service starting up...")
    logger.info(f"   Model artifact dir: {settings.model_artifact_dir}")
    logger.info(f"   NestJS base URL:    {settings.nestjs_internal_base_url}")
    logger.info(f"   Log level:          {settings.log_level}")

    # ── Ensure artifacts available BEFORE loading models ─────────────
    # In HF Space: download from Hub. In dev: use local folder.
    # MUST run before load_lr/load_xgb/load_cv — else loaders fall back
    # to heuristic/mock mode silently.
    t0 = time.time()

    from app.core.artifact_loader import ensure_model_artifacts, ensure_runtime_data
    ensure_model_artifacts()
    data_dir = ensure_runtime_data()
    os.environ["MATERIN_DATA_DIR"] = str(data_dir)
    logger.info(f"   Runtime data dir:   {data_dir}")

    # ── Load ML models ke memori ─────────────────────────────────────
    from app.models.preeclampsia_lr.inference import load_model as load_lr
    from app.models.risk_aggregator_xgb.inference import load_model as load_xgb
    from app.models.anemia_cv.inference import load_model as load_cv

    logger.info("Loading ML models...")
    load_lr()
    load_xgb()
    load_cv()

    elapsed = time.time() - t0
    logger.info(f"✅ All models loaded in {elapsed:.2f}s")

    yield

    logger.info("🛑 MaternIn AI Service shutting down...")


# ── FastAPI App ──────────────────────────────────────────────────────────
app = FastAPI(
    title="MaternIn AI Service",
    description=(
        "Layanan AI untuk deteksi dini risiko kehamilan (perdarahan, infeksi, preeklampsia) "
        "dan pemantauan ibu hamil. Bagian dari ekosistem MaternIn — Tim IRICH, GEMASTIK XIX."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS (izinkan NestJS backend memanggil) ──────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Di produksi: ganti dengan domain NestJS spesifik
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Middleware: X-Request-Id Tracing ─────────────────────────────────────
@app.middleware("http")
async def request_id_middleware(request: Request, call_next) -> Response:
    """
    Pastikan setiap request punya X-Request-Id untuk tracing lintas service.
    Jika NestJS tidak mengirimkan, generate UUID baru.
    """
    request_id = request.headers.get("X-Request-Id") or str(uuid.uuid4())
    # Simpan ke request state agar bisa diakses di dependency/handler
    request.state.request_id = request_id

    response: Response = await call_next(request)
    response.headers["X-Request-Id"] = request_id
    return response


# ── Global Exception Handler ────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Tangkap semua exception yang tidak tertangani.
    JANGAN bocorkan stack trace, path file model, atau isi env var.
    """
    request_id = getattr(request.state, "request_id", "unknown")
    logger.error(f"Unhandled error [request_id={request_id}]: {type(exc).__name__}: {exc}")

    return JSONResponse(
        status_code=500,
        content={
            "detail": "Terjadi kesalahan internal pada AI Service. Silakan coba lagi.",
            "request_id": request_id,
        },
    )


# ── Health Check ─────────────────────────────────────────────────────────
@app.get("/health", tags=["System"])
async def health_check():
    """Health check endpoint — digunakan oleh load balancer / monitoring."""
    return {"status": "ok", "service": "maternin-ai"}


# ── Register Routers ─────────────────────────────────────────────────────
from app.routers.triage import router as triage_router
from app.routers.chat import router as chat_router
from app.routers.postpartum import router as postpartum_router
from app.routers.trend import router as trend_router
from app.routers.visit_brief import router as visit_brief_router
from app.routers.nutrition import router as nutrition_router

app.include_router(triage_router)
app.include_router(chat_router)
app.include_router(postpartum_router)
app.include_router(trend_router)
app.include_router(visit_brief_router)
app.include_router(nutrition_router)

