# Task 01: Foundation & Setup (P0) ✅ COMPLETED

**Scope:** Setup struktur dasar proyek FastAPI, validasi environment variables, middleware autentikasi internal token & tracing, serta Pydantic schemas v2.

---

## 📑 Sub-tasks

### 1.1 Project Scaffolding & Config Validation
- [x] Buat struktur folder proyek `/ai-service`:
  ```text
  /ai-service
    /app
      /routers
      /models
      /agents
      /services
      /schemas
      /core
      /training
      /model_artifacts
  ```
- [x] Buat `app/core/config.py` menggunakan `pydantic-settings` untuk memuat dan memvalidasi `.env`:
  - `GROQ_API_KEY`: string (wajib)
  - `INTERNAL_SERVICE_TOKEN`: string (wajib, min length 32)
  - `NESTJS_INTERNAL_BASE_URL`: string uri (wajib)
  - `FONNTE_API_KEY`: string (wajib)
  - `MODEL_ARTIFACT_DIR`: string (default `./app/model_artifacts`)
  - `LOG_LEVEL`: string (default `info`)
- [x] Buat `.env.example` dengan placeholder variabel di atas.

### 1.2 Auth & Tracing Middleware
- [x] Buat `app/core/auth.py` untuk validasi header `X-Internal-Token` terhadap `INTERNAL_SERVICE_TOKEN`. Jika token salah/kosong, kembalikan HTTP 401 Unauthorized.
- [x] Buat middleware untuk mengekstrak `X-Request-Id` dari header masuk dan mempassing-nya ke konteks async request.

### 1.3 Pydantic Request & Response Schemas (Pydantic v2)
- [x] Buat `app/schemas/triage.py`:
  - `TriageAnalyzeRequest`: `pregnancy_profile_id`, `symptom_checkin_id`, `answers` (dict), `conjunctiva_image_url` (optional string), `latest_anc` (dict), `has_preeclampsia_history` (bool).
  - `TriageAnalyzeResponse`: `risk_badge` (enum: `hijau`,`kuning`,`merah`), `aggregate_score` (float/int), `risk_factors` (list of str), `recommendation_text` (str).
- [x] Buat `app/schemas/postpartum.py`:
  - `PostpartumEvaluateRequest` & `PostpartumEvaluateResponse`.
- [x] Buat `app/schemas/chat.py`:
  - `ChatRequest` (`pregnancy_profile_id`, `message`) & `ChatResponse` (`reply`, `disclaimer_included`).
- [x] Buat `app/schemas/nestjs_callback.py`:
  - Payload schemas untuk memanggil endpoint NestJS `/internal/risk-assessments` dan `/internal/postpartum-flags`.

---

## 🎯 Target Output Files
- [x] `app/core/config.py`
- [x] `app/core/auth.py`
- [x] `app/schemas/triage.py`
- [x] `app/schemas/postpartum.py`
- [x] `app/schemas/chat.py`
- [x] `app/schemas/nestjs_callback.py`
- [x] `.env.example`
- [x] `app/main.py` (bonus: entrypoint + health check + global exception handler)
- [x] `requirements.txt`
- [x] `app/schemas/trend.py` (bonus: P1 schema)
- [x] `app/schemas/visit_brief.py` (bonus: P2 schema)
- [x] `app/schemas/nutrition.py` (bonus: P2 schema)

## ✅ Acceptance Criteria
1. ✅ Aplikasi FastAPI gagal meluncur (*fail-fast*) jika ada variabel wajib di `.env` yang tidak terisi.
2. ✅ Endpoint yang diproteksi dengan `X-Internal-Token` menolak request tanpa token yang valid.
3. ✅ Semua skema Pydantic tervalidasi menggunakan `pydantic` v2 tanpa warning deprecation.
