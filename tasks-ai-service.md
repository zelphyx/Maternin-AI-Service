# Task Breakdown — AI Service (FastAPI)

**Tim IRICH — GEMASTIK XIX**
Versi 1.0 — Breakdown tugas teknis berdasarkan `prd-ai-service.md` untuk pengerjaan modul `/ai-service`.

---

## Task Checklist Overview

- [ ] **Phase 1: Foundation & Setup** (P0)
- [ ] **Phase 2: Core Triage Engine & Stubs** (P0)
- [ ] **Phase 3: Service Callbacks & Emergency WA** (P0)
- [ ] **Phase 4: ML Inference Models & CV** (P0)
- [ ] **Phase 5: LLM Agents & Chatbot** (P0 / P1)
- [ ] **Phase 6: Postpartum & Extended Endpoints** (P1 / P2)
- [ ] **Phase 7: Testing, Guardrails & Polish** (P0)

---

## Phase 1: Foundation & Setup (P0)

### Task 1.1: Project Scaffolding & Configuration
- [ ] Setup folder structure `/ai-service/app/` (`routers`, `models`, `agents`, `services`, `schemas`, `core`, `training`, `model_artifacts`).
- [ ] Create `app/core/config.py` using `pydantic-settings` to load and validate `.env` variables (`GROQ_API_KEY`, `INTERNAL_SERVICE_TOKEN`, `NESTJS_INTERNAL_BASE_URL`, `FONNTE_API_KEY`, `LOG_LEVEL`).
- [ ] Create `.env.example` with placeholder variables.

### Task 1.2: Authentication & Tracing Middleware
- [ ] Implement `app/core/auth.py` to validate header `X-Internal-Token` against `INTERNAL_SERVICE_TOKEN`.
- [ ] Implement middleware to extract/pass `X-Request-Id` across incoming and outgoing HTTP requests.

### Task 1.3: Request & Response Schemas (Pydantic v2)
- [ ] `app/schemas/triage.py`: Request & Response models for `/api/v1/triage/analyze`.
- [ ] `app/schemas/postpartum.py`: Request & Response models for `/api/v1/postpartum/evaluate`.
- [ ] `app/schemas/chat.py`: Request & Response models for `/api/v1/chat`.
- [ ] `app/schemas/trend.py`: Request & Response models for `/api/v1/trend/predict`.
- [ ] `app/schemas/visit_brief.py`: Request & Response models for `/api/v1/visit-brief/generate`.
- [ ] `app/schemas/nutrition.py`: Request & Response models for `/api/v1/nutrition/parse`.
- [ ] `app/schemas/nestjs_callback.py`: Payload schemas for NestJS internal endpoints.

---

## Phase 2: Core Triage Engine & Mock Stubs (P0)

### Task 2.1: Triage Engine Lapis 1 (Rule-Based Deterministic Scoring)
- [ ] Implement `app/models/triage_rules.py` with weighted scoring rules referring to PNPK Obstetri (e.g. severe bleeding, extreme BP values).
- [ ] Ensure rule output produces base risk flags and deterministic scores without LLM dependency.

### Task 2.2: Mock Model Wrappers & Pipeline Orchestration
- [ ] Create stub wrapper for Logistic Regression Preeclampsia (`app/models/preeclampsia_lr/`).
- [ ] Create stub wrapper for MobileNetV3 Anemia CV (`app/models/anemia_cv/`).
- [ ] Create stub wrapper for XGBoost Aggregator (`app/models/risk_aggregator_xgb/`).
- [ ] Implement orchestrator `app/pipelines/triage_engine.py` connecting Lapis 1 -> LR -> CV -> XGBoost Lapis 2.

### Task 2.3: `POST /api/v1/triage/analyze` Endpoint (Mock Mode)
- [ ] Implement router `app/routers/triage.py`.
- [ ] Return mock valid triage response (`risk_badge`, `aggregate_score`, `risk_factors`, `recommendation_text`).
- [ ] **Goal:** Unblock NestJS integration testing early.

---

## Phase 3: Service Callbacks & Emergency WA Alert (P0)

### Task 3.1: NestJS Internal Webhook Client
- [ ] Implement `app/services/nestjs_client.py` using `httpx` async client.
- [ ] Add 5-second timeout, exponential backoff retries (max 3x), and `X-Internal-Token` / `X-Request-Id` headers.
- [ ] Hook into pipeline to POST results to NestJS `/internal/risk-assessments`.

### Task 3.2: Emergency WhatsApp Alert Client (Fonnte)
- [ ] Implement `app/services/whatsapp_client.py`.
- [ ] Trigger async Fonnte API call directly from AI Service when `risk_badge == "merah"`.
- [ ] Add retry mechanism (max 3x) and log status (`alert_delivery_status`) to NestJS callback if delivery fails.

---

## Phase 4: ML Inference Models & Computer Vision (P0)

### Task 4.1: Logistic Regression Preeclampsia Model
- [ ] Create training/export script `app/training/preeclampsia_lr_train.py` (target: 98% accuracy, 100% precision/recall on split 55:45).
- [ ] Save trained artifact to `app/model_artifacts/preeclampsia_lr_v1.pkl`.
- [ ] Implement actual inference wrapper in `app/models/preeclampsia_lr/`.

### Task 4.2: MediaPipe Landmark & ROI Auto-Crop
- [ ] Implement `app/models/landmark_roi.py` using MediaPipe Face Mesh.
- [ ] Crop palpebral conjunctiva Region of Interest (ROI) from `conjunctiva_image_url`.
- [ ] Handle fallback if face/eye landmark detection fails on poorly lit images.

### Task 4.3: MobileNetV3 Anemia CV Model
- [ ] Create preprocessing and inference wrapper in `app/models/anemia_cv/` (convert to ONNX/TFLite for low latency).
- [ ] Create training script `app/training/anemia_cv_train.py` for fine-tuning with local conjunctiva dataset.

### Task 4.4: XGBoost Risk Aggregator & Explainability Unit
- [ ] Create training/export script `app/training/risk_aggregator_train.py`.
- [ ] Save artifact to `app/model_artifacts/risk_aggregator_v1.pkl`.
- [ ] Implement `app/core/explainability.py` to extract feature importance and format human-readable, traceable `risk_factors`.

---

## Phase 5: LLM Agents & Chatbot (P0 / P1)

### Task 5.1: Recommendation Narrative LLM Agent (Triage Lapis 3 - P0)
- [ ] Implement `app/agents/recommendation_agent.py` using LangChain + GROQ / Qwen API.
- [ ] Transform `risk_factors` and clinical metrics into empathetic `recommendation_text`.
- [ ] Sanitize patient PII (strip names/phone numbers) before passing to third-party LLM API.
- [ ] Implement generic fallback text if LLM API times out.

### Task 5.2: `POST /api/v1/chat` Endpoint & Agent (P0)
- [ ] Implement `app/agents/chatbot_agent.py` using LangChain.
- [ ] Enforce mandatory medical disclaimer guardrail (`disclaimer_included: true`).
- [ ] Prevent LLM from inventing risk scores (must query NestJS `risk_assessments` if user asks for score).
- [ ] Create router `app/routers/chat.py`.

---

## Phase 6: Postpartum & Extended Endpoints (P1 / P2)

### Task 6.1: `POST /api/v1/postpartum/evaluate` Endpoint (P1)
- [ ] Create router `app/routers/postpartum.py`.
- [ ] Evaluate secondary hemorrhage, wound infection, fever, and `mood_flag` patterns (mental health red flag).
- [ ] Trigger callback to NestJS `/internal/postpartum-flags`.

### Task 6.2: `POST /api/v1/trend/predict` Endpoint (P1)
- [ ] Create router `app/routers/trend.py`.
- [ ] Implement linear regression / exponential smoothing over historical `aggregate_score` points.
- [ ] Return `trend_direction` and `predicted_badge_in_days`.

### Task 6.3: `POST /api/v1/visit-brief/generate` Endpoint (P2)
- [ ] Create router `app/routers/visit_brief.py` & agent `app/agents/visit_brief_agent.py`.
- [ ] Summarize 2-3 sentences of ANC history and active red flags for midwives without hallucinating extra clinical facts.

### Task 6.4: `POST /api/v1/nutrition/parse` Endpoint (P2)
- [ ] Create router `app/routers/nutrition.py` & agent `app/agents/nutrition_parser.py`.
- [ ] Extract structured JSON (food items, estimated portions) from freeform WhatsApp message.

---

## Phase 7: Testing, Guardrails & Production Polish (P0)

### Task 7.1: Model & Pipeline Unit / Integration Tests
- [ ] Write pytest tests for Triage Lapis 1 rules.
- [ ] Write pytest tests for Preeclampsia LR & XGBoost Aggregator inference wrappers.
- [ ] Write pytest tests for MediaPipe ROI cropper.
- [ ] Write pytest API integration tests for `/api/v1/triage/analyze` and `/api/v1/chat`.

### Task 7.2: Security & Reliability Audit
- [ ] Ensure 0 hardcoded credentials in codebase (`.env` validation on startup).
- [ ] Ensure all endpoints validate `X-Internal-Token`.
- [ ] Confirm models are pre-loaded at startup (`lifespan` event) to avoid per-request load overhead.
- [ ] Verify global exception handler hides stack traces and internal file paths.
