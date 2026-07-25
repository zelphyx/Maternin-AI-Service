# Task Index — AI Service (FastAPI)

**Tim IRICH — GEMASTIK XIX**
Index dari seluruh berkas task teknis untuk implementasi AI Service (`/ai-service`).

---

## 📋 Daftar File Task

| No | File Task | Modul / Scope | Prioritas | Status |
|---|---|---|---|---|
| 01 | [01-foundation-and-setup.md](file:///Users/zelphyx/Projects/Maternin-AI/tasks-ai-service/01-foundation-and-setup.md) | FastAPI Skeleton, Config, Auth, & Pydantic Schemas | **P0** | ⏳ Pending |
| 02 | [02-core-triage-engine-and-stubs.md](file:///Users/zelphyx/Projects/Maternin-AI/tasks-ai-service/02-core-triage-engine-and-stubs.md) | Rule-based Engine Lapis 1, Mock Model Wrappers, & Endpoint Triage | **P0** | ⏳ Pending |
| 03 | [03-service-callbacks-and-emergency-wa.md](file:///Users/zelphyx/Projects/Maternin-AI/tasks-ai-service/03-service-callbacks-and-emergency-wa.md) | Webhook Client NestJS (`/internal/*`) & WhatsApp Client Fonnte | **P0** | ⏳ Pending |
| 04 | [04-ml-inference-models-and-cv.md](file:///Users/zelphyx/Projects/Maternin-AI/tasks-ai-service/04-ml-inference-models-and-cv.md) | Logistic Regression, MediaPipe ROI, MobileNetV3, XGBoost & Explainability | **P0** | ⏳ Pending |
| 05 | [05-llm-agents-and-chatbot.md](file:///Users/zelphyx/Projects/Maternin-AI/tasks-ai-service/05-llm-agents-and-chatbot.md) | LangChain Narasi Rekomendasi & Endpoint Chatbot | **P0 / P1** | ⏳ Pending |
| 06 | [06-postpartum-and-extended-endpoints.md](file:///Users/zelphyx/Projects/Maternin-AI/tasks-ai-service/06-postpartum-and-extended-endpoints.md) | Endpoint Postpartum, Prediksi Tren, Visit Brief, & Nutrition Parser | **P1 / P2** | ⏳ Pending |
| 07 | [07-testing-and-guardrails.md](file:///Users/zelphyx/Projects/Maternin-AI/tasks-ai-service/07-testing-and-guardrails.md) | Pytest Suite, Performance Check, & Security Audit | **P0** | ⏳ Pending |

---

## 🚀 Panduan Eksekusi

1. Selesaikan **File 01 - 03** terlebih dahulu agar endpoint `/api/v1/triage/analyze` dapat merespons request NestJS secara mock (unblock integrasi lintas service).
2. Lanjutkan ke **File 04 & 05** untuk menyambungkan model Machine Learning asli dan agent LLM.
3. Selesaikan **File 06** sesuai prioritas P1 dan P2.
4. Lakukan verifikasi di **File 07** sebelum melakukan deployment.
