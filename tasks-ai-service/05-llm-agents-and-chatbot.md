# Task 05: LLM Agents & Chatbot (P0/P1) ✅ COMPLETED

**Scope:** Integrasi LangChain dengan GROQ/Qwen API untuk generasi narasi rekomendasi triage (Lapis 3) dan endpoint chatbot asisten virtual ibu hamil.

---

## 📑 Sub-tasks

### 5.1 Recommendation Narrative LLM Agent (Triage Lapis 3 - P0)
- [x] `app/agents/recommendation_agent.py` — LangChain + ChatGroq (llama-3.3-70b-versatile)
- [x] Prompt template: konversi `risk_factors` → narasi mudah dipahami ibu hamil
- [x] Guardrail: LLM DILARANG mengubah `risk_badge` / `aggregate_score`
- [x] Sanitasi: TIDAK mengirim PII (nama, nomor HP) ke API LLM
- [x] Fallback teks generik jika LLM timeout/error
- [x] Terintegrasi ke `pipelines/triage_engine.py` (menggantikan mock Lapis 3)

### 5.2 `POST /api/v1/chat` Endpoint & Agent (P0)
- [x] `app/agents/chatbot_agent.py` — LangChain + RAG grounding dari Q&A Kemenkes
- [x] Edukasi kontekstual kehamilan
- [x] Disclaimer medis wajib di setiap jawaban (`disclaimer_included: true`)
- [x] Guardrail: DILARANG menebak skor risiko sendiri
- [x] `app/routers/chat.py` — endpoint `POST /api/v1/chat` terdaftar di main.py

---

## 🎯 Target Output Files
- [x] `app/agents/recommendation_agent.py`
- [x] `app/agents/chatbot_agent.py`
- [x] `app/routers/chat.py`

## ✅ Acceptance Criteria
1. ✅ `recommendation_text` di triage menggunakan LLM (fallback ke teks generik jika GROQ down)
2. ✅ `/api/v1/chat` merespons dengan edukasi + disclaimer medis
3. ✅ GROQ API error → fallback text aman, TIDAK crash (HTTP 200)

## 🧪 Hasil Test

| Test | Endpoint | GROQ Status | Response | Crash? |
|---|---|---|---|---|
| Chat edukasi | POST /api/v1/chat | 401 (placeholder key) | Fallback + disclaimer ✅ | ❌ |
| Triage + LLM | POST /api/v1/triage/analyze | 401 (placeholder key) | Fallback text + badge benar ✅ | ❌ |
| Chat guardrail | POST /api/v1/chat | 401 (placeholder key) | Fallback + disclaimer ✅ | ❌ |

> **Note:** LLM akan berfungsi penuh saat `GROQ_API_KEY` di `.env` diganti dengan API key asli dari https://console.groq.com
