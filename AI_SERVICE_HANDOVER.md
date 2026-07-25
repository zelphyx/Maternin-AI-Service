# Dokumentasi Integrasi AI Service — MaternIn
## Handoff Document untuk Tim IRICH (GEMASTIK XIX)

**Dibuat:** 2026-07-24
**Versi AI Service:** 1.0.0
**Author:** AI Developer (FastAPI) & Backend Developer (NestJS)

---

## 1. Arsitektur Sistem

```
┌──────────────────────────────────────────────────────────────┐
│                      Frontend (Mobile App)                   │
└──────────────────────────────┬───────────────────────────────┘
                               │ HTTPS
                               ▼
┌──────────────────────────────────────────────────────────────┐
│                  NestJS Backend (Backend Developer)          │
│  - CRUD data pasien, ANC, checkin, postpartum logs          │
│  - Menyimpan hasil triage ke database                        │
│  - Men-trigger AI Service saat pasien submit checkin         │
│  - Endpoint: /internal/risk-assessments (callback target)   │
│  - Endpoint: /internal/postpartum-flags (callback target)    │
└──────┬─────────────────────────┬────────────────────────────┘
       │ Internal API (HTTPS)    │ Internal API (HTTPS)
       │ X-Internal-Token        │ X-Internal-Token
       ▼                        ▼
┌──────────────────────────────────────────────────────────────┐
│                  AI Service (AI Developer)                  │
│  FastAPI + Uvicorn — Port 8000                             │
│                                                              │
│  Endpoints yang NestJS panggil:                             │
│    POST /api/v1/triage/analyze    → Risk triage full pipeline│
│    POST /api/v1/chat              → Chatbot edukasi          │
│    POST /api/v1/postpartum/evaluate → Postpartum evaluation │
│    POST /api/v1/trend/predict     → Trend prediction        │
│    POST /api/v1/visit-brief/generate → Visit summary        │
│    POST /api/v1/nutrition/parse   → Food log parser         │
│                                                              │
│  Endpoints yang AI Service panggil balik ke NestJS:          │
│    POST {NESTJS_BASE_URL}/internal/risk-assessments          │
│    POST {NESTJS_BASE_URL}/internal/postpartum-flags          │
│                                                              │
│  External APIs:                                              │
│    - GROQ API → LLM chatbot + recommendation narrative      │
│    - Fonnte API → WhatsApp emergency alert (risk=merah)     │
└──────────────────────────────────────────────────────────────┘
```

---

## 2. Credential Sharing

### 2.1 Shared Token (WAJIB — Harus Sama di Kedua Sisi)

```
INTERNAL_SERVICE_TOKEN = "minimal-32-karakter-shared-secret"

Contoh:
INTERNAL_SERVICE_TOKEN = "maternin-ai-nestjs-shared-secret-2026-v1"
```

| Tempat | File | Variabel |
|--------|------|----------|
| AI Service (kamu) | `ai-service/.env` | `INTERNAL_SERVICE_TOKEN` |
| NestJS (temenmu) | `.env` NestJS | `INTERNAL_SERVICE_TOKEN` |

> ⚠️ **WAJIB SAMA PERSIS.** Kalau beda, semua request akan direject dengan 401 Unauthorized.

### 2.2 AI Service — Environment Variables

Tempat: `ai-service/.env`

```bash
# ── Credential (dari vendor API) ──────────────────────────────
GROQ_API_KEY=sk-xxxxx        # dari dashboard.groq.com
FONNTE_API_KEY=abc123        # dari fonnte.com (untuk WA darurat)

# ── Shared Auth (harus sama dengan NestJS .env) ───────────────
INTERNAL_SERVICE_TOKEN=minimal-32-karakter-shared-secret

# ── Backend URL (URL NestJS production — tanpa trailing slash) ─
NESTJS_INTERNAL_BASE_URL=https://api.maternin.id
# Contoh dev: NESTJS_INTERNAL_BASE_URL=http://localhost:3000

# ── Paths ────────────────────────────────────────────────────
MODEL_ARTIFACT_DIR=./app/model_artifacts
LOG_LEVEL=info
```

### 2.3 NestJS — Yang Harus Disiapkan Temenmu

Tempat: `.env` NestJS

```bash
# Shared token (WAJIB sama dengan AI Service .env)
INTERNAL_SERVICE_TOKEN=minimal-32-karakter-shared-secret
```

---

## 3. Kontrak API — NestJS Panggil AI Service

**Base URL AI Service:** `https://ai-service.maternin.id` (production) atau `http://localhost:8000` (dev)

**Header WAJIB di setiap request:**

| Header | Value | Mandatory |
|--------|-------|-----------|
| `X-Internal-Token` | Shared secret (sama dengan AI Service .env) | ✅ YA |
| `X-Request-Id` | UUID dari NestJS untuk tracing | ✅ YA (NESTJSSISIPKAN) |
| `Content-Type` | `application/json` | ✅ YA |

---

### 3.1 POST `/api/v1/triage/analyze` — Risk Triage (P0)

**Kapan dipanggil:** Saat pasien submit symptom checkin.

**Request:**
```json
{
  "pregnancy_profile_id": "550e8400-e29b-41d4-a716-446655440001",
  "symptom_checkin_id": "550e8400-e29b-41d4-a716-446655440002",
  "answers": {
    "bengkak_kaki": true,
    "sakit_kepala": "berat",
    "pandangan_kabur": false,
    "perdarahan": false,
    "demam_tinggi": false
  },
  "conjunctiva_image_url": "https://storage.maternin.id/images/conjunctiva/abc123.jpg",
  "latest_anc": {
    "systolic": 145,
    "diastolic": 95,
    "protein_urine": "positif"
  },
  "has_preeclampsia_history": false,
  "bidan_phone": "6281234567890"
}
```

**Response:**
```json
{
  "risk_badge": "kuning",
  "aggregate_score": 58.5,
  "risk_factors": [
    "Tekanan darah tinggi (145/95 mmHg)",
    "Sakit kepala hebat",
    "Protein urine positif"
  ],
  "recommendation_text": "⚡ Perhatian — Risiko Sedang (skor 58/100)...",
  "triage_score": 45.0,
  "anemia_probability": 0.32,
  "preeclampsia_probability": 0.65,
  "alert_delivery_status": "not_triggered"
}
```

**Catatan penting:**
- `risk_badge` hanya: `"hijau"` | `"kuning"` | `"merah"`
- `aggregate_score` range: 0–100
- `anemia_probability` bisa `null` kalau `conjunctiva_image_url` tidak dikirim
- `alert_delivery_status`: `"sent"` | `"failed"` | `"not_triggered"`
- Kalau `risk_badge == "merah"`, AI Service OTOMATIS kirim WhatsApp ke `bidan_phone`
- Response timeout target: **< 5 detik**

---

### 3.2 POST `/api/v1/chat` — Chatbot Edukasi (P0)

**Kapan dipanggil:** Saat pasien kirim pesan di chatbot in-app.

**Request:**
```json
{
  "pregnancy_profile_id": "550e8400-e29b-41d4-a716-446655440001",
  "message": "Halo, apa tanda-tanda bahaya kehamilan trimester pertama?"
}
```

**Response:**
```json
{
  "reply": "Halo! Berikut adalah tanda-tanda bahaya kehamilan trimester pertama...",
  "disclaimer_included": true
}
```

> ⚠️ `disclaimer_included` SELALU `true`. Reply chatbot SELALU menyertakan disclaimer bahwa ini edukasi, bukan pengganti nasihat medis.

---

### 3.3 POST `/api/v1/postpartum/evaluate` — Evaluasi Postpartum (P1)

**Request:**
```json
{
  "pregnancy_profile_id": "550e8400-e29b-41d4-a716-446655440001",
  "postpartum_logs": [
    {
      "log_date": "2026-07-20T00:00:00Z",
      "bleeding_level": "berat",
      "fever": false,
      "wound_infection": false,
      "severe_headache": false,
      "mood_flags": ["sering_sedih"]
    }
  ],
  "had_preeclampsia_history": true
}
```

**Response:**
```json
{
  "red_flag_triggered": true,
  "reason": "Perdarahan berat terdeteksi",
  "mental_health_flag": false
}
```

---

### 3.4 POST `/api/v1/trend/predict` — Prediksi Tren (P1)

**Request:**
```json
{
  "pregnancy_profile_id": "550e8400-e29b-41d4-a716-446655440001",
  "score_history": [
    { "aggregate_score": 45, "created_at": "2026-07-10T00:00:00Z" },
    { "aggregate_score": 58, "created_at": "2026-07-15T00:00:00Z" },
    { "aggregate_score": 66, "created_at": "2026-07-20T00:00:00Z" }
  ]
}
```

**Response:**
```json
{
  "trend_direction": "naik",
  "predicted_badge_in_days": 5,
  "predicted_badge": "merah",
  "confidence_note": "Berdasarkan 3 titik data, interpretasi tetap perlu validasi bidan"
}
```

---

### 3.5 POST `/api/v1/visit-brief/generate` — Ringkasan Kunjungan (P2)

**Request:**
```json
{
  "pregnancy_profile_id": "550e8400-e29b-41d4-a716-446655440001",
  "anc_history": [...],
  "risk_assessments": [...],
  "postpartum_logs": [...]
}
```

**Response:**
```json
{
  "brief_text": "Ibu G1P0 usia 28 tahun, trimester 2, dengan risiko kuning..."
}
```

---

### 3.6 POST `/api/v1/nutrition/parse` — Parser Log Makanan (P2)

**Request:**
```json
{
  "pregnancy_profile_id": "550e8400-e29b-41d4-a716-446655440001",
  "raw_message": "tadi pagi makan nasi telur sama sayur bayam"
}
```

**Response:**
```json
{
  "parsed_items": [
    { "name": "nasi", "portion_estimate": "1 centong" },
    { "name": "telur", "portion_estimate": "1 butir" },
    { "name": "sayur bayam", "portion_estimate": "1 mangkuk kecil" }
  ],
  "insight_text": "Porsi初步, perkiraan bukan hasil lab..."
}
```

---

## 4. Kontrak API — AI Service Callback ke NestJS

**AI Service memanggil endpoint ini SETELAH selesai memproses request.**

### 4.1 POST `/internal/risk-assessments` (di sisi NestJS)

**Header yang AI Service Kirim:**
```
X-Internal-Token: minimal-32-karakter-shared-secret
X-Request-Id: uuid-yang-sama-dari-request-awal
Content-Type: application/json
```

**Payload:**
```json
{
  "pregnancy_profile_id": "550e8400-e29b-41d4-a716-446655440001",
  "symptom_checkin_id": "550e8400-e29b-41d4-a716-446655440002",
  "triage_score": 45.0,
  "aggregate_score": 58.5,
  "risk_badge": "kuning",
  "risk_factors": [
    "Tekanan darah tinggi (145/95 mmHg)",
    "Sakit kepala hebat",
    "Protein urine positif"
  ],
  "preeclampsia_probability": 0.65,
  "anemia_probability": 0.32,
  "recommendation_text": "⚡ Perhatian — Risiko Sedang...",
  "alert_delivery_status": "not_triggered"
}
```

**Yang NestJS harus lakuin:**
- Simpan ke tabel `risk_assessments`
- Kalau `alert_delivery_status == "failed"`, NestJS bisa kirim notifikasi in-app sebagai fallback

---

### 4.2 POST `/internal/postpartum-flags` (di sisi NestJS)

**Payload:**
```json
{
  "pregnancy_profile_id": "550e8400-e29b-41d4-a716-446655440001",
  "red_flag_triggered": true,
  "reason": "Perdarahan berat terdeteksi",
  "mental_health_flag": false
}
```

**Yang NestJS harus lakuin:**
- Simpan ke tabel `postpartum_flags` atau update `postpartum_logs`

---

## 5. Perilaku Penting yang Harus Diketahui

### 5.1 WhatsApp Darurat — Otomatis dari AI Service

Kalau `risk_badge == "merah"`:
- AI Service langsung kirim WhatsApp ke `bidan_phone` yang dikirim NestJS di request triage
- Tidak perlu NestJS triggermanual — sudah otomatis
- Nomor bidan tidak disimpan di AI Service, hanya dipakai untuk kirim pesan

### 5.2 Timeout & Retry

| Arah | Timeout | Retry |
|------|---------|-------|
| NestJS → AI Service | NestJS handle timeout 5 detik | Di sisi NestJS |
| AI Service → NestJS (callback) | 5 detik | 3x dengan exponential backoff |
| AI Service → Fonnte (WA) | 10 detik | 3x dengan exponential backoff |
| AI Service → GROQ (LLM) | 15 detik (chatbot), 10 detik (recommendation) | 2x |

### 5.3 Fallback — Kalau LLM Down

Kalau GROQ API down/error:
- Chatbot reply: fallback teks generik + disclaimer medis
- Recommendation text: fallback teks generik sesuai risk_badge
- **Pipeline triage tetap jalan** — tidak depend pada LLM

### 5.4 Error Handling

Kalau AI Service error:
- NestJS akan terima HTTP 500
- Response body: `{"detail": "...", "request_id": "..."}` — tidak ada stack trace

---

## 6. Network & Firewall

### Dari Sisi AI Service (Outbound)
AI Service harus bisa reach:
- `{NESTJS_INTERNAL_BASE_URL}` → untuk callback
- `https://api.groq.cloud` → untuk LLM (GROQ API)
- `https://api.fonnte.com` → untuk WhatsApp darurat

### Dari Sisi NestJS (Outbound)
- `https://ai-service.maternin.id` → untuk panggil AI Service endpoints

---

## 7. Checklist Sebelum Go-Live

### AI Developer (Kamu) — Checklist
- [ ] `.env` terisi dengan credential asli (GROQ_API_KEY, FONNTE_API_KEY, INTERNAL_SERVICE_TOKEN)
- [ ] `NESTJS_INTERNAL_BASE_URL` menunjuk ke URL production NestJS
- [ ] INTERNAL_SERVICE_TOKEN sudah dibagikan ke backend developer
- [ ] Model `.pkl` sudah ada di `app/model_artifacts/`
- [ ] Docker image sudah build dan di-push ke registry
- [ ] Health check `/health` mengembalikan `{"status": "ok"}`

### Backend Developer (Temenmu) — Checklist
- [ ] Endpoint `POST /internal/risk-assessments` sudah ada dan bisa terima payload (lihat section 4.1)
- [ ] Endpoint `POST /internal/postpartum-flags` sudah ada dan bisa terima payload (lihat section 4.2)
- [ ] `INTERNAL_SERVICE_TOKEN` di `.env` sudah diset sama dengan AI Service
- [ ] Header `X-Internal-Token` + `X-Request-Id` sudah disisipkan di setiap request ke AI Service
- [ ] Allow outbound ke: `ai-service.maternin.id`, `api.groq.cloud`, `api.fonnte.com`
- [ ] Handle timeout 5 detik dari AI Service

---

## 8. Testing — Verifikasi Integrasi

### Test Manual (cURL)

**1. Test Health Check:**
```bash
curl -X GET https://ai-service.maternin.id/health
# Expected: {"status":"ok","service":"maternin-ai"}
```

**2. Test Triage (harus dapat 401 tanpa token):**
```bash
curl -X POST https://ai-service.maternin.id/api/v1/triage/analyze \
  -H "Content-Type: application/json" \
  -H "X-Internal-Token: minimal-32-karakter-shared-secret" \
  -H "X-Request-Id: test-123" \
  -d '{
    "pregnancy_profile_id": "test",
    "symptom_checkin_id": "test",
    "answers": {"bengkak_kaki": true},
    "latest_anc": {"systolic": 150, "diastolic": 95}
  }'
# Expected: HTTP 200, risk_badge="kuning"
```

**3. Test Chatbot:**
```bash
curl -X POST https://ai-service.maternin.id/api/v1/chat \
  -H "Content-Type: application/json" \
  -H "X-Internal-Token: minimal-32-karakter-shared-secret" \
  -d '{"pregnancy_profile_id": "test", "message": "Nutrisi untuk ibu hamil"}'
# Expected: HTTP 200, disclaimer_included=true
```

**4. Test Unauthorized:**
```bash
curl -X POST https://ai-service.maternin.id/api/v1/triage/analyze \
  -H "Content-Type: application/json" \
  -H "X-Internal-Token: wrong-token" \
  -d '{"pregnancy_profile_id": "test", "symptom_checkin_id": "test", "answers": {}}'
# Expected: HTTP 401
```

---

## 9. Run AI Service Secara Lokal

```bash
cd ai-service

# Install dependencies
pip install -r requirements.txt

# Buat .env (copy dari .env.example, isi credential)
cp .env.example .env
nano .env  # isi GROQ_API_KEY, FONNTE_API_KEY, INTERNAL_SERVICE_TOKEN, NESTJS_INTERNAL_BASE_URL

# Jalankan
uvicorn app.main:app --reload --port 8000

# Test health
curl http://localhost:8000/health
```

---

## 10. Deploy AI Service (Docker)

### Dockerfile
```dockerfile
FROM python:3.12-slim
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY app/ ./app/

# Environment variables diset saat runtime (bukan di-build)
# Pastikan .env tidak ikut ke image
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8000

# Start
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Build & Run
```bash
docker build -t maternin-ai-service:latest .
docker run -d \
  -p 8000:8000 \
  -e GROQ_API_KEY="sk-xxxxx" \
  -e FONNTE_API_KEY="abc123" \
  -e INTERNAL_SERVICE_TOKEN="minimal-32-karakter-shared-secret" \
  -e NESTJS_INTERNAL_BASE_URL="https://api.maternin.id" \
  -e LOG_LEVEL="info" \
  maternin-ai-service:latest
```

---

## 11. Kontak & Escalation

| Pertanyaan | Tangan |
|-----------|--------|
| Error 401 Unauthorized | Cek INTERNAL_SERVICE_TOKEN sama di kedua .env |
| AI Service timeout > 5 detik | Cek GROQ API status, atau skip LLM (triage tetap jalan) |
| WhatsApp darurat tidak terkirim | Cek FONNTE_API_KEY, cek saldo Fonnte, cek `alert_delivery_status` di response |
| Callback tidak sampai ke NestJS | Cek `NESTJS_INTERNAL_BASE_URL`, cek endpoint `/internal/*` di NestJS |
| Model anemia CV tidak jalan | Pastikan `.onnx` ada di `app/model_artifacts/` |

---

*Dokumen ini adalah living document — update sesuai kebutuhan tim saat development berlangsung.*
