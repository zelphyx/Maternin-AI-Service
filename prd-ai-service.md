# PRD AI Service — FastAPI (MaternIn)

**Tim IRICH — GEMASTIK XIX**
Versi 1.0 — Turunan dari `PRD_Backend_MaternIn.md` (master), di-scope khusus buat sesi coding FastAPI di folder `/ai-service`.

---

## 0. Ruang lingkup dokumen ini

**Dokumen ini HANYA mencakup implementasi FastAPI di folder `/ai-service`.**

NestJS (`/backend`) diperlakukan sebagai **dependency eksternal** — dia satu-satunya pemilik database. AI Service **tidak boleh** nulis langsung ke PostgreSQL manapun; semua hasil kalkulasi wajib dikirim balik lewat endpoint internal NestJS (lihat section 6). Kalau nemu kebutuhan field baru, kesepakatan tetap harus disinkronkan dulu ke `PRD_Backend_MaternIn.md` (master) sebelum diimplementasikan di sini.

Prompt buat mulai scaffolding:
```
Baca PRD_AI_Service_MaternIn.md ini secara penuh. Aku cuma ngerjain sisi
FastAPI di folder /ai-service. Anggap NestJS di /backend itu sudah ada dan
berjalan sesuai kontrak internal di section 6 — jangan implementasikan apa
pun di dalamnya, cukup panggil callback-nya sesuai kontrak itu. Bantu aku
setup skeleton project FastAPI sesuai struktur module di section 3.
```

---

## 1. Tech stack (FastAPI)

| Layer | Teknologi |
|---|---|
| Framework | FastAPI + Uvicorn (async) |
| Validasi request/response | Pydantic v2 |
| CV — landmark & auto-crop ROI | MediaPipe Face Mesh (pre-built, bukan model diagnosis) |
| CV — deteksi anemia | MobileNetV3-Small, custom-trained (PyTorch atau TF Lite untuk kompresi mobile-latency) |
| Deteksi preeklampsia | Logistic Regression, custom-trained (scikit-learn) |
| Aggregator risiko | XGBoost, custom-trained (ensemble 3 skor input) |
| Triage engine (lapis 1) | Rule-based weighted scoring, in-house, threshold PNPK Obstetri — **bukan ML**, harus transparan & bisa diaudit |
| Chatbot & narasi rekomendasi | LangChain agent + LLM (GROQ / Qwen API) |
| NLP parser laporan makan | LLM (GROQ / Qwen) dengan structured output (JSON) |
| Queue/async job (opsional) | Celery + Redis, kalau inferensi CV butuh diproses di background |
| WhatsApp (darurat saja) | Fonnte API, dipanggil langsung dari AI Service saat `risk_badge == "merah"` |
| Auth antar-service | Header `X-Internal-Token` (shared secret dengan NestJS) |

**Konvensi:** field JSON request/response pakai `snake_case`, konsisten sama skema NestJS. Semua timestamp UTC. Setiap request wajib bawa `X-Request-Id` yang sama dengan yang dikirim NestJS, untuk tracing lintas service.

---

## 2. Peran AI Service dalam sistem

AI Service **cuma menghitung dan mengembalikan hasil** — bukan pemilik data. Dua mode respons:

```
NestJS -> AI Service (sync, timeout 5 detik dari sisi NestJS)
   <- AI Service balikin hasil langsung (kalau inferensi cepat, mis. rule + LR + XGBoost)

ATAU (kalau inferensi CV lebih berat / butuh antrian)

NestJS -> AI Service (terima "accepted", proses async)
   AI Service -> callback ke endpoint internal NestJS (section 6) begitu selesai
```

Pembagian tanggung jawab notifikasi WhatsApp:
- **Alert darurat** (risiko Merah) → dikirim **langsung dari AI Service** lewat Fonnte, begitu `risk_badge == "merah"` ditentukan. AI Service tetap wajib callback hasil ke NestJS buat disimpan — NestJS tidak mengirim WA darurat duplikat.
- **Reminder terjadwal** → tanggung jawab NestJS (di luar scope dokumen ini).

---

## 3. Struktur module FastAPI

```
/ai-service
  /app
    /main.py              -> entrypoint, register router, middleware
    /routers
      triage.py            -> POST /api/v1/triage/analyze
      postpartum.py         -> POST /api/v1/postpartum/evaluate
      chat.py               -> POST /api/v1/chat
      trend.py              -> POST /api/v1/trend/predict
      visit_brief.py        -> POST /api/v1/visit-brief/generate
      nutrition.py          -> POST /api/v1/nutrition/parse
    /models
      triage_rules.py       -> rule-based weighted scoring (lapis 1)
      anemia_cv/            -> MobileNetV3 inference wrapper + preprocessing
      preeclampsia_lr/       -> Logistic Regression inference wrapper
      risk_aggregator_xgb/   -> XGBoost inference wrapper
      landmark_roi.py        -> MediaPipe Face Mesh, auto-crop ROI konjungtiva
    /agents
      chatbot_agent.py       -> LangChain agent, edukasi + instruksi mitigasi
      nutrition_parser.py    -> NLP ekstraksi gizi dari teks bebas WA
      visit_brief_agent.py    -> ringkasan 2-3 kalimat riwayat + red flag
    /services
      nestjs_client.py        -> HTTP client buat callback ke endpoint internal NestJS
      whatsapp_client.py       -> Fonnte client, khusus alert darurat
    /schemas
      -> Pydantic request/response models per endpoint
    /core
      config.py              -> env var loading + validasi (pydantic-settings)
      auth.py                -> validasi X-Internal-Token masuk
      explainability.py       -> util buat rangkai risk_factors yang bisa ditelusuri
    /training                 -> notebook/script training terpisah dari runtime inference
      anemia_cv_train.py
      preeclampsia_lr_train.py
      risk_aggregator_train.py
    /model_artifacts           -> file model terkompresi (.pkl/.onnx/.tflite), versioned
```

---

## 4. Endpoint yang diimplementasikan di AI Service

Kontrak berikut **wajib diikuti persis** — NestJS (dikerjakan orang lain di tim) sudah menulis kode pemanggilnya berdasarkan format ini. Semua endpoint di section ini menerima header `X-Internal-Token` dan `X-Request-Id` dari NestJS.

### 4.1 `POST /api/v1/triage/analyze` — P0

Request:
```json
{
  "pregnancy_profile_id": "uuid",
  "symptom_checkin_id": "uuid",
  "answers": { "bengkak_kaki": true, "sakit_kepala": "berat", "pandangan_kabur": false },
  "conjunctiva_image_url": "https://...",
  "latest_anc": { "systolic": 145, "diastolic": 95, "protein_urine": "positif" },
  "has_preeclampsia_history": false
}
```

Pipeline internal: rule-based triage (lapis 1) → CV anemia + LR preeklampsia (lapis inferensi paralel) → XGBoost aggregator (lapis 2) → narasi rekomendasi via LLM (lapis 3).

Response:
```json
{
  "risk_badge": "merah",
  "aggregate_score": 84,
  "risk_factors": ["Tekanan darah tinggi (145/95)", "Sakit kepala hebat", "Protein urine positif"],
  "recommendation_text": "..."
}
```

Kalau `risk_badge == "merah"`: AI Service trigger `whatsapp_client` (Fonnte) ke bidan wilayah pasien **sebelum atau bersamaan** dengan mengembalikan response ke NestJS — jangan sampai delay LLM narasi bikin alert darurat telat kirim.

### 4.2 `POST /api/v1/postpartum/evaluate` — P1

Request: data dari `postpartum_logs` + `had_preeclampsia_history` dari profil.
Response:
```json
{ "red_flag_triggered": true, "reason": "Perdarahan banyak + sakit kepala hebat", "mental_health_flag": false }
```

`mental_health_flag` dihitung dari pola `mood_flag` (mis. `sering_sedih` berturut-turut) — bukan diagnosis, cuma penanda buat bidan tindak lanjuti.

### 4.3 `POST /api/v1/chat` — P0

Request: `{ "pregnancy_profile_id": "uuid", "message": "..." }`
Response: `{ "reply": "...", "disclaimer_included": true }`

Wajib: setiap jawaban chatbot yang menyentuh gejala/kondisi medis harus menyertakan disclaimer "bukan pengganti tenaga medis", ditandai lewat `disclaimer_included: true`. Chatbot LLM tidak boleh mengeluarkan angka risk score sendiri — kalau user tanya soal skor risiko, agent wajib ambil dari `risk_assessments` terakhir (via NestJS), bukan menghitung ulang atau menebak.

### 4.4 `POST /api/v1/trend/predict` — P1 (baru, untuk fitur trend-based early warning)

Request:
```json
{
  "pregnancy_profile_id": "uuid",
  "score_history": [
    { "aggregate_score": 45, "created_at": "2026-07-10T00:00:00Z" },
    { "aggregate_score": 58, "created_at": "2026-07-15T00:00:00Z" },
    { "aggregate_score": 66, "created_at": "2026-07-20T00:00:00Z" }
  ]
}
```
Response:
```json
{
  "trend_direction": "naik",
  "predicted_badge_in_days": 5,
  "predicted_badge": "merah",
  "confidence_note": "Berdasarkan 3 titik data, interpretasi tetap perlu validasi bidan"
}
```
Ini sinyal prediktif sederhana (regresi tren linear/exponential smoothing sederhana), **bukan** model kompleks baru — cukup pakai histori `aggregate_score` yang sudah ada, jangan over-engineer di tahap kompetisi ini.

### 4.5 `POST /api/v1/visit-brief/generate` — P2 (baru, untuk auto-generated visit brief)

Request: riwayat ANC + risk_assessments + postpartum_logs terakhir pasien (dikirim NestJS sebagai payload, karena NestJS yang punya data).
Response:
```json
{ "brief_text": "Ringkasan 2-3 kalimat riwayat + red flag terbaru pasien..." }
```
Dibangun via LangChain agent yang cuma merangkum data yang dikirim — **tidak boleh** menambahkan interpretasi klinis baru di luar data yang diberikan (hindari halusinasi angka/gejala yang tidak ada di payload).

### 4.6 `POST /api/v1/nutrition/parse` — P2 (baru, untuk nutrition log NLP parser)

Request: `{ "pregnancy_profile_id": "uuid", "raw_message": "tadi pagi makan nasi telur sama sayur bayam" }`
Response:
```json
{
  "parsed_items": [{ "name": "nasi", "portion_estimate": "1 centong" }, { "name": "telur", "portion_estimate": "1 butir" }, { "name": "sayur bayam", "portion_estimate": "1 mangkuk kecil" }],
  "insight_text": "..."
}
```
Estimasi porsi & nilai gizi bersifat perkiraan kasar (bukan hasil lab) — `insight_text` wajib eksplisit menyebut ini estimasi, bukan angka presisi.

---

## 5. Model & pipeline — detail per komponen

| Komponen | Jenis model | Target metrik minimum | Catatan implementasi |
|---|---|---|---|
| Triage engine (lapis 1) | Rule-based weighted scoring | — (bukan ML, jadi tidak ada metrik akurasi) | Threshold hardcode mengacu PNPK, taruh di config terpisah biar gampang diaudit/diubah tanpa redeploy model |
| Deteksi anemia (CV) | MobileNetV3-Small | Ditentukan setelah retraining dataset lokal | Preprocessing wajib pakai hasil crop dari `landmark_roi.py`, bukan foto mentah |
| Deteksi preeklampsia | Logistic Regression | Akurasi 98%, presisi 100%, recall 100%, F1 99% (split 55:45) | Presisi/recall tinggi krusial — kalau retraining ulang, jangan turunkan recall demi akurasi umum |
| Aggregator risiko (lapis 2) | XGBoost | Akurasi 93%, presisi 93%, recall 94%, F1 93% | Input: skor triage + probabilitas anemia + probabilitas preeklampsia — pastikan `feature_importance` bisa diekstrak buat isi `risk_factors` |
| Face/eye landmark ROI | MediaPipe Face Mesh | — (bukan model diagnosis) | Hanya preprocessing/UX, aman pakai model umum |
| Narasi & chatbot (lapis 3) | LLM (GROQ/Qwen) via LangChain | — (bukan pengambil keputusan diagnosis) | Prompt wajib membatasi LLM cuma menjelaskan hasil, tidak mengoreksi/mengganti skor dari lapis 1-2 |

**Explainability wajib:** setiap `risk_factors` di response harus bisa ditelusuri balik ke nilai input klinis spesifik (bukan cuma label generik) — util `explainability.py` bertugas merangkai ini dari feature importance XGBoost + threshold rule triage.

---

## 6. Callback & komunikasi ke NestJS

AI Service memanggil endpoint internal NestJS berikut setelah selesai proses (terutama untuk mode async):

| Method | Path (di NestJS) | Kapan dipanggil |
|---|---|---|
| POST | `/internal/risk-assessments` | Setelah pipeline triage selesai — kirim hasil buat disimpan |
| POST | `/internal/postpartum-flags` | Setelah evaluasi rule postpartum selesai |

Wajib disertakan di setiap panggilan ke NestJS:
- Header `X-Internal-Token` (shared secret, sama dengan yang divalidasi NestJS)
- Header `X-Request-Id` yang sama dari request awal (tracing)
- Timeout eksplisit (mis. 5 detik) + retry terbatas (maks 3x, exponential backoff) — kalau NestJS down, log ke local queue/dead-letter, jangan silently drop hasil kalkulasi.

---

## 7. Alert WhatsApp darurat (tanggung jawab AI Service)

- Trigger: `risk_badge == "merah"` dari `/api/v1/triage/analyze`.
- Kirim via `whatsapp_client.py` (Fonnte) ke nomor bidan penanggung jawab wilayah pasien.
- Data nomor bidan/puskesmas **tidak disimpan di AI Service** — harus diambil dari payload yang dikirim NestJS di request awal, atau lewat endpoint read-only NestJS kalau perlu lookup tambahan (hindari AI Service punya salinan data pasien sendiri).
- Kalau pengiriman Fonnte gagal: retry maks 3x, kalau tetap gagal, tetap lanjutkan callback risk-assessment ke NestJS dengan flag tambahan (mis. `alert_delivery_status: "failed"`) biar NestJS bisa fallback (in-app notification), bukan silent fail.

---

## 8. Business rules relevan buat AI Service

### 8.1 Cadence reminder — bukan tanggung jawab AI Service

Nilai `risk_badge` cuma dikonsumsi NestJS untuk hitung cadence reminder (3/7/14 hari) — AI Service tidak perlu tahu/hitung ini, cukup pastikan `risk_badge` yang dikembalikan konsisten dengan tiga nilai enum: `hijau | kuning | merah`.

### 8.2 Sync offline (kader)

Kalau data masuk lewat `/sync/batch` NestJS, AI Service tetap dipanggil dengan kontrak yang **persis sama** seperti input langsung dari pasien (section 4.1/4.2) — AI Service tidak perlu tahu apakah request berasal dari sync offline atau input real-time.

---

## 9. Validasi & governance model

- Setiap output risiko tinggi harus bisa ditelusuri ke faktor klinis pemicunya (lihat section 5, explainability wajib).
- Cross-validation hasil model terhadap PNPK Obstetri Kemenkes RI dan standar ICD-MM WHO — dicek manual sebelum model artifact baru dipakai di production.
- Model versioning: setiap `model_artifacts` diberi nama dengan versi + tanggal training (mis. `xgboost_aggregator_v2_20260715.pkl`), jangan overwrite file lama.
- Rencana retraining berkala kalau ada data klinis baru — pipeline training (`/training`) terpisah total dari runtime inference, supaya training gagal/eksperimen tidak mengganggu service yang jalan.

---

## 10. Kebutuhan non-fungsional

- Latensi inferensi (triage + CV + LR + XGBoost, tanpa LLM narasi) target < 2 detik.
- LLM narasi/chatbot boleh streaming/async terpisah kalau perlu, supaya tidak memperlambat response utama risk badge.
- Model CV wajib terkompresi (TFLite/ONNX) untuk ukuran & kecepatan inferensi.
- Endpoint internal (dipanggil NestJS) maupun endpoint yang AI Service panggil ke NestJS, dua-duanya wajib tervalidasi token — tidak ada endpoint tanpa auth.
- Data yang dikirim ke LLM pihak ketiga (GROQ/Qwen) diminimalkan — hindari kirim identitas pasien (nama, nomor HP) ke prompt, cukup data klinis yang relevan untuk narasi.

---

## 11. Environment variables (scope AI Service)

```
GROQ_API_KEY=
INTERNAL_SERVICE_TOKEN=
NESTJS_INTERNAL_BASE_URL=
FONNTE_API_KEY=
MODEL_ARTIFACT_DIR=./app/model_artifacts
LOG_LEVEL=info
```

(`DATABASE_URL`, `JWT_SECRET`, `FONNTE_API_KEY` untuk reminder terjadwal itu milik `.env` NestJS, bukan tanggung jawab dokumen/env AI Service ini — kecuali `FONNTE_API_KEY` di atas yang memang dipakai khusus buat alert darurat dari sisi AI Service, pastikan kredensial ini terpisah dari punya NestJS kalau memungkinkan biar gampang di-rotate.)

---

## 12. Guardrails wajib (scope AI Service)

### 12.1 Keamanan credential
- **DILARANG** hardcode API key (GROQ, Fonnte) atau token apa pun di kode. Semua wajib lewat env var.
- `.env` wajib masuk `.gitignore`, cuma `.env.example` (placeholder kosong) yang boleh di-commit.
- Validasi env var wajib di startup (`pydantic-settings`), app gagal start dari awal kalau ada yang kosong — sama seperti prinsip Joi di NestJS.
- **DILARANG** log/print value secret, payload lengkap pasien, atau isi percakapan chatbot mentah ke console produksi.
- Endpoint AI Service wajib validasi `X-Internal-Token` masuk dari NestJS di setiap request — jangan asumsikan network internal otomatis aman.

### 12.2 Reliability & error handling
- Timeout eksplisit + retry terbatas di setiap panggilan keluar (ke NestJS callback, ke Fonnte, ke LLM API) — jangan biarin satu service lambat bikin seluruh request hang.
- Kalau LLM API (GROQ/Qwen) down/timeout: chatbot & narasi rekomendasi wajib ada fallback teks generik ("Detail lengkap sedang tidak bisa ditampilkan, silakan hubungi bidan"), bukan error mentah ke user.
- Global exception handler wajib pastikan response error tidak bocorin stack trace, path file model, atau isi env var.
- Model inference wajib dibungkus try-except — kalau satu komponen (mis. CV anemia) gagal, pipeline tetap bisa lanjut pakai komponen lain + catat di response bahwa satu skor tidak tersedia, jangan gagalkan seluruh triage.

### 12.3 Checklist sebelum fitur dianggap selesai
- [ ] Nggak ada credential hardcoded di kode
- [ ] Endpoint yang menerima panggilan dari NestJS udah validasi `X-Internal-Token`
- [ ] Panggilan keluar (NestJS callback, Fonnte, LLM) punya timeout + retry + fallback
- [ ] Model artifact versioned, tidak overwrite versi lama
- [ ] `risk_factors` yang dikembalikan bisa ditelusuri ke input klinis spesifik (explainability)
- [ ] Data sensitif pasien (nama, nomor HP) tidak ikut terkirim ke prompt LLM pihak ketiga

---

## 13. Urutan implementasi (scope AI Service)

1. Setup FastAPI skeleton + config/env validation + auth token middleware (fondasi)
2. Triage engine rule-based (lapis 1) — bisa jalan duluan tanpa nunggu model ML selesai dilatih
3. `POST /api/v1/triage/analyze` dengan mock/stub untuk CV & LR & XGBoost dulu (biar NestJS bisa mulai integrasi tanpa nunggu training selesai)
4. Training & integrasi model asli: Logistic Regression (preeklampsia) → XGBoost (aggregator) → MobileNetV3 (anemia, paling berat, taruh belakangan)
5. Callback client ke NestJS (`nestjs_client.py`) + endpoint internal target di section 6
6. WhatsApp client darurat (Fonnte) untuk `risk_badge == "merah"`
7. `POST /api/v1/postpartum/evaluate`
8. `POST /api/v1/chat` (LangChain agent + disclaimer wajib)
9. `POST /api/v1/trend/predict` (P1) → `POST /api/v1/visit-brief/generate` (P2) → `POST /api/v1/nutrition/parse` (P2), belakangan sesuai prioritas roadmap

Catatan: sama seperti sisi NestJS, kamu bisa mulai dari langkah 3 pakai mock response yang sesuai kontrak section 4.1 dulu, biar tim NestJS nggak keblok nunggu model asli selesai dilatih.
