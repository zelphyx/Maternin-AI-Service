# Task 07: Testing & Guardrails (P0)

**Scope:** Pengujian otomatis (pytest), verifikasi performa/latensi, serta audit keamanan dan reliability sebelum deployment.

---

## 📑 Sub-tasks

### 7.1 Automated Unit & Integration Testing (Pytest Suite)
- [ ] Buat file pengujian di folder `/ai-service/tests/`:
  - `tests/test_triage_rules.py`: Verifikasi logika aturan Lapis 1 (boundary value testing pada tensi sistolik/diastolik & protein urine).
  - `tests/test_ml_inference.py`: Verifikasi output model Logistic Regression & XGBoost Aggregator.
  - `tests/test_landmark_roi.py`: Verifikasi fungsi crop MediaPipe pada gambar valid dan invalid.
  - `tests/test_api_triage.py`: Integration test untuk `POST /api/v1/triage/analyze` dengan mock dan real model.
  - `tests/test_api_chat.py`: Verifikasi keberadaan disclaimer medis di respons chatbot.

### 7.2 Performance & Memory Optimization Check
- [ ] Pastikan seluruh model ML (`.pkl`, `.onnx`) dimuat ke memori pada event startup aplikasi (`lifespan`), bukan di-load ulang pada setiap HTTP request.
- [ ] Ukur waktu inferensi total pipeline triage (tanpa LLM): harus < 2 detik.
- [ ] Pastikan model CV menggunakan format ONNX/TFLite yang terkompresi.

### 7.3 Security, Privacy & Reliability Guardrails Audit
- [ ] Lakukan audit pencarian teks (*code search*) untuk memastikan **TIDAK ADA** credential, API Key, atau token yang di-hardcode di dalam kode.
- [ ] Verifikasi bahwa `.env` terdaftar di `.gitignore`.
- [ ] Pastikan endpoint publik/internal menolak request tanpa header `X-Internal-Token` yang valid.
- [ ] Pastikan data PII pasien (nama, nomor telp) disanitasi sebelum dikirim ke prompt LLM pihak ketiga.
- [ ] Verifikasi global exception handler tidak membocorkan stack trace mentah atau path file internal ke pengguna.

---

## 🎯 Target Output Files
- `tests/test_triage_rules.py`
- `tests/test_ml_inference.py`
- `tests/test_landmark_roi.py`
- `tests/test_api_triage.py`
- `tests/test_api_chat.py`

## ✅ Acceptance Criteria
1. Seluruh suite pengujian `pytest` lulus 100% tanpa error.
2. Latensi inferensi pipeline triage < 2 detik.
3. Bebas dari credential hardcode & seluruh endpoint tervalidasi token internal.
