# Task 02: Core Triage Engine & Stubs (P0) ✅ COMPLETED

**Scope:** Implementasi Triage Engine Lapis 1 deterministik (rule-based), wrapper mock/stub untuk model ML, dan pembuatan endpoint `POST /api/v1/triage/analyze` mode mock untuk membendung ketergantungan NestJS.

---

## 📑 Sub-tasks

### 2.1 Triage Engine Lapis 1 (Rule-Based Engine)
- [x] Buat `app/models/triage_rules.py`:
  - Implementasikan fungsi scoring berbasis PNPK Obstetri Kemenkes RI.
  - Periksa parameter bahaya mutlak (misal: perdarahan hebat, tensi sistolik ≥ 160 atau diastolik ≥ 110, protein urine positif tinggi).
  - Hasilkan skor dasar lapis 1 dan daftar bendera merah (*red flags*).

### 2.2 Mock Model Wrappers & Pipeline Integration
- [x] Buat `app/models/preeclampsia_lr/inference.py` wrapper dengan fungsi `predict_preeclampsia(data)` yang mengembalikan probabilitas mock.
- [x] Buat `app/models/anemia_cv/inference.py` wrapper dengan fungsi `predict_anemia(image_url)` yang mengembalikan probabilitas mock.
- [x] Buat `app/models/risk_aggregator_xgb/inference.py` wrapper dengan fungsi `aggregate_risk(...)` yang mengembalikan `aggregate_score` dan `risk_badge`.
- [x] Buat orchestrator pipeline `app/pipelines/triage_engine.py` yang menghubungkan Triage Lapis 1 -> LR -> CV -> XGBoost Lapis 2.

### 2.3 `POST /api/v1/triage/analyze` Endpoint (Mock Mode)
- [x] Buat `app/routers/triage.py` dan daftarkan ke `app/main.py`.
- [x] Panggil `app/pipelines/triage_engine.py` untuk mengolah data dan mengembalikan JSON respons valid.
- [x] Pastikan waktu respons < 2 detik. (**Aktual: ~3ms**)

---

## 🎯 Target Output Files
- [x] `app/models/triage_rules.py`
- [x] `app/models/preeclampsia_lr/inference.py` (Mock)
- [x] `app/models/anemia_cv/inference.py` (Mock)
- [x] `app/models/risk_aggregator_xgb/inference.py` (Mock)
- [x] `app/pipelines/triage_engine.py`
- [x] `app/routers/triage.py`

## ✅ Acceptance Criteria
1. ✅ Endpoint `POST /api/v1/triage/analyze` aktif dan merespons sesuai kontrak.
2. ✅ Latensi inferensi mock ~3ms (jauh di bawah target 2 detik).
3. ✅ Tim Backend NestJS dapat melakukan integrasi tanpa keblokir oleh proses pelatihan model ML asli.

## 🧪 Hasil Test

| Kasus | Risk Badge | Score | Faktor Risiko | HTTP |
|---|---|---|---|---|
| Preeklampsia berat (tensi 165/110, protein+++) | 🔴 merah | 100.0 | 6 faktor | 200 |
| Tekanan borderline (142/92, protein+) | 🟡 kuning | 53.4 | 4 faktor | 200 |
| Normal (118/75, tanpa gejala) | 🟢 hijau | 0.0 | 0 faktor | 200 |
| Tanpa token (unauthorized) | — | — | — | 422 |
