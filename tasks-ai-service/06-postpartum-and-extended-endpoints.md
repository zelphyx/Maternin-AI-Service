# Task 06: Postpartum & Extended Endpoints (P1/P2) ✅ COMPLETED

**Scope:** Endpoint evaluasi nifas, prediksi tren risiko, ringkasan kunjungan bidan, dan NLP parser makanan.

---

## 📑 Sub-tasks

### 6.1 `POST /api/v1/postpartum/evaluate` (P1)
- [x] `app/routers/postpartum.py` — evaluasi checklist harian nifas
- [x] Red flags: perdarahan banyak, demam, infeksi luka, sakit kepala hebat
- [x] Mental health: pola `sering_sedih` 3+ hari berturut-turut atau 5+ total
- [x] Callback ke NestJS `/internal/postpartum-flags` (fire-and-forget)

### 6.2 `POST /api/v1/trend/predict` (P1)
- [x] `app/routers/trend.py` — regresi linear pada histori aggregate_score
- [x] Output: `trend_direction` (naik/stabil/turun) + `predicted_badge_in_days`
- [x] Confidence note dengan jumlah data points

### 6.3 `POST /api/v1/visit-brief/generate` (P2)
- [x] `app/agents/visit_brief_agent.py` — LLM summarizer (fallback tanpa LLM)
- [x] `app/routers/visit_brief.py` — endpoint terdaftar
- [x] Guardrail: tidak menambahkan klaim di luar data payload

### 6.4 `POST /api/v1/nutrition/parse` (P2)
- [x] `app/agents/nutrition_parser.py` — LLM structured JSON + keyword fallback
- [x] `app/routers/nutrition.py` — endpoint terdaftar
- [x] Guardrail: insight_text wajib menyebut "estimasi kasar"

---

## 🧪 Hasil Test

| Endpoint | Test Case | HTTP | Response | Status |
|---|---|---|---|---|
| Postpartum | Red flag (perdarahan + demam + luka infeksi + sakit kepala + riwayat PE) | 200 ✅ | `red_flag: true, mental_health: true` | ✅ |
| Postpartum | Normal (semua baik) | 200 ✅ | `red_flag: false, mental_health: false` | ✅ |
| Trend | Skor naik (25 → 42 → 58 → 66) | 200 ✅ | `naik, predicted: merah in 2 days` | ✅ |
| Nutrition | "nasi telur sayur bayam susu" | 200 ✅ | 6 items parsed, estimasi kasar | ✅ |
| Visit Brief | ANC + risk assessment history | 200 ✅ | "kuning (skor 55/100). 2 ANC." | ✅ |

## 🎯 Target Output Files
- [x] `app/routers/postpartum.py`
- [x] `app/routers/trend.py`
- [x] `app/agents/visit_brief_agent.py`
- [x] `app/routers/visit_brief.py`
- [x] `app/agents/nutrition_parser.py`
- [x] `app/routers/nutrition.py`

## ✅ Acceptance Criteria
1. ✅ Postpartum mendeteksi red flags nifas + callback ke NestJS
2. ✅ Trend memperkirakan badge change dari 4 data points
3. ✅ Visit brief menghasilkan ringkasan akurat dari data
4. ✅ Nutrition parse mengekstrak makanan + porsi estimasi
