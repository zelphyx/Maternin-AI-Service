# Task 03: Service Callbacks & Emergency WA (P0) ✅ COMPLETED

**Scope:** Integrasi pengiriman callback internal ke NestJS (`/internal/*`) dan modul WhatsApp Fonnte untuk peringatan darurat langsung (status Merah).

---

## 📑 Sub-tasks

### 3.1 NestJS Internal Webhook Client
- [x] Buat `app/services/nestjs_client.py` menggunakan `httpx.AsyncClient`.
- [x] Implementasikan fungsi `post_risk_assessment_callback(payload)`:
  - Kirim HTTP POST ke `{NESTJS_INTERNAL_BASE_URL}/internal/risk-assessments`.
  - Sertakan header `X-Internal-Token` dan `X-Request-Id`.
  - Set timeout eksplisit 5 detik.
  - Implementasikan retry otomatis maks 3x dengan exponential backoff (0.5s → 1s → 2s).
- [x] Implementasikan fungsi `post_postpartum_flag_callback(payload)` ke `/internal/postpartum-flags`.

### 3.2 Emergency WhatsApp Alert Client (Fonnte)
- [x] Buat `app/services/whatsapp_client.py` menggunakan `httpx.AsyncClient`.
- [x] Implementasikan fungsi `send_emergency_alert(phone_number, message)`:
  - Kirim request ke Fonnte API (`https://api.fonnte.com/send`) menggunakan `FONNTE_API_KEY`.
  - Retry maks 3x dengan exponential backoff (1s → 2s → 4s).
  - Sanitasi nomor telepon (08xxx → 628xxx).
  - Template pesan darurat (`build_emergency_message()`).
- [x] Sambungkan ke pipeline triage: JIKA `risk_badge == "merah"`, picu pengiriman WA darurat.
- [x] Jika Fonnte gagal setelah 3x retry, tandai `alert_delivery_status = "failed"` — TANPA crash.

### 3.3 Integrasi ke Triage Router
- [x] Update `app/routers/triage.py` — ganti TODO placeholder dengan WA alert + NestJS callback.
- [x] Callback NestJS sebagai fire-and-forget background task (`asyncio.create_task`).

---

## 🎯 Target Output Files
- [x] `app/services/nestjs_client.py`
- [x] `app/services/whatsapp_client.py`
- [x] `app/routers/triage.py` (updated)

## ✅ Acceptance Criteria
1. ✅ Setiap hasil triage otomatis dikirim balik ke NestJS via callback (3x retry + backoff).
2. ✅ `risk_badge == "merah"` → sistem langsung memanggil Fonnte API untuk WA darurat.
3. ✅ Kegagalan koneksi ke Fonnte/NestJS TIDAK menghentikan proses — semua response tetap HTTP 200.

## 🧪 Hasil Test

| Kasus | Badge | WA Status | Callback Status | Server Crash? |
|---|---|---|---|---|
| Merah + bidan_phone | 🔴 merah | `failed` (token placeholder) | Retry 3x (NestJS off) | ❌ Tidak crash |
| Kuning (tanpa WA) | 🟡 kuning | `not_triggered` | Retry 3x (NestJS off) | ❌ Tidak crash |
| Merah tanpa bidan_phone | 🔴 merah | `no_phone` | Retry 3x (NestJS off) | ❌ Tidak crash |
