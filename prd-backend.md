# PRD Backend — NestJS (MaternIn)

**Tim IRICH — GEMASTIK XIX**
Versi 1.0 — Turunan dari `PRD_Backend_MaternIn.md`, di-scope khusus buat sesi coding NestJS.

---

## 0. Ruang lingkup dokumen ini

**Dokumen ini HANYA mencakup implementasi NestJS di folder `/backend`.**

FastAPI (`/ai-service`) di sini diperlakukan sebagai **dependency eksternal** — kontrak endpoint-nya didokumentasikan di section 6 sekadar buat referensi pemanggilan (biar tau format data yang dikirim/diterima), **BUKAN** buat diimplementasikan di sesi coding ini. Model ML, agent LangChain, dan segala logika di dalam FastAPI itu di luar tanggung jawab dokumen ini.

`PRD_Backend_MaternIn.md` (versi gabungan) tetap jadi source of truth utama buat kontrak lintas service. Kalau ada perubahan endpoint/field di sisi AI Service, update dulu di dokumen master itu, baru sinkronkan ke sini — biar nggak ada dua dokumen yang isinya beda.

Prompt buat mulai scaffolding:
```
Baca PRD_Backend_NestJS_MaternIn.md ini secara penuh. Aku cuma ngerjain sisi
NestJS di folder /backend. Anggap AI Service di /ai-service itu sudah ada
dan berjalan sesuai kontrak di section 6 — jangan implementasikan apa pun
di dalamnya, cukup panggil sesuai kontrak itu. Bantu aku setup skeleton
project NestJS sesuai struktur module di section 4.
```

---

## 1. Tech stack (NestJS)

| Layer | Teknologi |
|---|---|
| Framework | NestJS v11 (stable — bukan v12, masih dalam pengembangan) |
| ORM | Prisma + PostgreSQL (`prisma`, `@prisma/client`) |
| Cache & queue | Redis + BullMQ |
| Auth | JWT (`@nestjs/jwt`, `passport-jwt`), role-based guard |
| Validasi env | `@nestjs/config` + Joi |
| Validasi request | `class-validator` + `class-transformer` |
| Rate limiting | `@nestjs/throttler` |
| WhatsApp (non-darurat) | Fonnte API, dipanggil langsung dari NestJS buat reminder terjadwal |
| Peta faskes | Proxy ke Nominatim/OpenStreetMap |

**Konvensi:** semua nama tabel, kolom, dan field JSON pakai `snake_case`. Semua timestamp UTC. Primary key semua tabel pakai `uuid` (bukan auto-increment), karena ada data yang masuk dari device offline yang generate ID sendiri sebelum sempat sync.

---

## 2. Peran NestJS dalam sistem

NestJS adalah **satu-satunya pemilik data** — semua write ke PostgreSQL wajib lewat NestJS, termasuk hasil kalkulasi AI (AI Service tidak boleh nulis langsung ke database, dia cuma hitung dan balikin hasil atau callback ke endpoint internal NestJS).

```
Client (Flutter / Next.js)
   -> NestJS (auth, validasi, business logic)
        -> panggil AI Service (eksternal, lihat kontrak section 6)
        <- terima hasil ATAU terima callback async di endpoint internal
   -> NestJS simpan ke Postgres, trigger notifikasi/reminder kalau perlu
```

Pembagian notifikasi WhatsApp:
- **Alert darurat** (risiko Merah) dikirim **dari sisi AI Service langsung** (di luar scope dokumen ini) — NestJS cuma perlu terima callback hasilnya buat disimpan.
- **Reminder terjadwal** (ANC checkup, postpartum check-in) dikirim **dari NestJS** lewat BullMQ — ini yang jadi tanggung jawab kamu.

---

## 3. Skema database (PostgreSQL)

Semua tabel di bawah ini milik dan dikelola penuh oleh NestJS.

### 3.1 `users`
| Kolom | Tipe | Keterangan |
|---|---|---|
| id | uuid, PK | |
| role | enum(`ibu_hamil`,`bidan`,`kader`,`admin`) | |
| full_name | varchar | |
| phone_number | varchar, unique | Nomor WA aktif, dipakai login & notifikasi |
| email | varchar, nullable | |
| password_hash | varchar | |
| puskesmas_id | uuid, FK -> puskesmas, nullable | Wajib untuk role bidan/kader |
| created_at | timestamp | |
| updated_at | timestamp | |

### 3.2 `pregnancy_profiles`
| Kolom | Tipe | Keterangan |
|---|---|---|
| id | uuid, PK | |
| user_id | uuid, FK -> users | |
| hpht | date | |
| hpl | date | Dihitung otomatis (HPHT + 280 hari) |
| gravida | int | |
| existing_conditions | jsonb | Array, mis. `["anemia","hipertensi"]` |
| status | enum(`hamil`,`nifas`,`selesai`) | |
| nifas_start_date | date, nullable | |
| had_preeclampsia_history | boolean, default false | |
| created_at | timestamp | |
| updated_at | timestamp | |

### 3.3 `anc_records`
| Kolom | Tipe | Keterangan |
|---|---|---|
| id | uuid, PK | |
| pregnancy_profile_id | uuid, FK | |
| recorded_by_user_id | uuid, FK -> users | |
| source | enum(`self`,`nakes`,`kader_offline`) | |
| systolic | int, nullable | |
| diastolic | int, nullable | |
| weight_kg | numeric, nullable | |
| fundal_height_cm | numeric, nullable | |
| protein_urine | varchar, nullable | |
| platelet_count | numeric, nullable | |
| recorded_at | timestamp | |
| client_uuid | uuid, nullable | Buat idempotency sync offline |
| created_at | timestamp | |

### 3.4 `symptom_checkins`
| Kolom | Tipe | Keterangan |
|---|---|---|
| id | uuid, PK | |
| pregnancy_profile_id | uuid, FK | |
| checkin_type | enum(`pregnancy`,`postpartum`) | |
| answers | jsonb | |
| conjunctiva_image_url | varchar, nullable | |
| source | enum(`self`,`kader_offline`) | |
| client_uuid | uuid, nullable | |
| created_at | timestamp | |

### 3.5 `risk_assessments`
| Kolom | Tipe | Keterangan |
|---|---|---|
| id | uuid, PK | |
| pregnancy_profile_id | uuid, FK | |
| symptom_checkin_id | uuid, FK, nullable | |
| triage_score | numeric | Hasil dari AI Service, NestJS cuma nyimpen |
| anemia_probability | numeric, nullable | |
| preeclampsia_probability | numeric, nullable | |
| aggregate_score | numeric | |
| risk_badge | enum(`hijau`,`kuning`,`merah`) | |
| risk_factors | jsonb | |
| recommendation_text | text | |
| created_at | timestamp | |

> Catatan: kolom-kolom di tabel ini **diisi lewat endpoint internal callback** dari AI Service (lihat section 5.2) — NestJS tidak menghitung nilai-nilai ini sendiri.

### 3.6 `postpartum_logs`
| Kolom | Tipe | Keterangan |
|---|---|---|
| id | uuid, PK | |
| pregnancy_profile_id | uuid, FK | |
| day_number | int | |
| bleeding_level | enum(`normal`,`banyak`,`sangat_banyak`) | |
| fever | boolean | |
| wound_condition | enum(`baik`,`bau`,`bengkak_merah`) | |
| headache_severe | boolean | |
| mood_flag | enum(`baik`,`kadang_sedih`,`sering_sedih`) | |
| red_flag_triggered | boolean | Diisi dari hasil evaluasi AI Service |
| created_at | timestamp | |

### 3.7 `family_circle`
| Kolom | Tipe | Keterangan |
|---|---|---|
| id | uuid, PK | |
| pregnancy_profile_id | uuid, FK | |
| contact_name | varchar | |
| contact_phone | varchar | |
| relation | varchar | |
| notify_on | enum(`merah_only`,`semua_perubahan`) | |
| created_at | timestamp | |

### 3.8 `puskesmas`
| Kolom | Tipe | Keterangan |
|---|---|---|
| id | uuid, PK | |
| name | varchar | |
| latitude | numeric | |
| longitude | numeric | |
| wilayah_kerja | varchar | |

### 3.9 `notifications_log`
| Kolom | Tipe | Keterangan |
|---|---|---|
| id | uuid, PK | |
| pregnancy_profile_id | uuid, FK | |
| channel | enum(`wa_patient`,`wa_bidan`,`wa_family`,`in_app`) | |
| message | text | |
| status | enum(`pending`,`sent`,`failed`,`no_device_fallback`) | |
| sent_at | timestamp, nullable | |
| created_at | timestamp | |

### 3.10 `reminders`
| Kolom | Tipe | Keterangan |
|---|---|---|
| id | uuid, PK | |
| pregnancy_profile_id | uuid, FK | |
| reminder_type | enum(`anc_checkup`,`postpartum_checkin`) | |
| cadence_days | int | Dihitung ulang tiap risk_badge berubah |
| next_trigger_at | timestamp | |
| last_sent_at | timestamp, nullable | |
| status | enum(`active`,`paused`,`done`) | |

### 3.11 `sync_queue`
| Kolom | Tipe | Keterangan |
|---|---|---|
| id | uuid, PK | |
| device_uuid | varchar | |
| payload_type | enum(`anc_record`,`symptom_checkin`) | |
| payload | jsonb | |
| client_created_at | timestamp | Dipakai buat last-write-wins |
| synced_at | timestamp, nullable | |
| status | enum(`pending`,`processed`,`failed`) | |

### 3.12 `consultations`, `chat_messages`
Struktur standar (id, pregnancy_profile_id, FK terkait, timestamp, status) — detail final ditentukan pas mulai fitur ini (prioritas P1, kerjain belakangan).

---

## 4. Struktur module NestJS

```
/backend
  /src
    /auth              -> login, register, JWT strategy, role guard
    /users
    /pregnancy-profiles
    /anc-records
    /symptom-checkins   -> terima input, panggil AI Service, terima callback
    /risk-assessments   -> endpoint internal buat callback dari AI Service
    /postpartum
    /family-circle
    /notifications      -> Fonnte client (reminder), BullMQ processor
    /reminders          -> BullMQ scheduler, cron hitung cadence
    /sync               -> endpoint batch sync mode kader offline
    /facilities         -> proxy ke Nominatim/OpenStreetMap
    /consultations
    /chat               -> proxy ke AI Service /chat, simpan histori
    /reports            -> export laporan bulanan MDSR
    /common
      /guards           -> role-based access guard
      /interceptors
      /dto
      /internal-auth    -> validasi X-Internal-Token dari AI Service
```

---

## 5. Endpoint yang diimplementasikan di NestJS

### 5.1 Publik (perlu JWT user)

| Method | Path | Role | Deskripsi |
|---|---|---|---|
| POST | `/auth/register` | public | Registrasi pakai `phone_number` |
| POST | `/auth/login` | public | Return JWT |
| POST | `/pregnancy-profiles` | ibu_hamil, bidan, kader | Buat profil kehamilan baru |
| PATCH | `/pregnancy-profiles/:id/status` | bidan, kader | Ubah status ke `nifas`/`selesai` |
| POST | `/symptom-checkins` | ibu_hamil, kader | Terima input, panggil AI Service (section 6), simpan hasil |
| GET | `/pregnancy-profiles/:id/risk-assessments` | ibu_hamil (milik sendiri), bidan | Histori skor risiko |
| POST | `/postpartum-logs` | ibu_hamil, kader | Check-in harian, panggil AI Service evaluasi rule |
| POST | `/family-circle` | ibu_hamil | Tambah kontak keluarga |
| GET | `/bidan/patients` | bidan | List pasien wilayahnya, terurut `risk_badge` |
| GET | `/bidan/patients/:id/visit-brief` | bidan | Panggil AI Service, kembalikan ringkasan |
| POST | `/sync/batch` | kader | Kirim data offline sekaligus (lihat section 7.3) |
| GET | `/facilities/nearby` | ibu_hamil | Proxy ke Nominatim |
| POST | `/chat` | ibu_hamil | Proxy ke AI Service `/chat`, simpan ke `chat_messages` |
| GET | `/reports/monthly` | bidan | Export laporan MDSR |

### 5.2 Internal (perlu header `X-Internal-Token`, dipanggil AI Service)

| Method | Path | Deskripsi |
|---|---|---|
| POST | `/internal/risk-assessments` | AI Service kirim hasil kalkulasi buat disimpan ke tabel `risk_assessments` |
| POST | `/internal/postpartum-flags` | AI Service kirim hasil evaluasi rule postpartum |

Endpoint di section 5.2 ini **WAJIB** kamu implementasikan biarpun kamu nggak megang FastAPI-nya — karena ini pintu masuk data dari AI Service ke database kamu. Tanpa endpoint ini, hasil kalkulasi AI nggak akan pernah tersimpan.

---

## 6. Kontrak AI Service (eksternal — referensi pemanggilan, BUKAN untuk diimplementasikan)

Bagian ini bukan tugas kamu untuk dibangun. Ini kontrak yang **wajib kamu ikuti persis** pas NestJS manggil AI Service, supaya integrasi dengan sisi FastAPI (dikerjakan orang lain di tim) tidak mismatch.

**`POST {AI_SERVICE_URL}/api/v1/triage/analyze`**

Request yang NestJS kirim:
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

Response yang NestJS terima:
```json
{
  "risk_badge": "merah",
  "aggregate_score": 84,
  "risk_factors": ["Tekanan darah tinggi (145/95)", "Sakit kepala hebat", "Protein urine positif"],
  "recommendation_text": "..."
}
```

Catatan penting: alert WA darurat (kalau `risk_badge == "merah"`) dikirim **dari sisi AI Service**, bukan dari NestJS — jadi NestJS tidak perlu (dan tidak boleh) kirim WA darurat duplikat untuk kasus ini.

**`POST {AI_SERVICE_URL}/api/v1/postpartum/evaluate`**

Request: data dari `postpartum_logs` + `had_preeclampsia_history` dari profil.
Response:
```json
{ "red_flag_triggered": true, "reason": "Perdarahan banyak + sakit kepala hebat", "mental_health_flag": false }
```

**`POST {AI_SERVICE_URL}/api/v1/chat`**

Request: `{ "pregnancy_profile_id": "uuid", "message": "..." }`
Response: `{ "reply": "...", "disclaimer_included": true }`

**Wajib diimplementasikan di sisi NestJS saat manggil ketiga endpoint di atas:**
- Timeout eksplisit (5 detik) — kalau AI Service lambat/down, jangan biarin request client nge-hang, balikin status "sedang diproses" dan retry lewat background job.
- Header `X-Internal-Token` disertakan di setiap panggilan.
- Header `X-Request-Id` yang sama dibawa lintas service, buat memudahkan tracing kalau ada error pas integrasi.

---

## 7. Business rules yang relevan buat NestJS

### 7.1 Risk badge (nilai enum, bukan cara hitungnya)

NestJS cuma perlu tau nilai yang mungkin muncul dari AI Service buat keperluan sorting/filter/UI — **bukan** cara hitungnya (itu domain AI Service):
```
risk_badge: "hijau" | "kuning" | "merah"
```

### 7.2 Cadence reminder (dihitung & dijadwalkan NestJS lewat BullMQ)

```
Tiap kali risk_assessment baru masuk (lewat callback internal):
  IF risk_badge == "merah"  -> reminder.cadence_days = 3
  IF risk_badge == "kuning" -> reminder.cadence_days = 7
  IF risk_badge == "hijau"  -> reminder.cadence_days = 14
  update next_trigger_at = now() + cadence_days
```

Cadence check-in postpartum (independen dari risk_badge, berdasarkan `day_number`):
```
hari 1–3   -> tiap hari
hari 4–14  -> tiap 2-3 hari
hari 15–42 -> tiap minggu
```

### 7.3 Sync offline & idempotency

Request `POST /sync/batch`:
```json
{
  "device_uuid": "device-abc-123",
  "records": [
    {
      "client_uuid": "uuid-generated-di-hp",
      "payload_type": "symptom_checkin",
      "payload": { "...": "..." },
      "client_created_at": "2026-07-20T09:00:00Z"
    }
  ]
}
```
- Cek `client_uuid` di `sync_queue` — kalau sudah ada, skip (idempotent).
- Kalau baru: insert ke tabel terkait, lalu panggil AI Service pipeline yang sama persis kayak input langsung dari pasien (pakai kontrak section 6).
- Konflik: **last-write-wins berdasarkan `client_created_at`**.

---

## 8. Auth & role-based access

| Role | Akses |
|---|---|
| `ibu_hamil` | CRUD data miliknya sendiri saja |
| `bidan` | Read semua pasien di `puskesmas_id`-nya, write visit brief request, akses laporan |
| `kader` | Write via sync batch, TIDAK bisa akses data klinis sensitif pasien lain |
| `admin` | Full access buat demo/testing |

Implementasi: `@Roles('bidan')` decorator + `RolesGuard`, cek `req.user.role` dari JWT payload.

---

## 9. Environment variables (scope NestJS)

```
DATABASE_URL=
REDIS_URL=
JWT_SECRET=
INTERNAL_SERVICE_TOKEN=
FONNTE_API_KEY=
AI_SERVICE_URL=
NOMINATIM_BASE_URL=https://nominatim.openstreetmap.org
```

(`GROQ_API_KEY` dan config model ML itu milik `.env` di sisi AI Service, bukan tanggung jawab dokumen/env NestJS ini.)

---

## 10. Guardrails wajib (scope NestJS)

### 10.1 Keamanan credential
- **DILARANG** hardcode API key, JWT secret, database password, atau credential apa pun di kode. Semua wajib lewat env var.
- `.env` wajib masuk `.gitignore`, cuma `.env.example` (placeholder kosong) yang boleh di-commit.
- Pakai `@nestjs/config` dengan validation schema Joi, biar app gagal start dari awal kalau ada env var wajib yang kosong:
  ```ts
  ConfigModule.forRoot({
    validationSchema: Joi.object({
      DATABASE_URL: Joi.string().required(),
      JWT_SECRET: Joi.string().min(32).required(),
      FONNTE_API_KEY: Joi.string().required(),
      INTERNAL_SERVICE_TOKEN: Joi.string().min(32).required(),
      AI_SERVICE_URL: Joi.string().uri().required(),
    }),
  })
  ```
- **DILARANG** log/print value secret apa pun ke console, biarpun buat debugging.
- Endpoint `/internal/*` wajib divalidasi `X-Internal-Token` di setiap request.

### 10.2 Index database (wajib sejak migrasi pertama)

| Tabel.Kolom | Kenapa |
|---|---|
| `users.phone_number` | Dipakai tiap login |
| `pregnancy_profiles.user_id` | FK, load profil pasien |
| `pregnancy_profiles.status` | Filter hamil vs nifas |
| `anc_records.pregnancy_profile_id, recorded_at` (composite) | Histori + trend analysis |
| `risk_assessments.pregnancy_profile_id, created_at` (composite) | Histori + trend |
| `risk_assessments.risk_badge` | Hot path — sort/filter list pasien bidan |
| `reminders.next_trigger_at` | Di-query tiap cron BullMQ jalan |
| `sync_queue.client_uuid` | **Unique index**, jamin idempotency sync |
| `users.puskesmas_id` | FK, scoping data per bidan |

### 10.3 Caching (Redis)

| Cache key | TTL | Invalidasi |
|---|---|---|
| `bidan:patients:{puskesmas_id}` | 5 menit | Invalidasi begitu ada `risk_assessment` baru masuk buat pasien di wilayah itu |
| `risk:latest:{pregnancy_profile_id}` | 10 menit | Invalidasi begitu ada assessment baru |
| `facilities:nearby:{lat}:{lng}:{radius}` | 24 jam | Lokasi faskes jarang berubah |

### 10.4 Reliability & error handling
- Panggilan ke AI Service wajib timeout eksplisit (5 detik) — jangan biarin request client hang.
- Panggilan ke Fonnte wajib try-catch + retry (maks 3x, exponential backoff). Kalau gagal, catat `status: failed` di `notifications_log`, jangan sampai gagal kirim WA bikin proses lain ikut gagal.
- Endpoint list wajib pagination (`limit`/`offset`, default 20-50).
- Rate limiting wajib di `/auth/login` dan `/symptom-checkins` (`@nestjs/throttler`).
- Global exception filter wajib pastikan response error tidak bocorin stack trace/query SQL mentah/env var.
- Connection pool database dibatasi wajar (mis. max 20 buat skala kompetisi ini).

### 10.5 Checklist sebelum fitur dianggap selesai
- [ ] Nggak ada credential hardcoded di kode
- [ ] Kolom yang sering di-query/sort udah ada index-nya
- [ ] Endpoint list udah pagination
- [ ] Panggilan ke AI Service/Fonnte punya timeout dan error handling
- [ ] Endpoint internal udah divalidasi token, endpoint yang butuh role tertentu udah dipasangin guard

---

## 11. Urutan implementasi (scope NestJS)

1. Auth + users + pregnancy_profiles (fondasi)
2. Symptom checkins — endpoint terima input, panggil AI Service sesuai kontrak section 6, endpoint internal callback (section 5.2) buat nyimpen hasil
3. Bidan dashboard read endpoints (monitoring, sort risk badge)
4. Postpartum logs + endpoint internal callback flag
5. Family circle
6. Reminder scheduler (BullMQ, cadence dinamis)
7. Notifications module (Fonnte buat reminder)
8. Sync offline (kader)
9. Facilities proxy, chat proxy, consultations, reports (P1/P2, belakangan)

Catatan: kamu tetap bisa mulai dari langkah 2 walau AI Service belum jadi — pakai mock response yang sesuai kontrak section 6 dulu (bikin endpoint AI Service palsu yang balikin data statis), biar kerjaan kamu nggak keblok nunggu temen kamu selesai.