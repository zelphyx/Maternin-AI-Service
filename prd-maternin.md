# Product Requirements Document — MaternIn (Maternal Intelligence)

**Tim IRICH — GEMASTIK XIX**
Versi 1.0

---

## 1. Ringkasan produk

MaternIn adalah ekosistem kesehatan multi-platform berbasis Agentic AI untuk deteksi dini risiko kehamilan (perdarahan, infeksi, preeklampsia) dan pemantauan ibu hamil dari trimester pertama hingga masa postpartum (0–42 hari). Terdiri dari aplikasi mobile untuk ibu hamil dan dashboard web untuk bidan Puskesmas, dihubungkan oleh lapisan AI yang memproses data secara real-time dan mengirim peringatan otomatis via WhatsApp.

**Value proposition:** deteksi dini yang proaktif dan akurat secara medis, dapat diakses ibu hamil di daerah dengan keterbatasan teknologi, tanpa membebani tenaga kesehatan secara administratif.

## 2. Latar belakang masalah (ringkas)

AKI Indonesia masih 144/100.000 kelahiran hidup (SUPAS 2025), jauh dari target SDGs 70/100.000 di 2030. Penyebab kematian bergeser ke hipertensi kehamilan dan komplikasi nonobstetrik yang sebetulnya bisa dicegah lewat pemantauan longitudinal — tapi penanganannya sering gagal karena "3 Terlambat": telat deteksi, telat ambil keputusan, telat dapat pertolongan. Solusi digital yang ada saat ini reaktif, tidak dilatih dengan data klinis lokal, dan mengabaikan fase postpartum.

## 3. Tujuan produk

- Menyediakan deteksi dini otomatis untuk perdarahan, infeksi, dan preeklampsia berbasis kuesioner adaptif, data klinis ANC, dan computer vision.
- Menjembatani kesenjangan akses antara ibu hamil perkotaan dan daerah terpencil lewat dual-platform (mobile + web + WhatsApp).
- Mengurangi ketiga "keterlambatan" lewat sistem yang proaktif menjangkau pasien dan pendamping, bukan menunggu pasien datang.
- Menjaga akurasi diagnostik setinggi mungkin karena ini menyangkut keselamatan nyawa, bukan sekadar UX.

## 4. Target pengguna

| Role | Platform | Kebutuhan utama |
|---|---|---|
| Ibu hamil | Mobile app | Screening mandiri cepat, edukasi kontekstual, akses mudah meski gaptek |
| Bidan / Nakes koordinator | Web dashboard | Prioritas kunjungan real-time, kurangi beban administratif |
| Kader posyandu | Web dashboard (akses terbatas) | Input data kehadiran & keluhan dasar untuk pasien tanpa smartphone |
| Pemerintah / pembuat kebijakan | Laporan agregat (bukan UI langsung) | Data pola risiko spasial untuk MDSR Kemenkes |

## 5. Daftar fitur

Prioritas: **P0** = wajib ada di MVP, **P1** = penambahan berdampak besar, **P2** = nice-to-have kalau waktu ada.

### 5.1 Aplikasi mobile (ibu hamil)

| Fitur | Deskripsi | Prioritas |
|---|---|---|
| Onboarding & profil kehamilan | Registrasi via WhatsApp, input HPHT, riwayat kehamilan, kondisi penyerta | P0 |
| Deteksi dini adaptif | Kuesioner bercabang (jawaban tertentu memicu pertanyaan lanjutan) + foto konjungtiva dengan overlay landmark untuk auto-crop ROI | P0 |
| Risk profile & explainability | Risk badge (Hijau/Kuning/Merah) disertai faktor risiko spesifik yang terdeteksi, bukan cuma angka | P0 |
| Asisten virtual (chatbot) | Edukasi on-demand, jawab pertanyaan kontekstual, instruksi mitigasi tegas (rawat mandiri vs rujuk IGD) | P0 |
| Peta faskes terdekat | Lokasi Puskesmas/RS via OpenStreetMap | P0 |
| Konsultasi & booking dokter | Jadwal, pembayaran, ruang chat dengan dokter | P1 |
| Edukasi kehamilan per trimester | Konten tanda bahaya, nutrisi, persiapan persalinan | P1 |
| **Postpartum tracker (0–42 hari)** | Checklist harian masa nifas + red flag khusus (perdarahan, infeksi luka, tanda baby blues) | **P1 — baru** |
| **Smart reminder agent** | Reminder ANC/checkup dengan cadence dinamis mengikuti risk level pasien, dikirim proaktif via WA | **P1 — baru** |
| **Family circle** | Undang pasangan/keluarga untuk menerima notifikasi saat status naik ke Merah | **P1 — baru** |
| **Nutrition log** | Tampilan visual dari hasil parsing NLP laporan makan via WhatsApp, dengan insight harian | **P2 — baru** |

### 5.2 Dashboard web (bidan / nakes)

| Fitur | Deskripsi | Prioritas |
|---|---|---|
| Overview dashboard | Statistik ringkas: total pasien, risiko tinggi, ANC bulan ini, tidak terpantau | P0 |
| Monitoring ibu hamil | Tabel pasien terurut otomatis berdasarkan risk badge | P0 |
| Input data ANC / proksi kader | Input manual untuk pasien tanpa smartphone | P0 |
| Alert & prioritas kunjungan | Notifikasi real-time untuk kasus risiko tinggi | P0 |
| Peta sebaran risiko wilayah | Visualisasi geografis kepadatan risiko | P1 |
| **Trend-based early warning** | Sinyal prediktif sederhana ("trending naik ke High Risk dalam ~5 hari") dari histori skor pasien, bukan cuma snapshot | **P1 — baru** |
| **Auto-generated visit brief** | Ringkasan 2–3 kalimat riwayat + red flag pasien sebelum bidan kunjungan rumah | **P2 — baru** |
| Laporan bulanan | Ekspor data untuk pelaporan MDSR | P1 |
| Rekam medis detail | Grafik histori klinis per pasien | P0 |

### 5.3 Kanal WhatsApp (lintas platform, pendukung)

| Fitur | Deskripsi | Prioritas |
|---|---|---|
| Alert darurat ke bidan | Terpicu otomatis saat status Merah | P0 |
| Reminder personalisasi ke pasien | Frekuensi mengikuti risk level (fitur smart reminder) | P1 |
| NLP parser laporan makan | Ekstraksi nilai gizi dari chat bebas | P1 |
| Notifikasi ke family circle | Terpicu otomatis saat status Merah | P1 |

## 6. Arsitektur & model AI — prinsip akurasi untuk kasus kesehatan

**Prinsip utama:** karena output sistem ini memengaruhi keputusan medis, model yang **menentukan skor risiko** harus dilatih dan dikontrol sendiri oleh tim (custom-trained di atas data klinis yang divalidasi PNPK), bukan sekadar memanggil API pihak ketiga generik. Kesalahan model kesehatan berdampak langsung — false negative berarti kasus berisiko tinggi luput terdeteksi. Model pre-built pihak ketiga hanya dipakai untuk fungsi yang **tidak memengaruhi keputusan diagnosis**.

| Komponen | Jenis model | Sumber | Alasan |
|---|---|---|---|
| Triage engine (lapis 1) | Rule-based, weighted scoring | In-house, threshold merujuk PNPK Obstetri | Keputusan awal harus transparan dan bisa diaudit |
| Deteksi anemia (CV) | MobileNetV3-Small | **Dilatih sendiri** di dataset citra konjungtiva lokal | Akurasi harus tervalidasi ke populasi & kondisi pencahayaan Indonesia, bukan model umum |
| Deteksi preeklampsia | Logistic Regression | **Dilatih sendiri** (98% akurasi, 100% presisi pada split 55:45) | Presisi tinggi krusial untuk minimalkan false negative pada kondisi fatal |
| Aggregator risiko (lapis 2) | XGBoost | **Dilatih sendiri**, ensemble dari 3 skor input | Keputusan akhir stratifikasi perlu interpretable untuk validasi klinis oleh Sp.OG |
| Face/eye landmark untuk auto-crop ROI | MediaPipe / ML Kit Face Mesh | Pre-built pihak ketiga | Hanya fungsi preprocessing/UX — tidak memengaruhi skor diagnosis, aman pakai model umum |
| Narasi rekomendasi & chatbot edukasi (lapis 3) | LLM (GROQ / Qwen) | Pre-built / API pihak ketiga | Hanya menjelaskan hasil dalam bahasa manusia — bukan pengambil keputusan diagnosis |

**Validasi & governance:**
- Setiap output risiko tinggi harus bisa ditelusuri ke faktor klinis pemicunya (explainability wajib, bukan opsional).
- Cross-validation hasil model terhadap PNPK Obstetri Kemenkes RI dan standar ICD-MM WHO.
- Rencana retraining berkala karena data bersifat longitudinal dan karakteristik populasi bisa bergeser dari waktu ke waktu.
- Target metrik minimum per komponen (acuan dari hasil benchmarking internal):

| Model | Akurasi | Presisi | Recall | F1-Score |
|---|---|---|---|---|
| XGBoost (aggregator) | 93% | 93% | 94% | 93% |
| Logistic Regression (preeklampsia) | 98% | 100% | 100% | 99% |
| MobileNetV3 (anemia) | — target ditentukan setelah pelatihan ulang dengan dataset lokal | | | |

## 7. Kebutuhan non-fungsional

- Latensi inferensi mobile < 2 detik, ukuran model terkompresi.
- Server-Side Rendering pada dashboard web agar tetap responsif di koneksi tidak stabil.
- Enkripsi JWT untuk seluruh perpindahan data klinis.
- Kepatuhan terhadap UU Pelindungan Data Pribadi untuk data kesehatan sensitif.

## 8. Metrik keberhasilan (KPI)

- Waktu penyelesaian screening mandiri < 3 menit.
- False negative rate untuk kasus risiko tinggi ditekan serendah mungkin (prioritas metrik di atas akurasi umum).
- Waktu respons bidan terhadap alert Merah.
- Retensi mingguan ibu hamil aktif menggunakan fitur deteksi dini.

## 9. Batasan (out of scope)

- Output AI bersifat alat bantu skrining, bukan diagnosis medis — pengguna tetap dianjurkan konsultasi tenaga profesional.
- Threshold klinis mengacu PNPK Kemenkes RI, sehingga penggunaan di luar konteks Indonesia perlu penyesuaian.
- Kader posyandu hanya punya akses input kehadiran & keluhan dasar, tanpa akses data klinis sensitif.

## 10. Roadmap prioritas

- **P0 (MVP inti):** onboarding, deteksi dini adaptif + CV, risk profile, chatbot, peta faskes, dashboard monitoring, alert WA, rekam medis.
- **P1 (quick win, dampak besar):** postpartum tracker, smart reminder, family circle, trend-based early warning, konsultasi, edukasi trimester, laporan bulanan.
- **P2 (nice to have):** nutrition log UI, auto-generated visit brief.