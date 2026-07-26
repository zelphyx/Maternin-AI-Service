# Real-Data Anemia CV Training dengan Augmentasi Agresif

**Tanggal:** 2026-07-26
**Status:** Draft — menunggu user review
**Scope:** `ai-service/app/training/` + `datasets/anemia_real/`

## 1. Latar Belakang & Masalah

Pipeline anemia CV di `ai-service/app/training/anemia_synthetic_train.py` saat ini melatih model dari **`datasets/anemia_synthetic/`** (400 gambar sintetik = 200 anemia + 200 normal). Model `anemia_mobilenetv3_v1.onnx` hasil training ini hanya belajar dari data "halu" — prediksinya tidak reliable di data nyata.

**Konsekuensi:**
- Akurasi tinggi pada test sintetis (~85-95%) tapi drop drastis di dunia nyata.
- Model bisa confidently wrong pada pasien nyata → risiko klinis.
- `metadata.json` lama sudah mencantumkan warning `"Trained on SYNTHETIC images. Not validated on real Indonesian patient data."`

## 2. Tujuan

Mengganti training dari data sintetis menjadi **100% data asli + augmentasi agresif**:
- ❌ Tidak ada data sintetis di training/validation/test.
- ✅ Augmentasi sebanyak mungkin untuk memperbanyak dataset.
- ✅ Dataset asli sebanyak mungkin (target ≥1.000 gambar).
- ✅ Output `.onnx` untuk inference (kompatibel dengan pipeline sekarang).

**Success criteria:**
- Akurasi pada hold-out test set gambar asli ≥ 85%.
- Overfitting gap (train_acc − val_acc) < 10 pp.
- ONNX inference output identik dengan PyTorch (max diff < 1e-5).
- `report.md` berisi metric + visualisasi augmentation samples.

## 3. Dataset Asli — Sumber Kandidat

| Sumber | Estimasi gambar | Lisensi | Cara dapat |
|--------|-----------------|---------|-----------|
| [EYES-DEFY-ANEMIA (Kaggle)](https://www.kaggle.com/datasets/eyes-defy-anemia) | 218 + segmentasi + Hb level | Kaggle (perlu akun) | `kaggle datasets download -d eyes-defy-anemia` |
| [Roboflow Anemia Conjunctiva](https://universe.roboflow.com/search?q=anemia%20conjunctiva) | bervariasi per project | Public/varies | Export via Roboflow API |
| HemaVision-Anemia-Triage (GH) | unclear (lihat notebook) | Open | Clone + cek folder dataset |
| CP-AnemiC Ghana (sudah ada di `datasets/real_datasets/cpanemic.zip`) | ~ratusan (perlu extract) | Academic | Extract RAR dengan `unar` / `unrar` |

**Fallback jika total <1.000 gambar:**
- Pakai EYES-DEFY-ANEMIA + extract CP-AnemiC + minimal 1 project Roboflow.
- Kalau masih <1.000 → **STOP dan diskusi ulang dengan user** sebelum augmentasi agresif (akan overfit).

## 4. Arsitektur Pipeline

```
Raw Real Datasets (multi-sumber)
    ↓ download_real_datasets.py
    ↓ harmonize_labels.py (map ke: anemia / normal)
datasets/anemia_real/
    ├── train/{anemia,normal}/    ← gambar ASLI, 80%
    ├── val/{anemia,normal}/      ← gambar ASLI, 10%
    └── test/{anemia,normal}/     ← gambar ASLI, 10%
    ↓ (training: aug on-the-fly per epoch)
anemia_real_train.py
    - MobileNetV3-Small pretrained ImageNet
    - Albumentations pipeline (15+ transformasi, lihat §5)
    - CrossEntropy + class weights
    - Adam (lr=1e-4, wd=1e-4)
    - Early stopping (patience=5)
    - 30 epochs default
    ↓ export ONNX
ai-service/app/model_artifacts/anemia_mobilenetv3_v2_real.onnx
    + anemia_mobilenetv3_v2_real_metadata.json
    + report.md (auto-generated)
```

## 5. Augmentasi (Albumentations, agresif tapi domain-aware)

Pipeline augmentasi **khusus gambar konjungtiva** — hindari distorsi geometris berlebihan karena struktur mata sensitif:

| Kategori | Transformasi | Parameter | Alasan |
|----------|--------------|-----------|--------|
| Geometri | HorizontalFlip | p=0.5 | Konjungtiva kiri ≈ kanan secara klinis |
| | Rotate | limit=±15° | Variasi angle kamera |
| | ShiftScaleRotate | shift=0.05, scale=±0.1, rotate=±10° | Framing bervariasi |
| | ElasticTransform | alpha=30, sigma=5, p=0.2 | Ringan, jangan distort parah |
| Color/Lighting | RandomBrightnessContrast | ±0.2 | Variasi pencahayaan |
| | HueSaturationValue | h=±5, s=±15, v=±10 | Variasi skin tone |
| | CLAHE | clip=2.0, p=0.3 | **Penting untuk konjungtiva pucat** |
| | ColorJitter | p=0.3 | Variasi kamera HP |
| Noise/Blur | GaussNoise | var=10-50, p=0.2 | Sensor noise |
| | GaussianBlur | blur_limit=(3,5), p=0.15 | Out-of-focus |
| | MotionBlur | p=0.1 | Hand shake |
| Occlusion | CoarseDropout | 2-4 holes, 8-32 px, p=0.3 | Rambut/jari遮挡 |
| Normalize | Normalize | ImageNet mean/std | Wajib untuk pretrained |
| Resize | Resize 224×224 | — | Input model |

**Multiplier efektif:** ~10-20× per gambar per epoch (on-the-fly, tidak di-generate ke disk).

**Validation/test:** augmentasi **nonaktif** (hanya Resize + Normalize) — wajib untuk ukur akurasi real.

## 6. File yang Berubah

### Dihapus
- ❌ `ai-service/app/training/anemia_synthetic_train.py`
- ❌ `ai-service/app/training/generate_synthetic_anemia.py`
- ❌ `datasets/anemia_synthetic/` (folder + isinya)
- ❌ `ai-service/app/model_artifacts/anemia_mobilenetv3_v1.onnx` (akan di-overwrite oleh v2)
- ❌ `ai-service/app/model_artifacts/anemia_mobilenetv3_v1_metadata.json`

### Dibuat
- ✨ `ai-service/app/training/download_real_datasets.py` — helper download/katalog
- ✨ `ai-service/app/training/harmonize_labels.py` — samakan label antar dataset (mapping di YAML)
- ✨ `ai-service/app/training/anemia_real_train.py` — training pipeline utama
- ✨ `datasets/anemia_real/{train,val,test}/{anemia,normal}/` — canonical real dataset
- ✨ `datasets/anemia_real/LABEL_MAPPING.yaml` — audit trail mapping label
- ✨ `datasets/anemia_real/DATASET_SOURCES.md` — provenance per gambar (sumber + URL)
- ✨ `ai-service/app/model_artifacts/anemia_mobilenetv3_v2_real.onnx`
- ✨ `ai-service/app/model_artifacts/anemia_mobilenetv3_v2_real_metadata.json`
- ✨ `ai-service/app/model_artifacts/report.md` — auto-generated training report

### Diupdate
- 🔧 `ai-service/app/training/anemia_cv_train.py` — biarkan (untuk fallback konversi `.keras`)
- 🔧 Router/service anemia di `ai-service/app/` — pastikan inference path load `.onnx` v2 (cek kompatibilitas input shape)

## 7. Testing & Validasi

1. **Hold-out test (real, no aug):**
   - Accuracy, Precision, Recall, F1, Confusion Matrix, ROC-AUC.
   - Target: Accuracy ≥ 0.85, Recall ≥ 0.80 (false negative lebih bahaya dari false positive untuk anemia screening).
2. **Overfitting check:**
   - Gap train_acc − val_acc < 10 pp.
   - Early stopping otomatis kalau gap melebar.
3. **Visual sanity:**
   - Simpan 8 contoh augmentasi per kelas (inline di `report.md`, bukan file di disk).
4. **ONNX validation:**
   - Inference 10 gambar asli via PyTorch dan ONNX → bandingkan. Max diff < 1e-5.
5. **Negative test:**
   - Masukkan gambar non-konjungtiva (wajah penuh, pemandangan) → prediksi harus acak/dekat 0.5, **bukan confidently wrong**.

## 8. Risiko & Mitigasi

| Risiko | Mitigasi |
|--------|----------|
| Dataset publik <1.000 gambar | Wajib diskusi ulang dengan user sebelum augmentasi agresif (risiko overfit) |
| Label tidak konsisten | `harmonize_labels.py` + `LABEL_MAPPING.yaml` diaudit manual dulu |
| Augmentasi terlalu agresif → belajar noise | Train tanpa aug sebagai baseline (sanity), lalu enable aug, bandingkan val_acc |
| Overfitting parah | Dropout 0.3 di classifier head, class weighting, early stopping patience=5 |
| ONNX inference mismatch | Wajib jalankan validasi step 4 di §7 |
| HemaVision `.keras` MobileNetV2 tidak kompatibel dengan MobileNetV3-Small | `.onnx` v2 adalah MobileNetV3-Small; router harus load model by name, bukan by path hardcode |

## 9. Alasan Pendekatan Ini

**Kenapa Albumentations (bukan torchvision):**
- Lebih banyak transformasi built-in yang cocok untuk domain medis (CLAHE, ElasticTransform).
- Pipeline composition lebih rapi.
- Performance: augmentasi di CPU paralel via DataLoader workers.

**Kenapa MobileNetV3-Small pretrained ImageNet:**
- Sudah dipakai di `anemia_cv_train.py` → kompatibel dengan pipeline ONNX.
- Ringan (cocok untuk inference di backend Python).
- Pretrained ImageNet membantu untuk dataset medis kecil (transfer learning).

**Kenapa on-the-fly aug (bukan pre-generate):**
- Hemat disk: tidak ada 10.000+ file di-hard disk.
- Variasi lebih kaya: tiap epoch dapat konfigurasi acak baru → model tidak menghafal augmented image.
- Bisa pakai DataLoader workers (CPU parallel).

## 10. Yang TIDAK Termasuk (Out of Scope)

- MC Dropout / uncertainty quantification (bisa jadi iterasi v3).
- MobileNetV2 HemaVision conversion (biarkan sebagai fallback).
- Fine-tuning hyperparameter exhaustive search.
- Deployment ke production (cuma training + ONNX export).

## 11. Sumber Referensi

- [EYES-DEFY-ANEMIA dataset (Kaggle)](https://www.kaggle.com/datasets/eyes-defy-anemia)
- [HemaVision-Anemia-Triage (GitHub)](https://github.com/AminahAsif/HemaVision-Anemia-Triage)
- [Albumentations library docs](https://albumentations.ai/docs/)
- [Torchvision MobileNetV3-Small](https://pytorch.org/vision/stable/models/mobilenetv3.html)