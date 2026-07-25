import os
import json
import csv

DATASETS_DIR = "/Users/zelphyx/Projects/Maternin-AI/datasets"

# 1. Populate TKPI (Tabel Komposisi Pangan Indonesia) Dataset
tkpi_file = os.path.join(DATASETS_DIR, "tkpi_nutrition", "tkpi_indonesian_food.csv")
os.makedirs(os.path.dirname(tkpi_file), exist_ok=True)

tkpi_data = [
    ["nama_bahan", "porsi_standar", "energi_kcal", "protein_g", "lemak_g", "karbohidrat_g", "zat_besi_mg", "kalsium_mg", "kategori"],
    ["Nasi Putih", "1 centong (100g)", 130, 2.4, 0.2, 28.6, 0.2, 25, "Makanan Pokok"],
    ["Nasi Merah", "1 centong (100g)", 110, 2.6, 0.9, 23.5, 0.8, 10, "Makanan Pokok"],
    ["Telur Ayam Rebus", "1 butir (50g)", 77, 6.3, 5.3, 0.6, 1.2, 25, "Lauk Pauk"],
    ["Telur Dadar", "1 butir (55g)", 110, 6.5, 8.8, 0.4, 1.3, 27, "Lauk Pauk"],
    ["Daging Sapi", "1 potong (50g)", 125, 13.0, 7.5, 0.0, 1.4, 6, "Lauk Pauk"],
    ["Ayam Goreng", "1 potong (50g)", 145, 12.5, 9.0, 1.5, 0.8, 12, "Lauk Pauk"],
    ["Ikan Kembung", "1 ekor (75g)", 112, 16.0, 4.0, 0.0, 1.0, 100, "Lauk Pauk"],
    ["Tahu Goreng", "1 potong (50g)", 58, 4.1, 4.2, 1.4, 1.7, 40, "Lauk Pauk"],
    ["Tempe Goreng", "1 potong (50g)", 118, 9.0, 7.7, 4.5, 1.5, 70, "Lauk Pauk"],
    ["Sayur Bayam", "1 mangkuk (100g)", 23, 2.9, 0.4, 3.6, 3.5, 166, "Sayuran"],
    ["Sayur Kangkung", "1 mangkuk (100g)", 19, 2.6, 0.2, 3.1, 2.5, 67, "Sayuran"],
    ["Sayur Sop Bening", "1 mangkuk (150g)", 45, 1.5, 1.0, 7.5, 0.7, 30, "Sayuran"],
    ["Buah Pisang", "1 buah (100g)", 89, 1.1, 0.3, 22.8, 0.3, 5, "Buah"],
    ["Buah Jeruk", "1 buah (100g)", 47, 0.9, 0.1, 11.8, 0.1, 40, "Buah"],
    ["Susu Ibu Hamil", "1 gelas (200ml)", 180, 8.0, 4.5, 26.0, 6.0, 400, "Minuman Nutrisi"]
]

with open(tkpi_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerows(tkpi_data)

print(f" -> TKPI dataset saved to {tkpi_file}")

# 2. Populate Buku KIA Knowledge Base (RAG & Guidance)
kia_kb_file = os.path.join(DATASETS_DIR, "buku_kia_kemenkes", "buku_kia_knowledge_base.json")
os.makedirs(os.path.dirname(kia_kb_file), exist_ok=True)

kia_kb_data = {
    "title": "Pedoman Kesehatan Ibu dan Anak (KIA) Kemenkes RI",
    "sections": [
        {
            "id": "trimester_1",
            "topic": "Trimester Pertama (0 - 12 Minggu)",
            "guidance": "Fokus pada pencegahan mual muntah (morning sickness), asupan asam folat (400 mcg/hari), pemeriksaan ANC pertama (K1), serta edukasi keluhan awal kehamilan.",
            "red_flags": ["Perdarahan per vaginam", "Mual muntah hebat berlebihan (hyperemesis gravidarum)", "Demam tinggi > 38°C"]
        },
        {
            "id": "trimester_2",
            "topic": "Trimester Kedua (13 - 27 Minggu)",
            "guidance": "Pemantauan gerakan janin awal, asupan zat besi (tablet tambah darah 60 mg/hari), deteksi dini tensi darah dan edema.",
            "red_flags": ["Sakit kepala hebat tak kunjung hilang", "Bengkak (edema) mendadak di wajah dan tangan", "Pandangan kabur atau berkunang-kunang", "Gerakan janin berkurang"]
        },
        {
            "id": "trimester_3",
            "topic": "Trimester Ketiga (28 - 40 Minggu)",
            "guidance": "Persiapan persalinan (P4K), pemantauan tanda persalinan, tensi darah rutin, kecukupan nutrisi dan istirahat.",
            "red_flags": ["Keluar air ketuban sebelum waktunya", "Perdarahan segar", "Kejang atau hilang kesadaran", "Tensi darah ≥ 140/90 mmHg (Risiko Preeklampsia)"]
        },
        {
            "id": "postpartum_nifas",
            "topic": "Masa Nifas (0 - 42 Hari)",
            "guidance": "Pemeriksaan kesehatan ibu nifas (KF1-KF4), ASI eksklusif, kebersihan luka jahitan/C-section, serta pemantauan kesehatan mental (baby blues).",
            "red_flags": ["Perdarahan nifas sangat banyak / berbau busuk", "Demam tinggi", "Luka jahitan membengkak/bernanah", "Rasa sedih mendalam dan putus asa terus-menerus"]
        }
    ]
}

with open(kia_kb_file, "w", encoding="utf-8") as f:
    json.dump(kia_kb_data, f, indent=2, ensure_ascii=False)

print(f" -> Buku KIA Knowledge Base saved to {kia_kb_file}")

# 3. Create Dataset Index & Metadata
metadata_file = os.path.join(DATASETS_DIR, "DATASET_INDEX.json")
metadata = {
    "project": "MaternIn-AI",
    "dataset_count": 4,
    "datasets": [
        {
            "name": "Maternal Health Risk Dataset (UCI)",
            "path": "datasets/maternal_health_risk/Maternal Health Risk Dataset.csv",
            "purpose": "Deteksi Preeklampsia & Triage Model (Logistic Regression & XGBoost)",
            "samples": 1014
        },
        {
            "name": "HemaVision & Conjunctiva Anemia Dataset",
            "path": "datasets/anemia_conjunctiva/HemaVision-Anemia-Triage",
            "purpose": "Deteksi Anemia (MobileNetV3 & MediaPipe ROI)",
            "source": "https://github.com/AminahAsif/HemaVision-Anemia-Triage"
        },
        {
            "name": "TKPI (Tabel Komposisi Pangan Indonesia)",
            "path": "datasets/tkpi_nutrition/tkpi_indonesian_food.csv",
            "purpose": "NLP Parser Laporan Makan & Gizi"
        },
        {
            "name": "Buku KIA Kemenkes RI Knowledge Base",
            "path": "datasets/buku_kia_kemenkes/buku_kia_knowledge_base.json",
            "purpose": "RAG Knowledge Base & Chatbot Edukasi Kehamilan"
        }
    ]
}

with open(metadata_file, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)

print(f" -> Dataset Index created at {metadata_file}")
