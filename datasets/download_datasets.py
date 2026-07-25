import os
import urllib.request
import json
import zipfile

DATASET_DIR = "/Users/zelphyx/Projects/Maternin-AI/datasets"
os.makedirs(DATASET_DIR, exist_ok=True)

print("--- Starting Dataset Downloads ---")

# 1. UCI Maternal Health Risk Dataset
uci_dir = os.path.join(DATASET_DIR, "maternal_health_risk")
os.makedirs(uci_dir, exist_ok=True)
uci_zip_path = os.path.join(uci_dir, "maternal_health_risk.zip")

uci_url = "https://archive.ics.uci.edu/static/public/863/maternal+health+risk.zip"
print(f"[1/4] Downloading UCI Maternal Health Risk dataset from {uci_url}...")
try:
    urllib.request.urlretrieve(uci_url, uci_zip_path)
    with zipfile.ZipFile(uci_zip_path, 'r') as zip_ref:
        zip_ref.extractall(uci_dir)
    print(" -> UCI Maternal Health Risk dataset extracted successfully.")
except Exception as e:
    print(f" -> Error downloading UCI dataset: {e}")

# 2. Buku KIA Kemenkes RI (Knowledge Base RAG)
kia_dir = os.path.join(DATASET_DIR, "buku_kia_kemenkes")
os.makedirs(kia_dir, exist_ok=True)
kia_pdf_path = os.path.join(kia_dir, "Buku_KIA_2020_Kemenkes.pdf")

# Direct official download link for Buku KIA Kemenkes
kia_url = "https://kesga.kemkes.go.id/assets/uploads/dokumen/Buku%20KIA%202020%20Lengkap.pdf"
print(f"[2/4] Downloading Buku KIA Kemenkes RI from {kia_url}...")
try:
    req = urllib.request.Request(
        kia_url, 
        headers={'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)'}
    )
    with urllib.request.urlopen(req) as response, open(kia_pdf_path, 'wb') as out_file:
        out_file.write(response.read())
    print(" -> Buku KIA Kemenkes RI downloaded successfully.")
except Exception as e:
    print(f" -> Error downloading Buku KIA PDF: {e}")

# 3. TKPI (Tabel Komposisi Pangan Indonesia) Dataset
tkpi_dir = os.path.join(DATASET_DIR, "tkpi_nutrition")
os.makedirs(tkpi_dir, exist_ok=True)
tkpi_csv_path = os.path.join(tkpi_dir, "tkpi_indonesian_food.csv")

# Downloading curated open TKPI dataset from public GitHub mirror
tkpi_raw_url = "https://raw.githubusercontent.com/ahmadfajar/indonesian-food-and-drink-nutrition-dataset/main/indonesian_food_nutrition.csv"
print(f"[3/4] Downloading TKPI Indonesian Food Dataset from GitHub...")
try:
    req = urllib.request.Request(
        tkpi_raw_url,
        headers={'User-Agent': 'Mozilla/5.0'}
    )
    with urllib.request.urlopen(req) as response, open(tkpi_csv_path, 'wb') as out_file:
        out_file.write(response.read())
    print(" -> TKPI Indonesian Food Dataset downloaded successfully.")
except Exception as e:
    # Backup source for TKPI if specific repo URL differs
    print(f" -> Retrying TKPI download with alternative public source...")
    alt_tkpi_url = "https://raw.githubusercontent.com/Kaggle/config/main/datasets/food_indonesia.csv"
    try:
        urllib.request.urlretrieve(alt_tkpi_url, tkpi_csv_path)
        print(" -> TKPI Dataset downloaded successfully.")
    except Exception as ex:
        print(f" -> Note on TKPI: {ex}")

# 4. Anemia Conjunctiva Sample Datasets / Repositories
anemia_dir = os.path.join(DATASET_DIR, "anemia_conjunctiva")
os.makedirs(anemia_dir, exist_ok=True)
print("[4/4] Setting up Anemia Conjunctiva Dataset folder...")

readme_anemia = os.path.join(anemia_dir, "README_ANEMIA_DATASETS.md")
with open(readme_anemia, "w") as f:
    f.write("""# Anemia Conjunctiva Image Datasets

1. **EYES-DEFY-ANEMIA (Kaggle)**
   - Requires Kaggle Authentication or direct download via Kaggle CLI:
     `kaggle datasets download -d eyes-defy-anemia`
   - Contains: 218 conjunctiva images + segmentations + Hb levels.

2. **HemaVision Anemia Triage Dataset (GitHub)**
   - Clone: `git clone https://github.com/AminahAsif/HemaVision-Anemia-Triage.git`

3. **Roboflow Anemia Conjunctiva Dataset**
   - Universe URL: https://universe.roboflow.com/search?q=anemia%20conjunctiva
""")
print(" -> Anemia dataset instructions created.")

print("--- Dataset Download Task Completed ---")
