"""
MaternIn AI Service — Triage Engine Lapis 1 (Rule-Based)
=========================================================
Weighted scoring deterministik berdasarkan PNPK Obstetri Kemenkes RI.
Threshold di sini BUKAN ML — harus transparan dan bisa diaudit.

Output:
  - triage_score (0-100): skor bahaya akumulatif
  - red_flags: list faktor risiko klinis yang terdeteksi
  - is_absolute_red: True jika ada parameter bahaya mutlak (langsung merah)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ── Konfigurasi Threshold PNPK ──────────────────────────────────────────
# Disimpan terpisah di sini agar mudah diaudit/diubah tanpa redeploy model.

THRESHOLDS = {
    # Tekanan darah (mmHg)
    "systolic_warning": 140,       # Kuning: ≥ 140
    "systolic_danger": 160,        # Merah: ≥ 160
    "diastolic_warning": 90,       # Kuning: ≥ 90
    "diastolic_danger": 110,       # Merah: ≥ 110

    # Protein urine
    "protein_urine_positive": ["positif", "positif_kuat", "+1", "+2", "+3", "+4"],
    "protein_urine_strong": ["positif_kuat", "+3", "+4"],
}

# Bobot per gejala/parameter — menentukan seberapa besar kontribusi ke skor
WEIGHTS = {
    # Tekanan darah
    "systolic_warning": 15,
    "systolic_danger": 30,
    "diastolic_warning": 15,
    "diastolic_danger": 30,

    # Protein urine
    "protein_urine_positive": 15,
    "protein_urine_strong": 25,

    # Gejala kuesioner
    "sakit_kepala_berat": 20,
    "sakit_kepala_ringan": 5,
    "pandangan_kabur": 20,
    "bengkak_kaki": 10,
    "bengkak_wajah_tangan": 15,
    "perdarahan": 35,
    "nyeri_perut_hebat": 25,
    "kejang": 40,
    "demam_tinggi": 15,
    "gerakan_janin_berkurang": 20,
    "keluar_air_ketuban": 30,
    "mual_muntah_hebat": 10,

    # Riwayat
    "preeclampsia_history": 15,
}

# Parameter bahaya mutlak — jika salah satu ada, langsung badge merah
ABSOLUTE_RED_FLAGS = {
    "perdarahan",
    "kejang",
    "systolic_danger",
    "diastolic_danger",
    "keluar_air_ketuban",
}


@dataclass
class TriageResult:
    """Hasil evaluasi rule-based triage engine lapis 1."""
    triage_score: float = 0.0
    red_flags: list[str] = field(default_factory=list)
    is_absolute_red: bool = False


def evaluate_triage(
    answers: dict[str, Any],
    latest_anc: dict[str, Any] | None = None,
    has_preeclampsia_history: bool = False,
) -> TriageResult:
    """
    Evaluasi rule-based triage berdasarkan PNPK Obstetri.

    Args:
        answers: Jawaban kuesioner adaptif (key -> value).
        latest_anc: Data ANC terakhir (systolic, diastolic, protein_urine, dll).
        has_preeclampsia_history: Riwayat preeklampsia sebelumnya.

    Returns:
        TriageResult dengan skor, red flags, dan flag bahaya mutlak.
    """
    score = 0.0
    red_flags: list[str] = []
    absolute_red = False
    anc = latest_anc or {}

    # ── 1. Evaluasi Tekanan Darah ────────────────────────────────────
    systolic = anc.get("systolic")
    diastolic = anc.get("diastolic")

    if systolic is not None:
        if systolic >= THRESHOLDS["systolic_danger"]:
            score += WEIGHTS["systolic_danger"]
            red_flags.append(f"Tekanan sistolik sangat tinggi ({systolic} mmHg)")
            absolute_red = True
        elif systolic >= THRESHOLDS["systolic_warning"]:
            score += WEIGHTS["systolic_warning"]
            red_flags.append(f"Tekanan sistolik tinggi ({systolic} mmHg)")

    if diastolic is not None:
        if diastolic >= THRESHOLDS["diastolic_danger"]:
            score += WEIGHTS["diastolic_danger"]
            red_flags.append(f"Tekanan diastolik sangat tinggi ({diastolic} mmHg)")
            absolute_red = True
        elif diastolic >= THRESHOLDS["diastolic_warning"]:
            score += WEIGHTS["diastolic_warning"]
            red_flags.append(f"Tekanan diastolik tinggi ({diastolic} mmHg)")

    # Gabungkan info tekanan darah jika kedua nilai ada
    if systolic is not None and diastolic is not None:
        # Ganti individual BP flags dengan satu flag gabungan untuk readability
        bp_flags = [f for f in red_flags if "Tekanan" in f and "mmHg" in f]
        if bp_flags:
            # Bersihkan individual flags, tambah satu gabungan
            for f in bp_flags:
                red_flags.remove(f)
            red_flags.append(f"Tekanan darah tinggi ({systolic}/{diastolic} mmHg)")

    # ── 2. Evaluasi Protein Urine ────────────────────────────────────
    protein_urine = anc.get("protein_urine", "").lower().strip()

    if protein_urine in THRESHOLDS["protein_urine_strong"]:
        score += WEIGHTS["protein_urine_strong"]
        red_flags.append("Protein urine positif kuat")
    elif protein_urine in THRESHOLDS["protein_urine_positive"]:
        score += WEIGHTS["protein_urine_positive"]
        red_flags.append("Protein urine positif")

    # ── 3. Evaluasi Jawaban Kuesioner ────────────────────────────────
    symptom_mapping = {
        # key di answers -> (weight_key, red_flag_text, is_absolute)
        "sakit_kepala": _evaluate_headache,
        "pandangan_kabur": ("pandangan_kabur", "Pandangan kabur/berkunang", False),
        "bengkak_kaki": ("bengkak_kaki", "Bengkak pada kaki", False),
        "bengkak_wajah_tangan": ("bengkak_wajah_tangan", "Bengkak mendadak di wajah/tangan", False),
        "perdarahan": ("perdarahan", "Perdarahan per vaginam", True),
        "nyeri_perut_hebat": ("nyeri_perut_hebat", "Nyeri perut hebat", False),
        "kejang": ("kejang", "Kejang / hilang kesadaran", True),
        "demam_tinggi": ("demam_tinggi", "Demam tinggi (> 38°C)", False),
        "gerakan_janin_berkurang": ("gerakan_janin_berkurang", "Gerakan janin berkurang", False),
        "keluar_air_ketuban": ("keluar_air_ketuban", "Keluar air ketuban sebelum waktunya", True),
        "mual_muntah_hebat": ("mual_muntah_hebat", "Mual muntah hebat (hyperemesis)", False),
    }

    for key, value in answers.items():
        if key not in symptom_mapping:
            continue

        mapping = symptom_mapping[key]

        # Handle special case: sakit_kepala punya level berat/ringan
        if callable(mapping):
            s, flag_text, is_abs = mapping(value)
            if s > 0:
                score += s
                red_flags.append(flag_text)
                if is_abs:
                    absolute_red = True
            continue

        weight_key, flag_text, is_abs = mapping

        # Evaluasi: boolean True / string truthy
        if _is_truthy(value):
            score += WEIGHTS[weight_key]
            red_flags.append(flag_text)
            if is_abs:
                absolute_red = True

    # ── 4. Evaluasi Riwayat ──────────────────────────────────────────
    if has_preeclampsia_history:
        score += WEIGHTS["preeclampsia_history"]
        red_flags.append("Riwayat preeklampsia pada kehamilan sebelumnya")

    # ── 5. Cap skor di 100 ───────────────────────────────────────────
    score = min(score, 100.0)

    return TriageResult(
        triage_score=round(score, 1),
        red_flags=red_flags,
        is_absolute_red=absolute_red,
    )


# ── Helper Functions ─────────────────────────────────────────────────────

def _evaluate_headache(value: Any) -> tuple[float, str, bool]:
    """Evaluasi sakit kepala berdasarkan severity level."""
    if isinstance(value, str):
        val = value.lower().strip()
        if val in ("berat", "hebat", "parah"):
            return WEIGHTS["sakit_kepala_berat"], "Sakit kepala hebat", False
        elif val in ("ringan", "sedang"):
            return WEIGHTS["sakit_kepala_ringan"], "Sakit kepala ringan", False
    elif _is_truthy(value):
        return WEIGHTS["sakit_kepala_ringan"], "Sakit kepala", False
    return 0, "", False


def _is_truthy(value: Any) -> bool:
    """Cek apakah value dianggap positif/ada gejala."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower().strip() in ("true", "ya", "iya", "yes", "1", "berat", "hebat", "parah", "banyak", "sangat_banyak")
    if isinstance(value, (int, float)):
        return value > 0
    return False
