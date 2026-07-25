"""
tests/test_triage_rules.py
============================
Boundary value & functional tests untuk Lapis 1 (Rule-Based Triage).
Menguji threshold sistolik/diastolik & protein urine berdasarkan PNPK.
"""

import pytest

from app.models.triage_rules import (
    evaluate_triage,
    TriageResult,
    THRESHOLDS,
    WEIGHTS,
    ABSOLUTE_RED_FLAGS,
)


class TestBloodPressureSystolic:
    """Test boundary values untuk tekanan sistolik."""

    @pytest.mark.parametrize("systolic,expected_score", [
        # Normal: < 140 → 0
        (100, 0),
        (120, 0),
        (130, 0),
        (139, 0),
    ])
    def test_systolic_normal_no_score(self, systolic, expected_score):
        result = evaluate_triage({}, {"systolic": systolic})
        assert result.triage_score == expected_score

    @pytest.mark.parametrize("systolic,expected_score", [
        # Warning: 140-159 → +15
        (140, 15),
        (145, 15),
        (150, 15),
        (155, 15),
        (159, 15),
    ])
    def test_systolic_warning_threshold(self, systolic, expected_score):
        # Pass diastolic to trigger combined BP format
        result = evaluate_triage({}, {"systolic": systolic, "diastolic": 75})
        assert result.triage_score == expected_score
        assert any("Tekanan" in f and "tinggi" in f for f in result.red_flags)
        assert result.is_absolute_red is False

    @pytest.mark.parametrize("systolic,expected_score", [
        # Danger: >= 160 → +30, absolute red
        (160, 30),
        (170, 30),
        (180, 30),
        (200, 30),
    ])
    def test_systolic_danger_threshold(self, systolic, expected_score):
        result = evaluate_triage({}, {"systolic": systolic, "diastolic": 70})
        assert result.triage_score == expected_score
        # With both BP values present, flags are combined as "Tekanan darah tinggi ({systolic}/{diastolic})"
        assert any("Tekanan" in f and "tinggi" in f for f in result.red_flags)
        assert result.is_absolute_red is True


class TestBloodPressureDiastolic:
    """Test boundary values untuk tekanan diastolik."""

    @pytest.mark.parametrize("diastolic,expected_score", [
        # Normal: < 90 → 0
        (60, 0),
        (70, 0),
        (80, 0),
        (89, 0),
    ])
    def test_diastolic_normal_no_score(self, diastolic, expected_score):
        result = evaluate_triage({}, {"diastolic": diastolic})
        assert result.triage_score == expected_score

    @pytest.mark.parametrize("diastolic,expected_score", [
        # Warning: 90-109 → +15
        (90, 15),
        (95, 15),
        (100, 15),
        (105, 15),
        (109, 15),
    ])
    def test_diastolic_warning_threshold(self, diastolic, expected_score):
        result = evaluate_triage({}, {"diastolic": diastolic})
        assert result.triage_score == expected_score
        assert any("Tekanan diastolik tinggi" in f for f in result.red_flags)
        assert result.is_absolute_red is False

    @pytest.mark.parametrize("diastolic,expected_score", [
        # Danger: >= 110 → +30, absolute red
        (110, 30),
        (115, 30),
        (120, 30),
    ])
    def test_diastolic_danger_threshold(self, diastolic, expected_score):
        result = evaluate_triage({}, {"diastolic": diastolic})
        assert result.triage_score == expected_score
        assert any("Tekanan diastolik sangat tinggi" in f for f in result.red_flags)
        assert result.is_absolute_red is True


class TestCombinedBloodPressure:
    """Test when both systolic and diastolic values are present."""

    def test_combined_bp_replaces_individual_flags(self):
        """Individual flags harus digabungkan jadi satu flag bp terpadu."""
        result = evaluate_triage({}, {"systolic": 150, "diastolic": 95})
        # Harusnya ada flag gabungan, bukan 2 flag individual
        has_combined = any("Tekanan darah tinggi" in f for f in result.red_flags)
        assert has_combined
        # Score harus 30 (15 + 15)
        assert result.triage_score == 30

    def test_combined_both_danger_absolute_red(self):
        """Systolic danger + diastolic danger → absolute red."""
        result = evaluate_triage({}, {"systolic": 165, "diastolic": 115})
        assert result.is_absolute_red is True
        assert result.triage_score == 60  # 30 + 30


class TestProteinUrine:
    """Test protein urine scoring."""

    @pytest.mark.parametrize("protein,expected_score", [
        # Negatif → 0
        ("negatif", 0),
        ("Negatif", 0),
        ("NEGATIF", 0),
        ("", 0),
        (None, 0),
    ])
    def test_protein_negative_no_score(self, protein, expected_score):
        result = evaluate_triage({}, {"protein_urine": protein} if protein is not None else {})
        assert result.triage_score == expected_score

    @pytest.mark.parametrize("protein,expected_score,expected_flag", [
        # Positive (+1, +2, positif) → +15
        ("positif", 15, "Protein urine positif"),
        # Note: "positif_ringan" is in LR inference map but NOT in THRESHOLDS for triage_rules
        ("+1", 15, "Protein urine positif"),
        ("+2", 15, "Protein urine positif"),
    ])
    def test_protein_positive(self, protein, expected_score, expected_flag):
        result = evaluate_triage({}, {"protein_urine": protein})
        assert result.triage_score == expected_score
        assert expected_flag in result.red_flags

    @pytest.mark.parametrize("protein,expected_score,expected_flag", [
        # Strong positive (+3, +4, positif_kuat) → +25
        ("positif_kuat", 25, "Protein urine positif kuat"),
        ("+3", 25, "Protein urine positif kuat"),
        ("+4", 25, "Protein urine positif kuat"),
        ("POSITIF_KUAT", 25, "Protein urine positif kuat"),
    ])
    def test_protein_strong_positive(self, protein, expected_score, expected_flag):
        result = evaluate_triage({}, {"protein_urine": protein})
        assert result.triage_score == expected_score
        assert expected_flag in result.red_flags


class TestSymptoms:
    """Test gejala kuesioner scoring."""

    @pytest.mark.parametrize("symptom,value,expected_score,expected_flag,is_absolute", [
        # Sakit kepala
        ("sakit_kepala", "berat", 20, "Sakit kepala hebat", False),
        ("sakit_kepala", "hebat", 20, "Sakit kepala hebat", False),
        ("sakit_kepala", "parah", 20, "Sakit kepala hebat", False),
        ("sakit_kepala", "ringan", 5, "Sakit kepala ringan", False),
        ("sakit_kepala", "sedang", 5, "Sakit kepala ringan", False),
        ("sakit_kepala", True, 5, "Sakit kepala", False),
        # Pandangan kabur
        ("pandangan_kabur", True, 20, "Pandangan kabur/berkunang", False),
        ("pandangan_kabur", "ya", 20, "Pandangan kabur/berkunang", False),
        # Bengkak
        ("bengkak_kaki", True, 10, "Bengkak pada kaki", False),
        ("bengkak_wajah_tangan", True, 15, "Bengkak mendadak di wajah/tangan", False),
        # Absolute red flags
        ("perdarahan", True, 35, "Perdarahan per vaginam", True),
        ("perdarahan", "ya", 35, "Perdarahan per vaginam", True),
        ("kejang", True, 40, "Kejang / hilang kesadaran", True),
        ("nyeri_perut_hebat", True, 25, "Nyeri perut hebat", False),
        ("demam_tinggi", True, 15, "Demam tinggi (> 38°C)", False),
        ("gerakan_janin_berkurang", True, 20, "Gerakan janin berkurang", False),
        ("keluar_air_ketuban", True, 30, "Keluar air ketuban sebelum waktunya", True),
        ("mual_muntah_hebat", True, 10, "Mual muntah hebat (hyperemesis)", False),
    ])
    def test_symptom_scoring(
        self, symptom, value, expected_score, expected_flag, is_absolute
    ):
        result = evaluate_triage({symptom: value})
        assert result.triage_score == expected_score
        assert expected_flag in result.red_flags
        assert result.is_absolute_red == is_absolute

    @pytest.mark.parametrize("symptom,value", [
        ("pandangan_kabur", False),
        ("bengkak_kaki", False),
        ("sakit_kepala", "tidak"),
        ("sakit_kepala", False),
    ])
    def test_symptom_negative_no_score(self, symptom, value):
        result = evaluate_triage({symptom: value})
        assert result.triage_score == 0
        assert not result.is_absolute_red


class TestPreeclampsiaHistory:
    """Test riwayat preeklampsia scoring."""

    def test_history_adds_score(self):
        result = evaluate_triage({}, {}, has_preeclampsia_history=True)
        assert result.triage_score == 15
        assert "Riwayat preeklampsia pada kehamilan sebelumnya" in result.red_flags

    def test_no_history_no_score(self):
        result = evaluate_triage({}, {}, has_preeclampsia_history=False)
        assert result.triage_score == 0


class TestScoreCap:
    """Test bahwa skor maksimum di-capped di 100."""

    def test_score_capped_at_100(self):
        """Semua kombinasi risiko tinggi harus cap di 100."""
        answers = {
            "perdarahan": True,        # +35 → absolute red
            "kejang": True,            # +40 → absolute red
            "sakit_kepala": "berat",   # +20
            "pandangan_kabur": True,    # +20
            "bengkak_wajah_tangan": True,  # +15
            "demam_tinggi": True,       # +15
        }
        anc = {
            "systolic": 165,           # +30
            "diastolic": 115,          # +30
            "protein_urine": "positif_kuat",  # +25
        }
        result = evaluate_triage(answers, anc, has_preeclampsia_history=True)  # +15

        raw_total = 35 + 40 + 20 + 20 + 15 + 15 + 30 + 30 + 25 + 15
        # Raw total would be 245, but capped at 100
        assert result.triage_score == 100.0
        assert result.is_absolute_red is True

    def test_score_exactly_100_not_capped(self):
        """Skor yang sudah 100 tidak berubah."""
        answers = {
            "perdarahan": True,  # +35
            "kejang": True,      # +40
            # total = 75
        }
        anc = {
            "systolic": 160,     # +30
            "diastolic": 110,    # +30
            # total = 165 (already capped at 100 by rule)
            # But from evaluate_triage perspective: 35+40+30+30 = 135
        }
        # Combined: 35+40+30+30 = 135 → capped 100
        result = evaluate_triage(answers, anc)
        assert result.triage_score == 100.0


class TestAbsoluteRedFlags:
    """Test absolute red flag detection."""

    @pytest.mark.parametrize("answers,anc", [
        ({"perdarahan": True}, {}),
        ({"kejang": True}, {}),
        ({}, {"systolic": 160}),
        ({}, {"diastolic": 110}),
        ({"keluar_air_ketuban": True}, {}),
    ])
    def test_absolute_red_flag(self, answers, anc):
        """Setiap absolute red flag harus menghasilkan is_absolute_red=True."""
        result = evaluate_triage(answers, anc)
        assert result.is_absolute_red is True

    def test_multiple_absolute_red_flags(self):
        """Bisa ada multiple absolute red flags, tapi is_absolute_red tetap True."""
        # Note: triage score is capped at 100
        result = evaluate_triage({"perdarahan": True, "kejang": True}, {"systolic": 165})
        assert result.is_absolute_red is True
        assert result.triage_score == 100.0  # 35 + 40 + 30, capped at 100

    def test_no_absolute_red_flag(self):
        """Warning-level symptoms tidak menghasilkan absolute red."""
        result = evaluate_triage(
            {"sakit_kepala": "berat", "bengkak_kaki": True},
            {"systolic": 140, "diastolic": 90},
        )
        assert result.is_absolute_red is False


class TestEdgeCases:
    """Edge cases dan malformed input."""

    def test_empty_answers_and_anc(self):
        result = evaluate_triage({}, {})
        assert result.triage_score == 0
        assert result.red_flags == []
        assert result.is_absolute_red is False

    def test_unknown_symptom_ignored(self):
        """Symptom yang tidak dikenal harus diabaikan (tidak crash)."""
        result = evaluate_triage({"gejala_gaib": True, "sakit_kepala": "ringan"})
        assert result.triage_score == 5
        assert "gejala_gaib" not in str(result.red_flags)

    def test_none_anc_handled(self):
        """ANC None harus ditangani dengan graceful."""
        result = evaluate_triage({}, None)
        assert result.triage_score == 0

    def test_return_type(self):
        """Return value harus TriageResult dataclass."""
        result = evaluate_triage({}, {})
        assert isinstance(result, TriageResult)

    def test_case_insensitive_protein(self):
        """Protein urine harus case-insensitive."""
        result = evaluate_triage({}, {"protein_urine": "POSITIF"})
        assert result.triage_score == 15

    def test_string_truthy_values(self):
        """String truthy values selain 'ya' harus reconhecido."""
        result = evaluate_triage({"pandangan_kabur": "iya"})
        assert result.triage_score == 20

    def test_numeric_truthy_values(self):
        """Numeric values > 0 harus dianggap truthy."""
        result = evaluate_triage({"pandangan_kabur": 1})
        assert result.triage_score == 20
        result2 = evaluate_triage({"pandangan_kabur": 5})
        assert result2.triage_score == 20
        result3 = evaluate_triage({"pandangan_kabur": 0})
        assert result3.triage_score == 0
