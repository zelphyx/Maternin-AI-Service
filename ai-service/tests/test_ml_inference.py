"""
tests/test_ml_inference.py
===========================
Tests untuk model inference: Logistic Regression (Preeklampsia)
dan XGBoost Aggregator (Risk). Menguji output range dan fallback mode.
"""

import pytest

from app.models.preeclampsia_lr import inference as lr_inf
from app.models.risk_aggregator_xgb import inference as xgb_inf
from app.schemas.triage import RiskBadge


class TestPreeclampsiaLRFallback:
    """Test preeklampsia LR inference dalam fallback mode (tanpa .pkl)."""

    def setup_method(self):
        # Reset module state to fallback mode for these tests
        lr_inf._pipeline = None
        lr_inf._model_loaded = False

    def test_predict_returns_float(self):
        """predict_preeclampsia harus return float."""
        result = lr_inf.predict_preeclampsia(systolic=120, diastolic=80)
        assert isinstance(result, float)

    def test_predict_probability_range(self):
        """Probabilitas harus selalu dalam range [0.0, 1.0]."""
        test_cases = [
            {"systolic": 100, "diastolic": 60},
            {"systolic": 160, "diastolic": 110},
            {"systolic": 120, "diastolic": 80, "protein_urine": "positif_kuat"},
            {"systolic": 180, "diastolic": 120, "has_preeclampsia_history": True},
        ]
        for kwargs in test_cases:
            result = lr_inf.predict_preeclampsia(**kwargs)
            assert 0.0 <= result <= 1.0, f"Result {result} out of range for {kwargs}"

    def test_predict_high_bp_increases_probability(self):
        """Tekanan darah tinggi harus meningkatkan probabilitas."""
        normal = lr_inf.predict_preeclampsia(systolic=120, diastolic=80)
        high = lr_inf.predict_preeclampsia(systolic=170, diastolic=110)
        assert high > normal, f"High BP ({high}) should be > normal ({normal})"

    def test_predict_protein_increases_probability(self):
        """Protein urine positif kuat harus meningkatkan probabilitas."""
        no_protein = lr_inf.predict_preeclampsia(
            systolic=120, diastolic=80, protein_urine="negatif"
        )
        with_protein = lr_inf.predict_preeclampsia(
            systolic=120, diastolic=80, protein_urine="positif_kuat"
        )
        assert with_protein > no_protein

    def test_predict_preeclampsia_history_increases_probability(self):
        """Riwayat preeklampsia harus meningkatkan probabilitas."""
        no_history = lr_inf.predict_preeclampsia(systolic=120, diastolic=80)
        with_history = lr_inf.predict_preeclampsia(
            systolic=120, diastolic=80, has_preeclampsia_history=True
        )
        assert with_history > no_history

    def test_predict_all_parameters(self):
        """Semua parameter harus dapat diproses tanpa error."""
        result = lr_inf.predict_preeclampsia(
            systolic=140,
            diastolic=90,
            protein_urine="positif",
            has_preeclampsia_history=True,
            has_hypertension_history=True,
            age=30,
            gestational_age_weeks=32,
            bmi=28.5,
        )
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_predict_none_values_handled(self):
        """None values untuk BP harus ditangani (fallback ke nilai normal)."""
        result = lr_inf.predict_preeclampsia()
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_predict_protein_case_insensitive(self):
        """Protein urine harus case-insensitive."""
        r1 = lr_inf.predict_preeclampsia(systolic=120, protein_urine="POSITIF")
        r2 = lr_inf.predict_preeclampsia(systolic=120, protein_urine="positif")
        r3 = lr_inf.predict_preeclampsia(systolic=120, protein_urine="Positif")
        assert r1 == r2 == r3

    def test_predict_protein_various_formats(self):
        """Berbagai format protein urine harus diakui."""
        formats = ["+1", "+2", "+3", "+4", "positif_ringan", "positif_kuat"]
        for pf in formats:
            result = lr_inf.predict_preeclampsia(systolic=120, protein_urine=pf)
            assert isinstance(result, float)
            assert 0.0 <= result <= 1.0

    def test_load_model_nonexistent_path(self):
        """load_model dengan path tidak ada harus set model_loaded=False."""
        lr_inf.load_model("/nonexistent/path/model.pkl")
        assert lr_inf._model_loaded is False
        assert lr_inf._pipeline is None


class TestRiskAggregatorFallback:
    """Test risk aggregator XGBoost dalam fallback mode (tanpa .pkl)."""

    def setup_method(self):
        xgb_inf._model_bundle = None
        xgb_inf._model_loaded = False

    def test_aggregate_returns_dict(self):
        """aggregate_risk harus return dict."""
        result = xgb_inf.aggregate_risk(triage_score=30.0, preeclampsia_prob=0.3)
        assert isinstance(result, dict)

    def test_aggregate_has_required_keys(self):
        """Return dict harus punya keys yang diperlukan."""
        result = xgb_inf.aggregate_risk(triage_score=30.0, preeclampsia_prob=0.3)
        assert "aggregate_score" in result
        assert "risk_badge" in result
        assert "feature_importances" in result

    def test_absolute_red_immediately_merah(self):
        """is_absolute_red=True harus langsung menghasilkan badge merah."""
        result = xgb_inf.aggregate_risk(
            triage_score=30.0, preeclampsia_prob=0.1, is_absolute_red=True
        )
        assert result["risk_badge"] == RiskBadge.merah

    def test_absolute_red_score_at_least_merah_threshold(self):
        """Absolute red harus punya score >= 65 (merah threshold)."""
        result = xgb_inf.aggregate_risk(
            triage_score=20.0, preeclampsia_prob=0.1, is_absolute_red=True
        )
        assert result["aggregate_score"] >= 65.0

    def test_high_triage_score_merah(self):
        """High triage score harus menghasilkan badge merah."""
        result = xgb_inf.aggregate_risk(
            triage_score=80.0, preeclampsia_prob=0.5
        )
        assert result["risk_badge"] == RiskBadge.merah

    def test_low_triage_score_hijau(self):
        """Low triage score harus menghasilkan badge hijau."""
        result = xgb_inf.aggregate_risk(
            triage_score=10.0, preeclampsia_prob=0.1
        )
        assert result["risk_badge"] == RiskBadge.hijau

    def test_medium_triage_score_kuning(self):
        """Medium triage score harus menghasilkan badge kuning."""
        result = xgb_inf.aggregate_risk(
            triage_score=45.0, preeclampsia_prob=0.3
        )
        assert result["risk_badge"] == RiskBadge.kuning

    @pytest.mark.parametrize("triage_score,preeclampsia_prob,expected_badge", [
        # High combined → merah
        (80, 0.7, RiskBadge.merah),
        (85, 0.5, RiskBadge.merah),
        # Medium → kuning
        (40, 0.4, RiskBadge.kuning),
        (35, 0.5, RiskBadge.kuning),
        # Low → hijau
        (10, 0.1, RiskBadge.hijau),
        (20, 0.2, RiskBadge.hijau),
    ])
    def test_badge_thresholds(self, triage_score, preeclampsia_prob, expected_badge):
        """Various score combinations harus menghasilkan badge yang tepat."""
        result = xgb_inf.aggregate_risk(
            triage_score=triage_score,
            preeclampsia_prob=preeclampsia_prob,
            is_absolute_red=False,
        )
        assert result["risk_badge"] == expected_badge

    def test_score_range(self):
        """Aggregate score harus dalam range [0, 100]."""
        test_cases = [
            (0.0, 0.0, False),
            (100.0, 1.0, False),
            (50.0, 0.5, True),  # absolute red
            (0.0, 1.0, False),
            (80.0, 0.9, False),
        ]
        for triage_score, preeclampsia_prob, is_abs_red in test_cases:
            result = xgb_inf.aggregate_risk(
                triage_score=triage_score,
                preeclampsia_prob=preeclampsia_prob,
                is_absolute_red=is_abs_red,
            )
            assert 0.0 <= result["aggregate_score"] <= 100.0, \
                f"Score {result['aggregate_score']} out of range"

    def test_anemia_prob_included_in_calculation(self):
        """Anemia probability harus mempengaruhi hasil."""
        no_anemia = xgb_inf.aggregate_risk(
            triage_score=30.0, preeclampsia_prob=0.3, anemia_prob=None
        )
        with_anemia = xgb_inf.aggregate_risk(
            triage_score=30.0, preeclampsia_prob=0.3, anemia_prob=0.8
        )
        # Adding anemia should increase the score
        assert with_anemia["aggregate_score"] > no_anemia["aggregate_score"]

    def test_none_anemia_uses_partial_weight(self):
        """anemia_prob=None harus menggunakan partial weight (tanpa anemia component)."""
        # With anemia=None, weights are re-normalized
        result_none = xgb_inf.aggregate_risk(
            triage_score=30.0, preeclampsia_prob=0.3, anemia_prob=None
        )
        result_zero = xgb_inf.aggregate_risk(
            triage_score=30.0, preeclampsia_prob=0.3, anemia_prob=0.0
        )
        # Both should be valid badges
        assert isinstance(result_none["risk_badge"], RiskBadge)
        assert isinstance(result_zero["risk_badge"], RiskBadge)

    def test_feature_importances_none_in_fallback(self):
        """Fallback mode harus punya feature_importances=None."""
        result = xgb_inf.aggregate_risk(triage_score=30.0, preeclampsia_prob=0.3)
        assert result["feature_importances"] is None

    def test_load_model_nonexistent_path(self):
        """load_model dengan path tidak ada harus set model_loaded=False."""
        xgb_inf.load_model("/nonexistent/path/model.pkl")
        assert xgb_inf._model_loaded is False
        assert xgb_inf._model_bundle is None


class TestComponentWeights:
    """Test bahwa COMPONENT_WEIGHTS digunakan dengan benar."""

    def test_weights_sum_to_one(self):
        """Weights harus menjumlah 1.0."""
        total = sum(xgb_inf.COMPONENT_WEIGHTS.values())
        assert abs(total - 1.0) < 0.001

    def test_triage_lapis1_has_highest_weight(self):
        """Triage Lapis 1 harus punya bobot tertinggi."""
        weights = xgb_inf.COMPONENT_WEIGHTS
        assert weights["triage_lapis1"] == max(weights.values())


class TestBadgeThresholds:
    """Test badge threshold constants."""

    def test_merah_threshold_higher_than_kuning(self):
        assert xgb_inf.BADGE_THRESHOLDS["merah"] > xgb_inf.BADGE_THRESHOLDS["kuning"]

    def test_merah_threshold_is_65(self):
        assert xgb_inf.BADGE_THRESHOLDS["merah"] == 65

    def test_kuning_threshold_is_35(self):
        assert xgb_inf.BADGE_THRESHOLDS["kuning"] == 35
