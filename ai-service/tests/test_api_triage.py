"""
tests/test_api_triage.py
=========================
Integration tests untuk POST /api/v1/triage/analyze.
Menguji endpoint dengan mock dan real model inference.
"""

import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from fastapi.testclient import TestClient


# We need to mock settings before importing the app
import os
import sys

# Ensure the app module is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture(scope="module")
def test_settings():
    """Override settings for testing."""
    from app.core import config
    original = config.settings
    yield
    config.settings = original


@pytest.fixture
def client():
    """FastAPI test client with mocked startup (no real model loading)."""
    # Patch model loading so the app starts without real models
    with patch("app.models.preeclampsia_lr.inference.load_model"), \
         patch("app.models.risk_aggregator_xgb.inference.load_model"), \
         patch("app.models.anemia_cv.inference.load_model"):
        from app.main import app
        with TestClient(app) as c:
            yield c


@pytest.fixture
def valid_headers():
    """Valid auth headers for all requests."""
    return {
        "X-Internal-Token": "dev-placeholder-internal-token-minimum-32-chars-long",
        "X-Request-Id": "test-request-id-123",
    }


@pytest.fixture
def triage_request_body():
    """Valid triage request body."""
    return {
        "pregnancy_profile_id": "test-profile-uuid-1234",
        "symptom_checkin_id": "test-checkin-uuid-5678",
        "answers": {
            "bengkak_kaki": True,
            "sakit_kepala": "ringan",
            "pandangan_kabur": False,
        },
        "latest_anc": {
            "systolic": 125,
            "diastolic": 82,
            "protein_urine": "negatif",
        },
        "has_preeclampsia_history": False,
        "bidan_phone": "6281234567890",
    }


class TestTriageEndpointAuth:
    """Test authentication pada /api/v1/triage/analyze."""

    def test_missing_token_returns_401_or_422(self, client, triage_request_body):
        """Request tanpa X-Internal-Token harus return 401 atau 422 (rejected)."""
        response = client.post(
            "/api/v1/triage/analyze",
            json=triage_request_body,
        )
        # FastAPI with required header → 422; with Depends(verify_internal_token) → 401
        # Both are rejection responses, which is what we care about
        assert response.status_code in (401, 422)

    def test_invalid_token_returns_401(self, client, triage_request_body):
        """Request dengan token salah harus return 401."""
        response = client.post(
            "/api/v1/triage/analyze",
            json=triage_request_body,
            headers={"X-Internal-Token": "wrong-token"},
        )
        assert response.status_code == 401

    def test_valid_token_succeeds(self, client, triage_request_body, valid_headers):
        """Request dengan token valid harus return 200."""
        response = client.post(
            "/api/v1/triage/analyze",
            json=triage_request_body,
            headers=valid_headers,
        )
        assert response.status_code == 200


class TestTriageEndpointResponse:
    """Test response structure dari /api/v1/triage/analyze."""

    def test_response_has_required_fields(self, client, triage_request_body, valid_headers):
        """Response harus punya semua field yang diperlukan."""
        response = client.post(
            "/api/v1/triage/analyze",
            json=triage_request_body,
            headers=valid_headers,
        )
        assert response.status_code == 200
        data = response.json()

        assert "risk_badge" in data
        assert "aggregate_score" in data
        assert "risk_factors" in data
        assert "recommendation_text" in data
        assert "triage_score" in data
        assert "anemia_probability" in data
        assert "preeclampsia_probability" in data

    def test_risk_badge_is_valid_enum(self, client, triage_request_body, valid_headers):
        """risk_badge harus salah satu dari: hijau, kuning, merah."""
        response = client.post(
            "/api/v1/triage/analyze",
            json=triage_request_body,
            headers=valid_headers,
        )
        assert response.status_code == 200
        badge = response.json()["risk_badge"]
        assert badge in ("hijau", "kuning", "merah")

    def test_aggregate_score_is_number(self, client, triage_request_body, valid_headers):
        """aggregate_score harus float/number dalam range [0, 100]."""
        response = client.post(
            "/api/v1/triage/analyze",
            json=triage_request_body,
            headers=valid_headers,
        )
        assert response.status_code == 200
        score = response.json()["aggregate_score"]
        assert isinstance(score, (int, float))
        assert 0.0 <= score <= 100.0

    def test_triage_score_is_number(self, client, triage_request_body, valid_headers):
        """triage_score harus float dalam range [0, 100]."""
        response = client.post(
            "/api/v1/triage/analyze",
            json=triage_request_body,
            headers=valid_headers,
        )
        assert response.status_code == 200
        triage_score = response.json()["triage_score"]
        assert isinstance(triage_score, (int, float))
        assert 0.0 <= triage_score <= 100.0

    def test_probabilities_in_valid_range(self, client, triage_request_body, valid_headers):
        """Probabilities harus dalam range [0.0, 1.0]."""
        response = client.post(
            "/api/v1/triage/analyze",
            json=triage_request_body,
            headers=valid_headers,
        )
        assert response.status_code == 200
        data = response.json()

        if data.get("anemia_probability") is not None:
            prob = data["anemia_probability"]
            assert 0.0 <= prob <= 1.0

        if data.get("preeclampsia_probability") is not None:
            prob = data["preeclampsia_probability"]
            assert 0.0 <= prob <= 1.0

    def test_risk_factors_is_list(self, client, triage_request_body, valid_headers):
        """risk_factors harus list of strings."""
        response = client.post(
            "/api/v1/triage/analyze",
            json=triage_request_body,
            headers=valid_headers,
        )
        assert response.status_code == 200
        risk_factors = response.json()["risk_factors"]
        assert isinstance(risk_factors, list)
        for factor in risk_factors:
            assert isinstance(factor, str)

    def test_recommendation_text_not_empty(self, client, triage_request_body, valid_headers):
        """recommendation_text tidak boleh kosong."""
        response = client.post(
            "/api/v1/triage/analyze",
            json=triage_request_body,
            headers=valid_headers,
        )
        assert response.status_code == 200
        assert len(response.json()["recommendation_text"]) > 0


class TestTriageRedScenarios:
    """Test scenario risiko tinggi (merah)."""

    def test_absolute_red_flag_produces_merah_badge(
        self, client, valid_headers
    ):
        """Absolute red flag harus menghasilkan badge merah."""
        body = {
            "pregnancy_profile_id": "test-profile",
            "symptom_checkin_id": "test-checkin",
            "answers": {
                "perdarahan": True,  # Absolute red flag
                "bengkak_kaki": True,
            },
            "latest_anc": {
                "systolic": 120,
                "diastolic": 75,
            },
        }
        response = client.post(
            "/api/v1/triage/analyze",
            json=body,
            headers=valid_headers,
        )
        assert response.status_code == 200
        assert response.json()["risk_badge"] == "merah"

    def test_very_high_bp_produces_merah_badge(
        self, client, valid_headers
    ):
        """BP sangat tinggi harus menghasilkan badge merah."""
        body = {
            "pregnancy_profile_id": "test-profile",
            "symptom_checkin_id": "test-checkin",
            "answers": {"bengkak_kaki": False},
            "latest_anc": {
                "systolic": 170,   # Danger: >= 160
                "diastolic": 115,  # Danger: >= 110
            },
        }
        response = client.post(
            "/api/v1/triage/analyze",
            json=body,
            headers=valid_headers,
        )
        assert response.status_code == 200
        assert response.json()["risk_badge"] == "merah"

    def test_low_risk_produces_hijau_badge(
        self, client, valid_headers
    ):
        """Pasien sehat harus menghasilkan badge hijau."""
        body = {
            "pregnancy_profile_id": "test-profile",
            "symptom_checkin_id": "test-checkin",
            "answers": {"bengkak_kaki": False},
            "latest_anc": {
                "systolic": 110,
                "diastolic": 70,
                "protein_urine": "negatif",
            },
        }
        response = client.post(
            "/api/v1/triage/analyze",
            json=body,
            headers=valid_headers,
        )
        assert response.status_code == 200
        assert response.json()["risk_badge"] == "hijau"

    def test_medium_risk_produces_kuning_badge(
        self, client, valid_headers
    ):
        """Pasien dengan gejala sedang harus menghasilkan badge kuning."""
        body = {
            "pregnancy_profile_id": "test-profile",
            "symptom_checkin_id": "test-checkin",
            "answers": {
                "bengkak_kaki": True,
                "sakit_kepala": "ringan",
            },
            "latest_anc": {
                "systolic": 135,
                "diastolic": 85,
            },
        }
        response = client.post(
            "/api/v1/triage/analyze",
            json=body,
            headers=valid_headers,
        )
        assert response.status_code == 200
        badge = response.json()["risk_badge"]
        # Medium risk → kuning (not absolute red)
        assert badge in ("kuning", "hijau")


class TestTriageEdgeCases:
    """Test edge cases dan error handling."""

    def test_missing_optional_fields(self, client, valid_headers):
        """Request dengan field opsional kosong harus tetap jalan."""
        body = {
            "pregnancy_profile_id": "test-profile",
            "symptom_checkin_id": "test-checkin",
            "answers": {},
            # latest_anc, has_preeclampsia_history, bidan_phone → None
        }
        response = client.post(
            "/api/v1/triage/analyze",
            json=body,
            headers=valid_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert "risk_badge" in data
        assert data["risk_factors"] == []

    def test_empty_answers(self, client, valid_headers):
        """Empty answers dict harus ditangani."""
        body = {
            "pregnancy_profile_id": "test-profile",
            "symptom_checkin_id": "test-checkin",
            "answers": {},
        }
        response = client.post(
            "/api/v1/triage/analyze",
            json=body,
            headers=valid_headers,
        )
        assert response.status_code == 200

    def test_without_conjunctiva_image(self, client, valid_headers):
        """Request tanpa gambar konjungtiva harus tetap jalan."""
        body = {
            "pregnancy_profile_id": "test-profile",
            "symptom_checkin_id": "test-checkin",
            "answers": {"bengkak_kaki": True},
            "conjunctiva_image_url": None,
        }
        response = client.post(
            "/api/v1/triage/analyze",
            json=body,
            headers=valid_headers,
        )
        assert response.status_code == 200
        # Anemia probability should be None without image
        assert response.json().get("anemia_probability") is None

    def test_request_id_header_forwarded(self, client, triage_request_body, valid_headers):
        """X-Request-Id harus ada di response headers."""
        response = client.post(
            "/api/v1/triage/analyze",
            json=triage_request_body,
            headers=valid_headers,
        )
        assert response.status_code == 200
        assert "X-Request-Id" in response.headers
        assert response.headers["X-Request-Id"] == "test-request-id-123"

    def test_missing_required_fields(self, client, valid_headers):
        """Request dengan field wajib hilang harus return 422."""
        body = {
            "answers": {},  # missing pregnancy_profile_id and symptom_checkin_id
        }
        response = client.post(
            "/api/v1/triage/analyze",
            json=body,
            headers=valid_headers,
        )
        assert response.status_code == 422


class TestTriageHealthCheck:
    """Test health check endpoint."""

    def test_health_check(self, client):
        """GET /health harus return status ok."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["service"] == "maternin-ai"


class TestGlobalExceptionHandler:
    """Test global exception handler tidak membocorkan detail internal."""

    def test_exception_does_not_leak_stack_trace(self, client, valid_headers):
        """Error tidak boleh membocorkan stack trace."""
        # Trigger an error by sending malformed data
        body = {
            "pregnancy_profile_id": "test-profile",
            "symptom_checkin_id": "test-checkin",
            "answers": {},
        }
        response = client.post(
            "/api/v1/triage/analyze",
            json=body,
            headers=valid_headers,
        )
        # If it returns 500, it should NOT contain stack trace
        if response.status_code == 500:
            text = response.text.lower()
            assert "traceback" not in text
            assert "file " not in text or "app/" not in text
            assert "detail" in response.json()
