"""
tests/test_api_chat.py
======================
Integration tests untuk POST /api/v1/chat.
Menguji keberadaan disclaimer medis di respons chatbot.
"""

import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from fastapi.testclient import TestClient

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture
def chat_client():
    """FastAPI test client with mocked startup."""
    with patch("app.models.preeclampsia_lr.inference.load_model"), \
         patch("app.models.risk_aggregator_xgb.inference.load_model"), \
         patch("app.models.anemia_cv.inference.load_model"):
        from app.main import app
        with TestClient(app) as c:
            yield c


@pytest.fixture
def chat_valid_headers():
    return {
        "X-Internal-Token": "dev-placeholder-internal-token-minimum-32-chars-long",
    }


@pytest.fixture
def chat_request_body():
    return {
        "pregnancy_profile_id": "test-profile-uuid-1234",
        "message": "Halo, apa tanda-tanda bahaya kehamilan trimester pertama?",
    }


class TestChatEndpointAuth:
    """Test authentication pada /api/v1/chat."""

    def test_missing_token_returns_401_or_422(self, chat_client, chat_request_body):
        """Request tanpa X-Internal-Token harus ditolak (401 atau 422)."""
        response = chat_client.post("/api/v1/chat", json=chat_request_body)
        assert response.status_code in (401, 422)

    def test_invalid_token_returns_401(self, chat_client, chat_request_body):
        """Request dengan token salah harus return 401."""
        response = chat_client.post(
            "/api/v1/chat",
            json=chat_request_body,
            headers={"X-Internal-Token": "wrong-token"},
        )
        assert response.status_code == 401

    def test_valid_token_succeeds(self, chat_client, chat_request_body, chat_valid_headers):
        """Request dengan token valid harus return 200."""
        response = chat_client.post(
            "/api/v1/chat",
            json=chat_request_body,
            headers=chat_valid_headers,
        )
        assert response.status_code == 200


class TestChatDisclaimer:
    """Test keberadaan disclaimer medis di respons chatbot."""

    def test_disclaimer_always_included_in_fallback(self, chat_client, chat_valid_headers):
        """Fallback reply harus selalu menyertakan disclaimer."""
        # Force fallback by using message that triggers it
        response = chat_client.post(
            "/api/v1/chat",
            json={
                "pregnancy_profile_id": "test-profile",
                "message": "Test message",
            },
            headers=chat_valid_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["disclaimer_included"] is True

        # The reply should contain disclaimer keywords
        reply_lower = data["reply"].lower()
        disclaimer_keywords = [
            "edukasi", "edukasi", "bukan pengganti",
            "konsultasi", "bidan", "dokter", "tenaga medis",
        ]
        has_keyword = any(kw in reply_lower for kw in disclaimer_keywords)
        assert has_keyword, f"Reply should contain medical disclaimer. Got: {data['reply'][:200]}"

    def test_disclaimer_included_flag_set(self, chat_client, chat_valid_headers):
        """Response harus punya disclaimer_included=True."""
        response = chat_client.post(
            "/api/v1/chat",
            json={
                "pregnancy_profile_id": "test-profile",
                "message": "Nutrisi apa saja yang penting untuk ibu hamil?",
            },
            headers=chat_valid_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert "disclaimer_included" in data
        assert data["disclaimer_included"] is True


class TestChatResponse:
    """Test response structure dari /api/v1/chat."""

    def test_response_has_required_fields(self, chat_client, chat_request_body, chat_valid_headers):
        """Response harus punya field reply dan disclaimer_included."""
        response = chat_client.post(
            "/api/v1/chat",
            json=chat_request_body,
            headers=chat_valid_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert "reply" in data
        assert "disclaimer_included" in data

    def test_reply_is_string(self, chat_client, chat_request_body, chat_valid_headers):
        """reply harus string."""
        response = chat_client.post(
            "/api/v1/chat",
            json=chat_request_body,
            headers=chat_valid_headers,
        )
        assert response.status_code == 200
        assert isinstance(response.json()["reply"], str)

    def test_reply_not_empty(self, chat_client, chat_request_body, chat_valid_headers):
        """reply tidak boleh kosong."""
        response = chat_client.post(
            "/api/v1/chat",
            json=chat_request_body,
            headers=chat_valid_headers,
        )
        assert response.status_code == 200
        assert len(response.json()["reply"]) > 0

    def test_disclaimer_included_is_boolean(self, chat_client, chat_request_body, chat_valid_headers):
        """disclaimer_included harus boolean."""
        response = chat_client.post(
            "/api/v1/chat",
            json=chat_request_body,
            headers=chat_valid_headers,
        )
        assert response.status_code == 200
        assert isinstance(response.json()["disclaimer_included"], bool)


class TestChatEdgeCases:
    """Test edge cases."""

    def test_long_message(self, chat_client, chat_valid_headers):
        """Message panjang harus ditangani."""
        long_message = "Apa " * 500  # ~2000 chars
        response = chat_client.post(
            "/api/v1/chat",
            json={
                "pregnancy_profile_id": "test-profile",
                "message": long_message,
            },
            headers=chat_valid_headers,
        )
        assert response.status_code == 200

    def test_empty_message_rejected(self, chat_client, chat_valid_headers):
        """Message kosong harus ditolak dengan 422."""
        response = chat_client.post(
            "/api/v1/chat",
            json={
                "pregnancy_profile_id": "test-profile",
                "message": "",
            },
            headers=chat_valid_headers,
        )
        # Pydantic min_length=1 should reject empty string
        assert response.status_code == 422

    def test_request_id_forwarded(self, chat_client, chat_request_body, chat_valid_headers):
        """X-Request-Id harus ada di response."""
        headers = {**chat_valid_headers, "X-Request-Id": "chat-test-123"}
        response = chat_client.post(
            "/api/v1/chat",
            json=chat_request_body,
            headers=headers,
        )
        assert response.status_code == 200
        assert "X-Request-Id" in response.headers
        assert response.headers["X-Request-Id"] == "chat-test-123"

    def test_various_health_topics(self, chat_client, chat_valid_headers):
        """Berbagai topik kesehatan harus direspons."""
        topics = [
            "Berapa berat badan ideal ibu hamil trimester 2?",
            "Kapan sebaiknya Ibu periksa ke bidan?",
            "Apa tanda persalinan akan dimulai?",
            "Nutrisi untuk ibu menyusui",
            "Mual saat hamil trimester pertama, apa yang harus dilakukan?",
        ]
        for message in topics:
            response = chat_client.post(
                "/api/v1/chat",
                json={
                    "pregnancy_profile_id": "test-profile",
                    "message": message,
                },
                headers=chat_valid_headers,
            )
            assert response.status_code == 200, f"Failed for message: {message}"
            data = response.json()
            assert "reply" in data
            assert len(data["reply"]) > 0
            assert data["disclaimer_included"] is True
