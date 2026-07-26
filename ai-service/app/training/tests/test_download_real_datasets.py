"""Tests for downloader manifest schema and retry helper."""
import json
from pathlib import Path

import pytest

from app.training.download_real_datasets import (
    build_manifest_entry,
    save_manifest,
    retry_with_backoff,
)


def test_build_manifest_entry_required_fields():
    entry = build_manifest_entry(
        source="kaggle_anemia_conjunctiva",
        url="https://www.kaggle.com/datasets/<some-anemia-dataset>",
        count=218,
        license="Kaggle Terms",
    )
    assert entry["source"] == "kaggle_anemia_conjunctiva"
    assert entry["count"] == 218
    assert "downloaded_at" in entry
    assert entry["license"] == "Kaggle Terms"


def test_save_manifest_creates_file(tmp_path):
    entries = [
        build_manifest_entry("src1", "https://x", 100, "MIT"),
    ]
    path = tmp_path / "MANIFEST.json"
    save_manifest(entries, str(path))
    data = json.loads(path.read_text())
    assert len(data) == 1
    assert data[0]["source"] == "src1"


def test_retry_with_backoff_succeeds_on_third_try(monkeypatch):
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("boom")
        return "ok"

    monkeypatch.setattr("time.sleep", lambda _: None)
    result = retry_with_backoff(flaky, max_attempts=3, base_delay=0.01)
    assert result == "ok"
    assert calls["n"] == 3


def test_retry_with_backoff_gives_up():
    def always_fail():
        raise RuntimeError("nope")

    with pytest.raises(RuntimeError, match="nope"):
        retry_with_backoff(always_fail, max_attempts=2, base_delay=0.01)
