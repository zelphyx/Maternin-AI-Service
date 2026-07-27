"""Pytest root conftest — add ai-service/ to sys.path so `from app...` works."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))