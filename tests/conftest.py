"""Shared pytest fixtures available to every test module."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.retrieval.base import load_chunks  # noqa: E402


@pytest.fixture(scope="session")
def chunks():
    return load_chunks(ROOT / "knowledge_base")
