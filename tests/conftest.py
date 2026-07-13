"""Shared pytest fixtures. Every test runs against a fresh temp SQLite file
instead of the real fanzi.db — autouse so this applies even to tests that
don't explicitly request it.
"""

from __future__ import annotations

import pytest

import database


@pytest.fixture(autouse=True)
def isolated_database(tmp_path, monkeypatch):
    db_path = tmp_path / "test_fanzi.db"
    monkeypatch.setattr(database, "DATABASE_PATH", str(db_path))
    database.init_db()
    yield


@pytest.fixture(autouse=True)
def block_real_ai_providers(monkeypatch):
    """Safety net so no automated test can ever reach the real Gemini or Groq
    API (and burn real quota / require real API keys): forces both
    GeminiProvider._get_client and GroqProvider._get_client to raise instead
    of lazily constructing a real client. A test that needs a response must
    explicitly patch it via patch.object(provider, "_get_client", ...) or
    patch.object(provider, "generate", ...) — that patch temporarily
    overrides this fixture for the duration of the `with` block, then this
    fixture is back in effect for the next test.
    """
    from listener.ai_providers import GeminiProvider, GroqProvider

    def _forbidden_client(*args, **kwargs):
        raise AssertionError(
            "Test attempted to reach a real AI provider API — mock "
            "GeminiProvider._get_client()/GroqProvider._get_client() (or "
            "provider.generate()) instead of letting this fall through."
        )

    monkeypatch.setattr(GeminiProvider, "_get_client", _forbidden_client)
    monkeypatch.setattr(GroqProvider, "_get_client", _forbidden_client)
    yield
