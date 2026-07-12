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
def block_real_gemini(monkeypatch):
    """Safety net so no automated test can ever reach the real Gemini API
    (and burn real quota / require a real API key): forces
    listener.analyzer._get_client() to raise instead of lazily constructing
    a real genai.Client. A test that needs a response must explicitly patch
    it via patch.object(analyzer_module, "_get_client", return_value=...) —
    that patch temporarily overrides this fixture for the duration of the
    `with` block, then this block is back in effect for the next test.
    """
    import listener.analyzer as analyzer_module

    def _forbidden_client(*args, **kwargs):
        raise AssertionError(
            "Test attempted to reach the real Gemini API — mock "
            "listener.analyzer._get_client() instead of letting this fall through."
        )

    monkeypatch.setattr(analyzer_module, "_get_client", _forbidden_client)
    monkeypatch.setattr(analyzer_module, "_client", None)
    yield
