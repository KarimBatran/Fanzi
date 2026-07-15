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


@pytest.fixture(autouse=True)
def isolated_scoring_caches():
    """listener.scoring caches the brand_reputation table (generation-
    invalidated) and category rules (TTL) in module state — both would
    otherwise leak across tests, since every test gets a fresh database
    but the module-level caches survive.
    """
    from listener import scoring

    scoring.clear_caches()
    yield
    scoring.clear_caches()


@pytest.fixture(autouse=True)
def isolated_redirect_cache():
    """listener.parser caches resolved redirects across calls (short TTL, by
    design, in production) — several tests reuse the same URL expecting
    different mocked responses, so the cache must not leak between tests.
    """
    import listener.parser as parser_module

    parser_module._redirect_cache.clear()
    yield
    parser_module._redirect_cache.clear()
