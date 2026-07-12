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
