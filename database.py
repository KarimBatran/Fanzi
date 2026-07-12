"""SQLite schema and queries. Plain sqlite3 (stdlib) — this is a personal,
single-user-scale bot, so a blocking connection per call is fine; no ORM.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from typing import Iterator

from config import DATABASE_PATH
from models.tracked_product import TrackedProduct
from models.user import User

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    telegram_id INTEGER NOT NULL UNIQUE,
    username TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS tracked_products (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL REFERENCES users(id),
    asin TEXT NOT NULL,
    title TEXT,
    url TEXT NOT NULL,
    current_price REAL,
    target_price REAL NOT NULL,
    currency TEXT NOT NULL DEFAULT 'EGP',
    last_checked TEXT,
    last_notified_price REAL,
    active INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


@contextmanager
def get_connection() -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with get_connection() as conn:
        conn.executescript(SCHEMA)


def get_or_create_user(telegram_id: int, username: str | None) -> User:
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM users WHERE telegram_id = ?", (telegram_id,)).fetchone()
        if row is None:
            conn.execute(
                "INSERT INTO users (telegram_id, username) VALUES (?, ?)",
                (telegram_id, username),
            )
            row = conn.execute("SELECT * FROM users WHERE telegram_id = ?", (telegram_id,)).fetchone()
        return _row_to_user(row)


def add_tracked_product(
    user_id: int,
    asin: str,
    title: str | None,
    url: str,
    current_price: float | None,
    target_price: float,
    currency: str = "EGP",
) -> TrackedProduct:
    with get_connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO tracked_products
                (user_id, asin, title, url, current_price, target_price, currency)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (user_id, asin, title, url, current_price, target_price, currency),
        )
        row = conn.execute(
            "SELECT * FROM tracked_products WHERE id = ?", (cursor.lastrowid,)
        ).fetchone()
        return _row_to_product(row)


def get_tracked_product_by_asin(user_id: int, asin: str) -> TrackedProduct | None:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM tracked_products WHERE user_id = ? AND asin = ? AND active = 1",
            (user_id, asin),
        ).fetchone()
        return _row_to_product(row) if row else None


def get_latest_price_for_asin(asin: str) -> float | None:
    """Most recently checked price for this ASIN across all users' tracked
    products — used as lightweight price history for deal analysis, since
    there is no dedicated price_points table.
    """
    with get_connection() as conn:
        row = conn.execute(
            """
            SELECT current_price FROM tracked_products
            WHERE asin = ? AND current_price IS NOT NULL
            ORDER BY last_checked DESC, created_at DESC LIMIT 1
            """,
            (asin,),
        ).fetchone()
        return row["current_price"] if row else None


def get_active_products(user_id: int) -> list[TrackedProduct]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM tracked_products WHERE user_id = ? AND active = 1 ORDER BY id",
            (user_id,),
        ).fetchall()
        return [_row_to_product(row) for row in rows]


def get_all_active_products_with_owner() -> list[tuple[TrackedProduct, int]]:
    """Every active product across all users, paired with the owning user's
    telegram_id — used by the scheduler to know who to alert.
    """
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT tp.*, u.telegram_id AS owner_telegram_id
            FROM tracked_products tp
            JOIN users u ON u.id = tp.user_id
            WHERE tp.active = 1
            ORDER BY tp.id
            """
        ).fetchall()
        return [(_row_to_product(row), row["owner_telegram_id"]) for row in rows]


def update_price_check(product_id: int, current_price: float) -> None:
    with get_connection() as conn:
        conn.execute(
            "UPDATE tracked_products SET current_price = ?, last_checked = datetime('now') WHERE id = ?",
            (current_price, product_id),
        )


def mark_notified(product_id: int, price: float) -> None:
    with get_connection() as conn:
        conn.execute(
            "UPDATE tracked_products SET last_notified_price = ? WHERE id = ?",
            (price, product_id),
        )


def remove_product(product_id: int, user_id: int) -> bool:
    """Deletes a product owned by user_id. Returns False if no matching row existed."""
    with get_connection() as conn:
        cursor = conn.execute(
            "DELETE FROM tracked_products WHERE id = ? AND user_id = ?", (product_id, user_id)
        )
        return cursor.rowcount > 0


def _row_to_user(row: sqlite3.Row) -> User:
    return User(
        id=row["id"],
        telegram_id=row["telegram_id"],
        username=row["username"],
        created_at=row["created_at"],
    )


def _row_to_product(row: sqlite3.Row) -> TrackedProduct:
    return TrackedProduct(
        id=row["id"],
        user_id=row["user_id"],
        asin=row["asin"],
        title=row["title"],
        url=row["url"],
        current_price=row["current_price"],
        target_price=row["target_price"],
        currency=row["currency"],
        last_checked=row["last_checked"],
        last_notified_price=row["last_notified_price"],
        active=bool(row["active"]),
        created_at=row["created_at"],
    )
