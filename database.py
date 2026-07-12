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

-- One row per calendar day — the Gemini call count for that day. Keyed by
-- date so "reset at local midnight" falls out naturally: a new day just
-- means a new row starting at 0, no explicit reset job needed.
CREATE TABLE IF NOT EXISTS gemini_quota (
    quota_date TEXT PRIMARY KEY,
    call_count INTEGER NOT NULL DEFAULT 0
);

-- Last-seen state per (channel, product) for duplicate-deal detection.
-- identifier is "asin:<ASIN>" or "title:<normalized title>" when no ASIN
-- was available. Re-processing updates price/discount/seen_at in place.
CREATE TABLE IF NOT EXISTS duplicate_deals (
    channel_name TEXT NOT NULL,
    identifier TEXT NOT NULL,
    last_price REAL,
    last_discount_percent INTEGER,
    last_seen_at TEXT NOT NULL,
    PRIMARY KEY (channel_name, identifier)
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


def set_product_active(product_id: int, user_id: int, active: bool) -> bool:
    """Pauses/resumes a product owned by user_id. Returns False if no matching row existed."""
    with get_connection() as conn:
        cursor = conn.execute(
            "UPDATE tracked_products SET active = ? WHERE id = ? AND user_id = ?",
            (1 if active else 0, product_id, user_id),
        )
        return cursor.rowcount > 0


def get_gemini_quota_count(quota_date: str) -> int:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT call_count FROM gemini_quota WHERE quota_date = ?", (quota_date,)
        ).fetchone()
        return row["call_count"] if row else 0


def increment_gemini_quota_count(quota_date: str) -> int:
    """Upserts today's row and returns the new count."""
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO gemini_quota (quota_date, call_count) VALUES (?, 1)
            ON CONFLICT(quota_date) DO UPDATE SET call_count = call_count + 1
            """,
            (quota_date,),
        )
        row = conn.execute(
            "SELECT call_count FROM gemini_quota WHERE quota_date = ?", (quota_date,)
        ).fetchone()
        return row["call_count"]


def get_duplicate_record(channel_name: str, identifier: str) -> sqlite3.Row | None:
    with get_connection() as conn:
        return conn.execute(
            """
            SELECT last_price, last_discount_percent, last_seen_at
            FROM duplicate_deals WHERE channel_name = ? AND identifier = ?
            """,
            (channel_name, identifier),
        ).fetchone()


def upsert_duplicate_record(
    channel_name: str,
    identifier: str,
    price: float | None,
    discount_percent: int | None,
    seen_at: str,
) -> None:
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO duplicate_deals (channel_name, identifier, last_price, last_discount_percent, last_seen_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(channel_name, identifier) DO UPDATE SET
                last_price = excluded.last_price,
                last_discount_percent = excluded.last_discount_percent,
                last_seen_at = excluded.last_seen_at
            """,
            (channel_name, identifier, price, discount_percent, seen_at),
        )


def count_active_duplicate_records(cutoff_iso: str) -> int:
    """Number of duplicate_deals rows not yet expired (last_seen_at >= cutoff)."""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM duplicate_deals WHERE last_seen_at >= ?", (cutoff_iso,)
        ).fetchone()
        return row["n"]


def delete_expired_duplicate_records(cutoff_iso: str) -> int:
    """Purges rows older than the duplicate window. Returns the count deleted."""
    with get_connection() as conn:
        cursor = conn.execute("DELETE FROM duplicate_deals WHERE last_seen_at < ?", (cutoff_iso,))
        return cursor.rowcount


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
