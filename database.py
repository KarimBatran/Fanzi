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
-- external_quota_exhausted tracks Google's own quota (as opposed to our
-- internal call_count/DAILY_ANALYSIS_CAP, which is just a self-imposed
-- ceiling and may not match Google's real limit) — set the moment a
-- RESOURCE_EXHAUSTED response is seen, so further calls stop being wasted
-- on a request that's guaranteed to fail. Reset by the same date rollover.
CREATE TABLE IF NOT EXISTS gemini_quota (
    quota_date TEXT PRIMARY KEY,
    call_count INTEGER NOT NULL DEFAULT 0,
    external_quota_exhausted INTEGER NOT NULL DEFAULT 0
);

-- Same shape as gemini_quota, for the Groq fallback provider
-- (listener/ai_providers.py) — kept as a separate table rather than a
-- generic provider-keyed one so each provider's existing call pattern
-- (get/increment/mark/reset) stays a simple one-to-one mirror.
CREATE TABLE IF NOT EXISTS groq_quota (
    quota_date TEXT PRIMARY KEY,
    call_count INTEGER NOT NULL DEFAULT 0,
    external_quota_exhausted INTEGER NOT NULL DEFAULT 0
);

-- Per-provider, per-day analytics (listener/ai_providers.py) — separate from
-- gemini_quota/groq_quota (which track raw call counts/quota state used for
-- provider selection) so this table is purely observational: successes,
-- failures, retries, circuit-breaker trips, and failovers-to-this-provider,
-- plus a running latency sum/count for the average shown in /status.
-- Keyed by (provider, stat_date) so both "today's" numbers and all-time
-- history (via SUM across dates) are available, and everything survives
-- restarts automatically since it's plain SQLite state.
CREATE TABLE IF NOT EXISTS provider_stats (
    provider TEXT NOT NULL,
    stat_date TEXT NOT NULL,
    successful_requests INTEGER NOT NULL DEFAULT 0,
    failed_requests INTEGER NOT NULL DEFAULT 0,
    quota_exhaustion_events INTEGER NOT NULL DEFAULT 0,
    circuit_breaker_activations INTEGER NOT NULL DEFAULT 0,
    total_latency_ms REAL NOT NULL DEFAULT 0,
    total_retries INTEGER NOT NULL DEFAULT 0,
    total_failovers INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (provider, stat_date)
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

-- Every successful (real Gemini/Groq) verdict — the raw training data for
-- listener/learning.py. Never written for unavailable/skipped/fallback
-- verdicts or parser failures (see listener/analyzer.py).
CREATE TABLE IF NOT EXISTS verdicts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    asin TEXT NOT NULL,
    provider TEXT NOT NULL,
    brand TEXT,
    category TEXT NOT NULL,
    title TEXT,
    current_price REAL NOT NULL,
    discount_percent INTEGER,
    deal_quality TEXT NOT NULL,
    reason TEXT,
    suggested_target INTEGER,
    channel TEXT,
    timestamp TEXT NOT NULL
);

-- Aggregated knowledge derived from verdict history (listener/learning.py).
-- key is rule-type-specific: brand name, "brand|category", "category|price_bucket",
-- or "category|discount_bucket". confidence/predicted_quality/sample_count
-- are recomputed incrementally on every new verdict (see rule_votes below),
-- never by rescanning all of `verdicts` (except an explicit /rebuildrules).
CREATE TABLE IF NOT EXISTS learned_rules (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    rule_type TEXT NOT NULL,
    key TEXT NOT NULL,
    predicted_quality TEXT NOT NULL,
    confidence REAL NOT NULL,
    sample_count INTEGER NOT NULL DEFAULT 0,
    last_validated TEXT,
    last_updated TEXT NOT NULL,
    rule_version INTEGER NOT NULL DEFAULT 1,
    enabled INTEGER NOT NULL DEFAULT 1,
    UNIQUE (rule_type, key)
);

-- Internal support table (not part of the learned_rules columns the spec
-- lists, but required to recompute confidence/predicted_quality
-- incrementally with recency decay applied, without rescanning history).
-- One row per (rule_type, key, quality) holding that quality's
-- decay-weighted vote total.
CREATE TABLE IF NOT EXISTS rule_votes (
    rule_type TEXT NOT NULL,
    key TEXT NOT NULL,
    quality TEXT NOT NULL,
    weight REAL NOT NULL DEFAULT 0,
    PRIMARY KEY (rule_type, key, quality)
);

-- Per-day learning-engine analytics (listener/learning.py) — survives
-- restarts like provider_stats.
CREATE TABLE IF NOT EXISTS learning_stats (
    stat_date TEXT PRIMARY KEY,
    ai_calls_saved INTEGER NOT NULL DEFAULT 0,
    rule_hits INTEGER NOT NULL DEFAULT 0,
    rule_misses INTEGER NOT NULL DEFAULT 0,
    validation_calls INTEGER NOT NULL DEFAULT 0,
    validation_agreements INTEGER NOT NULL DEFAULT 0,
    validation_disagreements INTEGER NOT NULL DEFAULT 0
);

-- Tiny key/value store — currently just the knowledge-base version counter,
-- bumped on /resetrules and /rebuildrules so /status can show it.
CREATE TABLE IF NOT EXISTS learning_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- Per-channel, per-day pipeline counters (listener/watcher.py) — survives
-- restarts like provider_stats/learning_stats. posts_received is bumped for
-- every single incoming post regardless of outcome, so it's the true
-- "is Telegram delivering updates for this channel at all" signal,
-- independent of whether anything downstream ever parses/forwards.
CREATE TABLE IF NOT EXISTS channel_stats (
    channel TEXT NOT NULL,
    stat_date TEXT NOT NULL,
    posts_received INTEGER NOT NULL DEFAULT 0,
    parsed INTEGER NOT NULL DEFAULT 0,
    forwarded INTEGER NOT NULL DEFAULT 0,
    duplicates INTEGER NOT NULL DEFAULT 0,
    no_price_failures INTEGER NOT NULL DEFAULT 0,
    no_asin_failures INTEGER NOT NULL DEFAULT 0,
    non_amazon_links INTEGER NOT NULL DEFAULT 0,
    ai_analyses INTEGER NOT NULL DEFAULT 0,
    rule_hits INTEGER NOT NULL DEFAULT 0,
    total_failures INTEGER NOT NULL DEFAULT 0,
    total_latency_ms REAL NOT NULL DEFAULT 0,
    latency_count INTEGER NOT NULL DEFAULT 0,
    total_ai_latency_ms REAL NOT NULL DEFAULT 0,
    ai_latency_count INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (channel, stat_date)
);

-- Current (not date-scoped) last-post state per channel, for /status and the
-- watchdog — updated on every post received, regardless of outcome.
CREATE TABLE IF NOT EXISTS channel_last_post (
    channel TEXT PRIMARY KEY,
    last_post_at TEXT NOT NULL,
    last_post_id INTEGER
);

-- Per-channel replay checkpoint (listener/replay.py) -- the highest Telegram
-- message ID successfully processed through the normal pipeline (live or
-- replayed). Updated only on success; never regresses (see
-- set_channel_replay_state's MAX()), so a slow live message racing an
-- in-progress replay can't un-advance the checkpoint.
CREATE TABLE IF NOT EXISTS channel_replay_state (
    channel TEXT PRIMARY KEY,
    channel_id INTEGER,
    last_message_id INTEGER NOT NULL,
    last_processed_at TEXT NOT NULL
);

-- Per-day count of messages recovered via replay (startup or reconnect),
-- for /status -- separate from channel_stats' posts_received, which counts
-- live *and* replayed posts together.
CREATE TABLE IF NOT EXISTS replay_stats (
    stat_date TEXT PRIMARY KEY,
    recovered_count INTEGER NOT NULL DEFAULT 0
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
        # Migration for DBs created before external_quota_exhausted existed —
        # CREATE TABLE IF NOT EXISTS above is a no-op on an existing table,
        # so an already-created gemini_quota table needs the column added
        # explicitly. SQLite has no "ADD COLUMN IF NOT EXISTS", so just
        # swallow the "duplicate column" error on a DB that already has it.
        try:
            conn.execute(
                "ALTER TABLE gemini_quota ADD COLUMN external_quota_exhausted INTEGER NOT NULL DEFAULT 0"
            )
        except sqlite3.OperationalError:
            pass


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


def delete_all_tracked_products() -> int:
    """One-time reset for the explicit-tracking UX migration: wipes every
    tracked product for every user, leaving My Tracks empty. Doesn't touch
    users, gemini_quota, or duplicate_deals.
    """
    with get_connection() as conn:
        cursor = conn.execute("DELETE FROM tracked_products")
        return cursor.rowcount


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


def get_gemini_external_quota_exhausted(quota_date: str) -> bool:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT external_quota_exhausted FROM gemini_quota WHERE quota_date = ?", (quota_date,)
        ).fetchone()
        return bool(row["external_quota_exhausted"]) if row else False


def mark_gemini_external_quota_exhausted(quota_date: str) -> None:
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO gemini_quota (quota_date, call_count, external_quota_exhausted)
            VALUES (?, 0, 1)
            ON CONFLICT(quota_date) DO UPDATE SET external_quota_exhausted = 1
            """,
            (quota_date,),
        )


def reset_gemini_external_quota(quota_date: str) -> None:
    """Manual override — lets an admin declare the external quota available
    again without waiting for the date to roll over.
    """
    with get_connection() as conn:
        conn.execute(
            "UPDATE gemini_quota SET external_quota_exhausted = 0 WHERE quota_date = ?",
            (quota_date,),
        )


def get_groq_quota_count(quota_date: str) -> int:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT call_count FROM groq_quota WHERE quota_date = ?", (quota_date,)
        ).fetchone()
        return row["call_count"] if row else 0


def increment_groq_quota_count(quota_date: str) -> int:
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO groq_quota (quota_date, call_count) VALUES (?, 1)
            ON CONFLICT(quota_date) DO UPDATE SET call_count = call_count + 1
            """,
            (quota_date,),
        )
        row = conn.execute(
            "SELECT call_count FROM groq_quota WHERE quota_date = ?", (quota_date,)
        ).fetchone()
        return row["call_count"]


def get_groq_external_quota_exhausted(quota_date: str) -> bool:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT external_quota_exhausted FROM groq_quota WHERE quota_date = ?", (quota_date,)
        ).fetchone()
        return bool(row["external_quota_exhausted"]) if row else False


def mark_groq_external_quota_exhausted(quota_date: str) -> None:
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO groq_quota (quota_date, call_count, external_quota_exhausted)
            VALUES (?, 0, 1)
            ON CONFLICT(quota_date) DO UPDATE SET external_quota_exhausted = 1
            """,
            (quota_date,),
        )


def reset_groq_external_quota(quota_date: str) -> None:
    with get_connection() as conn:
        conn.execute(
            "UPDATE groq_quota SET external_quota_exhausted = 0 WHERE quota_date = ?",
            (quota_date,),
        )


def _bump_provider_stat(provider: str, stat_date: str, column: str, delta: float = 1) -> None:
    # `column` is always one of the fixed literals below (never user input),
    # so building the SQL with an f-string here is safe.
    assert column in (
        "successful_requests",
        "failed_requests",
        "quota_exhaustion_events",
        "circuit_breaker_activations",
        "total_latency_ms",
        "total_retries",
        "total_failovers",
    )
    with get_connection() as conn:
        conn.execute(
            f"""
            INSERT INTO provider_stats (provider, stat_date, {column}) VALUES (?, ?, ?)
            ON CONFLICT(provider, stat_date) DO UPDATE SET {column} = {column} + excluded.{column}
            """,
            (provider, stat_date, delta),
        )


def record_provider_success(provider: str, stat_date: str, latency_ms: float) -> None:
    _bump_provider_stat(provider, stat_date, "successful_requests", 1)
    _bump_provider_stat(provider, stat_date, "total_latency_ms", latency_ms)


def record_provider_failure(provider: str, stat_date: str) -> None:
    _bump_provider_stat(provider, stat_date, "failed_requests", 1)


def record_provider_quota_exhaustion(provider: str, stat_date: str) -> None:
    _bump_provider_stat(provider, stat_date, "quota_exhaustion_events", 1)


def record_provider_circuit_breaker_activation(provider: str, stat_date: str) -> None:
    _bump_provider_stat(provider, stat_date, "circuit_breaker_activations", 1)


def record_provider_retry(provider: str, stat_date: str) -> None:
    _bump_provider_stat(provider, stat_date, "total_retries", 1)


def record_provider_failover(provider: str, stat_date: str) -> None:
    _bump_provider_stat(provider, stat_date, "total_failovers", 1)


def get_provider_stats(provider: str, stat_date: str) -> dict:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM provider_stats WHERE provider = ? AND stat_date = ?",
            (provider, stat_date),
        ).fetchone()
    if row is None:
        return {
            "successful_requests": 0,
            "failed_requests": 0,
            "quota_exhaustion_events": 0,
            "circuit_breaker_activations": 0,
            "total_latency_ms": 0.0,
            "total_retries": 0,
            "total_failovers": 0,
        }
    return {
        "successful_requests": row["successful_requests"],
        "failed_requests": row["failed_requests"],
        "quota_exhaustion_events": row["quota_exhaustion_events"],
        "circuit_breaker_activations": row["circuit_breaker_activations"],
        "total_latency_ms": row["total_latency_ms"],
        "total_retries": row["total_retries"],
        "total_failovers": row["total_failovers"],
    }


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


# --- Self-improving knowledge engine (listener/learning.py) --------------


def insert_verdict(
    asin: str,
    provider: str,
    brand: str | None,
    category: str,
    title: str | None,
    current_price: float,
    discount_percent: int | None,
    deal_quality: str,
    reason: str | None,
    suggested_target: int | None,
    channel: str | None,
    timestamp: str,
) -> None:
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO verdicts (
                asin, provider, brand, category, title, current_price,
                discount_percent, deal_quality, reason, suggested_target,
                channel, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                asin, provider, brand, category, title, current_price,
                discount_percent, deal_quality, reason, suggested_target,
                channel, timestamp,
            ),
        )


def get_all_verdicts_chronological() -> list[sqlite3.Row]:
    """Used only by /rebuildrules to replay history from scratch."""
    with get_connection() as conn:
        return conn.execute("SELECT * FROM verdicts ORDER BY timestamp ASC, id ASC").fetchall()


def category_seen_before(category: str) -> bool:
    """Whether any verdict has ever been recorded for this category —
    listener.learning uses this to treat a genuinely new category as an
    outlier (always call AI) rather than guessing from zero history.
    """
    with get_connection() as conn:
        row = conn.execute("SELECT 1 FROM verdicts WHERE category = ? LIMIT 1", (category,)).fetchone()
        return row is not None


def get_learned_rule(rule_type: str, key: str) -> sqlite3.Row | None:
    with get_connection() as conn:
        return conn.execute(
            "SELECT * FROM learned_rules WHERE rule_type = ? AND key = ?", (rule_type, key)
        ).fetchone()


def upsert_learned_rule(
    rule_type: str,
    key: str,
    predicted_quality: str,
    confidence: float,
    sample_count: int,
    last_updated: str,
    last_validated: str | None,
    enabled: bool,
    rule_version: int,
) -> None:
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO learned_rules (
                rule_type, key, predicted_quality, confidence, sample_count,
                last_validated, last_updated, rule_version, enabled
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(rule_type, key) DO UPDATE SET
                predicted_quality = excluded.predicted_quality,
                confidence = excluded.confidence,
                sample_count = excluded.sample_count,
                last_validated = excluded.last_validated,
                last_updated = excluded.last_updated,
                rule_version = excluded.rule_version,
                enabled = excluded.enabled
            """,
            (
                rule_type, key, predicted_quality, confidence, sample_count,
                last_validated, last_updated, rule_version, 1 if enabled else 0,
            ),
        )


def set_rule_last_validated(rule_type: str, key: str, last_validated: str) -> None:
    with get_connection() as conn:
        conn.execute(
            "UPDATE learned_rules SET last_validated = ? WHERE rule_type = ? AND key = ?",
            (last_validated, rule_type, key),
        )


def get_rule_votes(rule_type: str, key: str) -> dict[str, float]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT quality, weight FROM rule_votes WHERE rule_type = ? AND key = ?", (rule_type, key)
        ).fetchall()
    return {row["quality"]: row["weight"] for row in rows}


def set_rule_votes(rule_type: str, key: str, votes: dict[str, float]) -> None:
    with get_connection() as conn:
        for quality, weight in votes.items():
            conn.execute(
                """
                INSERT INTO rule_votes (rule_type, key, quality, weight) VALUES (?, ?, ?, ?)
                ON CONFLICT(rule_type, key, quality) DO UPDATE SET weight = excluded.weight
                """,
                (rule_type, key, quality, weight),
            )


def list_learned_rules(enabled_only: bool = False) -> list[sqlite3.Row]:
    query = "SELECT * FROM learned_rules"
    if enabled_only:
        query += " WHERE enabled = 1"
    query += " ORDER BY confidence DESC, sample_count DESC"
    with get_connection() as conn:
        return conn.execute(query).fetchall()


def count_rules_by_type() -> dict[str, int]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT rule_type, COUNT(*) AS n FROM learned_rules WHERE enabled = 1 GROUP BY rule_type"
        ).fetchall()
    return {row["rule_type"]: row["n"] for row in rows}


def clear_learned_rules() -> None:
    """Used by /resetrules and /rebuildrules — leaves verdict history intact."""
    with get_connection() as conn:
        conn.execute("DELETE FROM learned_rules")
        conn.execute("DELETE FROM rule_votes")


def get_kb_version() -> int:
    with get_connection() as conn:
        row = conn.execute("SELECT value FROM learning_meta WHERE key = 'kb_version'").fetchone()
    return int(row["value"]) if row else 1


def bump_kb_version() -> int:
    new_version = get_kb_version() + 1
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO learning_meta (key, value) VALUES ('kb_version', ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            (str(new_version),),
        )
    return new_version


def _bump_learning_stat(stat_date: str, column: str, delta: int = 1) -> None:
    assert column in (
        "ai_calls_saved",
        "rule_hits",
        "rule_misses",
        "validation_calls",
        "validation_agreements",
        "validation_disagreements",
    )
    with get_connection() as conn:
        conn.execute(
            f"""
            INSERT INTO learning_stats (stat_date, {column}) VALUES (?, ?)
            ON CONFLICT(stat_date) DO UPDATE SET {column} = {column} + excluded.{column}
            """,
            (stat_date, delta),
        )


def record_ai_call_saved(stat_date: str) -> None:
    _bump_learning_stat(stat_date, "ai_calls_saved")


def record_rule_hit(stat_date: str) -> None:
    _bump_learning_stat(stat_date, "rule_hits")


def record_rule_miss(stat_date: str) -> None:
    _bump_learning_stat(stat_date, "rule_misses")


def record_validation_call(stat_date: str) -> None:
    _bump_learning_stat(stat_date, "validation_calls")


def record_validation_agreement(stat_date: str) -> None:
    _bump_learning_stat(stat_date, "validation_agreements")


def record_validation_disagreement(stat_date: str) -> None:
    _bump_learning_stat(stat_date, "validation_disagreements")


def get_learning_stats(stat_date: str) -> dict:
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM learning_stats WHERE stat_date = ?", (stat_date,)).fetchone()
    if row is None:
        return {
            "ai_calls_saved": 0,
            "rule_hits": 0,
            "rule_misses": 0,
            "validation_calls": 0,
            "validation_agreements": 0,
            "validation_disagreements": 0,
        }
    return {
        "ai_calls_saved": row["ai_calls_saved"],
        "rule_hits": row["rule_hits"],
        "rule_misses": row["rule_misses"],
        "validation_calls": row["validation_calls"],
        "validation_agreements": row["validation_agreements"],
        "validation_disagreements": row["validation_disagreements"],
    }


_CHANNEL_STAT_COLUMNS = (
    "posts_received",
    "parsed",
    "forwarded",
    "duplicates",
    "no_price_failures",
    "no_asin_failures",
    "non_amazon_links",
    "ai_analyses",
    "rule_hits",
    "total_failures",
)


def _bump_channel_stat(channel: str, stat_date: str, column: str, delta: float = 1) -> None:
    assert column in _CHANNEL_STAT_COLUMNS + ("total_latency_ms", "latency_count", "total_ai_latency_ms", "ai_latency_count")
    with get_connection() as conn:
        conn.execute(
            f"""
            INSERT INTO channel_stats (channel, stat_date, {column}) VALUES (?, ?, ?)
            ON CONFLICT(channel, stat_date) DO UPDATE SET {column} = {column} + excluded.{column}
            """,
            (channel, stat_date, delta),
        )


def record_channel_post_received(channel: str, stat_date: str) -> None:
    _bump_channel_stat(channel, stat_date, "posts_received")


def record_channel_parsed(channel: str, stat_date: str) -> None:
    _bump_channel_stat(channel, stat_date, "parsed")


def record_channel_forwarded(channel: str, stat_date: str) -> None:
    _bump_channel_stat(channel, stat_date, "forwarded")


def record_channel_duplicate(channel: str, stat_date: str) -> None:
    _bump_channel_stat(channel, stat_date, "duplicates")


def record_channel_no_price(channel: str, stat_date: str) -> None:
    _bump_channel_stat(channel, stat_date, "no_price_failures")


def record_channel_no_asin(channel: str, stat_date: str) -> None:
    _bump_channel_stat(channel, stat_date, "no_asin_failures")


def record_channel_non_amazon_link(channel: str, stat_date: str) -> None:
    _bump_channel_stat(channel, stat_date, "non_amazon_links")


def record_channel_ai_analysis(channel: str, stat_date: str) -> None:
    _bump_channel_stat(channel, stat_date, "ai_analyses")


def record_channel_rule_hit(channel: str, stat_date: str) -> None:
    _bump_channel_stat(channel, stat_date, "rule_hits")


def record_channel_failure(channel: str, stat_date: str) -> None:
    _bump_channel_stat(channel, stat_date, "total_failures")


def record_channel_latency(channel: str, stat_date: str, latency_ms: float) -> None:
    _bump_channel_stat(channel, stat_date, "total_latency_ms", latency_ms)
    _bump_channel_stat(channel, stat_date, "latency_count", 1)


def record_channel_ai_latency(channel: str, stat_date: str, latency_ms: float) -> None:
    _bump_channel_stat(channel, stat_date, "total_ai_latency_ms", latency_ms)
    _bump_channel_stat(channel, stat_date, "ai_latency_count", 1)


def get_channel_stats(channel: str, stat_date: str) -> dict:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM channel_stats WHERE channel = ? AND stat_date = ?", (channel, stat_date)
        ).fetchone()
    if row is None:
        return {col: 0 for col in _CHANNEL_STAT_COLUMNS} | {
            "total_latency_ms": 0.0, "latency_count": 0, "total_ai_latency_ms": 0.0, "ai_latency_count": 0,
        }
    return {col: row[col] for col in _CHANNEL_STAT_COLUMNS} | {
        "total_latency_ms": row["total_latency_ms"],
        "latency_count": row["latency_count"],
        "total_ai_latency_ms": row["total_ai_latency_ms"],
        "ai_latency_count": row["ai_latency_count"],
    }


def get_channel_stats_range(channel: str, start_date: str, end_date: str) -> dict:
    """Summed counters for `channel` over [start_date, end_date] inclusive —
    used by the watchdog to compute a 7-day average posting frequency.
    """
    with get_connection() as conn:
        row = conn.execute(
            """
            SELECT COALESCE(SUM(posts_received), 0) AS posts_received
            FROM channel_stats WHERE channel = ? AND stat_date BETWEEN ? AND ?
            """,
            (channel, start_date, end_date),
        ).fetchone()
    return {"posts_received": row["posts_received"]}


def get_channel_last_post(channel: str) -> sqlite3.Row | None:
    with get_connection() as conn:
        return conn.execute(
            "SELECT * FROM channel_last_post WHERE channel = ?", (channel,)
        ).fetchone()


def record_channel_last_post(channel: str, posted_at: str, post_id: int | None) -> None:
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO channel_last_post (channel, last_post_at, last_post_id) VALUES (?, ?, ?)
            ON CONFLICT(channel) DO UPDATE SET last_post_at = excluded.last_post_at, last_post_id = excluded.last_post_id
            """,
            (channel, posted_at, post_id),
        )


def get_channel_replay_state(channel: str) -> sqlite3.Row | None:
    with get_connection() as conn:
        return conn.execute(
            "SELECT * FROM channel_replay_state WHERE channel = ?", (channel,)
        ).fetchone()


def set_channel_replay_state(
    channel: str, channel_id: int | None, last_message_id: int, last_processed_at: str
) -> None:
    """Only ever advances last_message_id (MAX with whatever's already
    stored) -- a slower live message for an older ID racing an in-progress
    replay for a newer one (or vice versa) can never regress the checkpoint.
    """
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO channel_replay_state (channel, channel_id, last_message_id, last_processed_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(channel) DO UPDATE SET
                channel_id = COALESCE(excluded.channel_id, channel_replay_state.channel_id),
                last_message_id = MAX(excluded.last_message_id, channel_replay_state.last_message_id),
                last_processed_at = CASE
                    WHEN excluded.last_message_id >= channel_replay_state.last_message_id
                    THEN excluded.last_processed_at ELSE channel_replay_state.last_processed_at
                END
            """,
            (channel, channel_id, last_message_id, last_processed_at),
        )


def record_replay_recovered(stat_date: str, count: int = 1) -> None:
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO replay_stats (stat_date, recovered_count) VALUES (?, ?)
            ON CONFLICT(stat_date) DO UPDATE SET recovered_count = recovered_count + excluded.recovered_count
            """,
            (stat_date, count),
        )


def get_replay_recovered(stat_date: str) -> int:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT recovered_count FROM replay_stats WHERE stat_date = ?", (stat_date,)
        ).fetchone()
    return row["recovered_count"] if row else 0


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
