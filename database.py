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

-- current_price doubles as "last seen price" (updated on every check cycle
-- regardless of whether a notification fires); last_checked doubles as
-- "last checked at". last_notified_price is shared by both the general
-- price-change notifier and the target-reached alert (see scheduler.py) --
-- both only ever set it to the price they just notified about, so it always
-- reflects "the price of the most recent notification of any kind".
-- available tracks the price-change state machine's third dimension
-- (in stock / unavailable) alongside price.
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
    available INTEGER NOT NULL DEFAULT 1,
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

-- DEPRECATED (kept only for historical data — no longer read or written).
-- Its (channel_name, identifier) primary key scoped duplicate detection
-- *per channel*, which was the root cause of the same product being
-- forwarded once per channel that posted it. Replaced by
-- global_duplicate_deals below, keyed on the canonical product identity
-- alone. Left in place rather than dropped/migrated to avoid any risk to
-- existing production data.
CREATE TABLE IF NOT EXISTS duplicate_deals (
    channel_name TEXT NOT NULL,
    identifier TEXT NOT NULL,
    last_price REAL,
    last_discount_percent INTEGER,
    last_seen_at TEXT NOT NULL,
    PRIMARY KEY (channel_name, identifier)
);

-- Last-seen state per canonical product (never per-channel) for duplicate
-- detection. identifier priority: "asin:<ASIN>" (the canonical Amazon
-- product identifier) when resolved, else "title:<normalized
-- title>|price:<normalized price>" as a fallback only when no ASIN could be
-- determined. The same product posted by two different channels now
-- correctly matches the same row, so only the first forwards.
CREATE TABLE IF NOT EXISTS global_duplicate_deals (
    identifier TEXT PRIMARY KEY,
    last_channel_name TEXT NOT NULL,
    last_price REAL,
    last_discount_percent INTEGER,
    last_seen_at TEXT NOT NULL
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

-- Product Family detection (listener/family.py). One row per detected
-- family of related Amazon ASINs (color/size/capacity/pack variants of the
-- same underlying product). normalized_title has all recognized variant
-- tokens stripped, so different colors of the same product share it.
-- deterministic_key is a hash of (brand, normalized_title, category) --
-- the fast-path exact-match signal. anchor_asin is the first ASIN that
-- created this family, used as the stable "other half" of the
-- (asin, anchor_asin) pair key for permanently caching AI similarity
-- decisions (family_ai_decisions below). lowest_price/highest_discount_
-- percent/best_verdict_quality/best_variant_label/best_variant_asin track
-- the best variant seen so far, updated every time a member's numbers beat
-- them -- this is what "Better Variant Found" vs "New Variant Available"
-- is decided against.
-- last_verdict_* columns (listener/family.py's AI-verdict cache, added
-- alongside listener/budget.py) hold the most recent REAL (non-cached,
-- non-rule) AI verdict obtained for any member of this family, so a new
-- variant arriving within FAMILY_VERDICT_CACHE_WINDOW_HOURS can reuse it
-- (provider="family_cache") instead of spending another AI call, unless
-- price/discount moved significantly or a genuinely new variant-attribute
-- kind appeared (last_verdict_variant_keys, a JSON list of attribute keys
-- e.g. ["color"] seen at cache time, compared against the new variant's
-- own keys).
CREATE TABLE IF NOT EXISTS product_families (
    family_id TEXT PRIMARY KEY,
    brand TEXT,
    normalized_title TEXT NOT NULL,
    category TEXT,
    deterministic_key TEXT NOT NULL,
    anchor_asin TEXT NOT NULL,
    lowest_price REAL,
    highest_discount_percent INTEGER,
    best_verdict_quality TEXT,
    best_variant_label TEXT,
    best_variant_asin TEXT,
    variant_count INTEGER NOT NULL DEFAULT 0,
    first_seen TEXT NOT NULL,
    last_seen TEXT NOT NULL,
    last_verdict_quality TEXT,
    last_verdict_reason TEXT,
    last_verdict_suggested_target INTEGER,
    last_verdict_category TEXT,
    last_verdict_provider TEXT,
    last_verdict_at TEXT,
    last_verdict_price REAL,
    last_verdict_discount_percent INTEGER,
    last_verdict_variant_keys TEXT
);

CREATE INDEX IF NOT EXISTS idx_product_families_key ON product_families(deterministic_key);
CREATE INDEX IF NOT EXISTS idx_product_families_brand_category ON product_families(brand, category);

-- One row per ASIN ever assigned to a family -- variant_json holds the
-- structured attributes extracted from that ASIN's own title (e.g.
-- {"color": "Red", "size": "42"}), never removed from the stored title
-- itself. last_price/last_discount_percent/last_seen_at are this specific
-- ASIN's own last-seen numbers (as opposed to product_families' family-wide
-- best), used to detect a true same-variant repost (see family.py's
-- duplicate-suppression check, which compares against the most recent
-- member sharing the same variant_json, not necessarily this exact ASIN).
CREATE TABLE IF NOT EXISTS family_members (
    asin TEXT PRIMARY KEY,
    family_id TEXT NOT NULL REFERENCES product_families(family_id),
    variant_json TEXT NOT NULL DEFAULT '{}',
    last_price REAL,
    last_discount_percent INTEGER,
    last_seen_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_family_members_family ON family_members(family_id);

-- Permanent cache of AI family-similarity decisions, keyed by the
-- (asin_a, asin_b) pair with asin_a < asin_b (sorted, so lookup order never
-- matters) -- once asked about a specific pair of ASINs, family.py never
-- asks AI about that exact pair again, regardless of outcome (true or
-- false answers are both cached).
CREATE TABLE IF NOT EXISTS family_ai_decisions (
    asin_a TEXT NOT NULL,
    asin_b TEXT NOT NULL,
    same_family INTEGER NOT NULL,
    confidence REAL NOT NULL,
    reason TEXT,
    decided_at TEXT NOT NULL,
    PRIMARY KEY (asin_a, asin_b)
);

-- Daily AI-call priority-tier counters (listener/budget.py) -- how many
-- incoming deals were classified Priority 1/2/3 today, for /status'
-- Budget section and for estimating deals/hour.
-- shadow_total/shadow_divergences (listener/scoring.py shadow mode) count
-- how many deals had both the legacy and scored classifiers evaluated
-- side-by-side today, and how many of those disagreed -- the /status
-- divergence-rate signal used to validate the score engine before
-- SCORE_ENGINE_ENABLED is ever flipped on.
CREATE TABLE IF NOT EXISTS priority_stats (
    stat_date TEXT PRIMARY KEY,
    priority_1 INTEGER NOT NULL DEFAULT 0,
    priority_2 INTEGER NOT NULL DEFAULT 0,
    priority_3 INTEGER NOT NULL DEFAULT 0,
    shadow_total INTEGER NOT NULL DEFAULT 0,
    shadow_divergences INTEGER NOT NULL DEFAULT 0
);

-- Deterministic brand-reputation scores (listener/scoring.py). One row per
-- brand ever seen in `verdicts`, holding a decay-weighted mean of that
-- brand's historical AI verdict qualities (skip=0.0, average=1/3, good=2/3,
-- great=1.0), using the same monthly-decay style as listener/learning.py's
-- _update_rule. Fully recomputed from `verdicts` by
-- backfill_brand_reputation() -- idempotent, safe to re-run any time, and
-- re-run on every init_db() so it never goes stale between deploys.
CREATE TABLE IF NOT EXISTS brand_reputation (
    brand TEXT PRIMARY KEY,
    decayed_quality_mean REAL NOT NULL,
    decayed_weight REAL NOT NULL,
    sample_count INTEGER NOT NULL DEFAULT 0,
    last_updated TIMESTAMP NOT NULL
);

-- Append-only price history feeding listener/scoring.py's historical price
-- percentile. One row per observed (asin, price) sighting -- written from
-- both the deal-forwarding path (listener/watcher.py, right after Product
-- Family pre_check so family_id is known) and the tracked-product path
-- (update_price_check below). Never updated in place. family_id is TEXT
-- (matching product_families.family_id, e.g. "fam_<hex>"), NULL when the
-- observation has no family context (tracked-product checks).
CREATE TABLE IF NOT EXISTS price_observations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    asin TEXT NOT NULL,
    family_id TEXT,
    price REAL NOT NULL,
    discount_percent REAL,
    observed_at TIMESTAMP NOT NULL
);

-- Covering indexes (asin/family_id + price) so listener/scoring.py's
-- MIN/MAX/COUNT aggregates are answered entirely from the index, never
-- touching table rows -- this is what keeps the per-deal Value Score cost
-- effectively free (benchmarked: 1,000 scores well under 50 ms).
CREATE INDEX IF NOT EXISTS idx_price_observations_asin ON price_observations(asin, price);
CREATE INDEX IF NOT EXISTS idx_price_observations_family ON price_observations(family_id, price);
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

        # Migration for DBs created before per-product availability tracking
        # existed (see scheduler.py's price-change state machine). Existing
        # rows default to available=1 (in stock) -- this does NOT itself
        # trigger a notification; the next check cycle simply compares
        # against it like any other state, and current_price already holds
        # each product's last-seen price from every prior check, so no
        # separate backfill is needed for that half of the migration.
        try:
            conn.execute("ALTER TABLE tracked_products ADD COLUMN available INTEGER NOT NULL DEFAULT 1")
        except sqlite3.OperationalError:
            pass

        # Migration for product_families rows created before the AI-verdict
        # cache (listener/family.py + listener/budget.py) existed -- adds
        # each last_verdict_* column individually since SQLite has no
        # multi-column ALTER TABLE ADD COLUMN. Existing rows get NULL,
        # which get_cached_verdict() already treats as "no cached verdict
        # yet" (falls through to a real AI call), so no backfill is needed.
        for column, coltype in (
            ("last_verdict_quality", "TEXT"),
            ("last_verdict_reason", "TEXT"),
            ("last_verdict_suggested_target", "INTEGER"),
            ("last_verdict_category", "TEXT"),
            ("last_verdict_provider", "TEXT"),
            ("last_verdict_at", "TEXT"),
            ("last_verdict_price", "REAL"),
            ("last_verdict_discount_percent", "INTEGER"),
            ("last_verdict_variant_keys", "TEXT"),
        ):
            try:
                conn.execute(f"ALTER TABLE product_families ADD COLUMN {column} {coltype}")
            except sqlite3.OperationalError:
                pass

        # Migration for priority_stats rows created before score-engine
        # shadow mode existed (listener/scoring.py) -- additive columns
        # only, same swallow-duplicate-column pattern as above.
        for column in ("shadow_total", "shadow_divergences"):
            try:
                conn.execute(
                    f"ALTER TABLE priority_stats ADD COLUMN {column} INTEGER NOT NULL DEFAULT 0"
                )
            except sqlite3.OperationalError:
                pass

    # Recomputed from `verdicts` on every startup -- idempotent full
    # rebuild, so brand reputation never drifts stale between deploys and
    # re-running init_db (tests do this constantly) is always safe.
    backfill_brand_reputation()


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


def update_price_check(product_id: int, current_price: float | None, available: bool = True) -> None:
    with get_connection() as conn:
        conn.execute(
            "UPDATE tracked_products SET current_price = ?, available = ?, last_checked = datetime('now') WHERE id = ?",
            (current_price, 1 if available else 0, product_id),
        )
        # Append-only price history for listener/scoring.py -- the
        # tracked-product half of price_observations (the deal-forwarding
        # half is written in listener/watcher.py). No family context here;
        # no row at all when the product was unavailable (price is None).
        if current_price is not None:
            row = conn.execute(
                "SELECT asin FROM tracked_products WHERE id = ?", (product_id,)
            ).fetchone()
            if row is not None:
                conn.execute(
                    """
                    INSERT INTO price_observations (asin, family_id, price, discount_percent, observed_at)
                    VALUES (?, NULL, ?, NULL, datetime('now'))
                    """,
                    (row["asin"], current_price),
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


def get_global_duplicate_record(identifier: str) -> sqlite3.Row | None:
    with get_connection() as conn:
        return conn.execute(
            """
            SELECT last_price, last_discount_percent, last_seen_at
            FROM global_duplicate_deals WHERE identifier = ?
            """,
            (identifier,),
        ).fetchone()


def upsert_global_duplicate_record(
    identifier: str,
    channel_name: str,
    price: float | None,
    discount_percent: int | None,
    seen_at: str,
) -> None:
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO global_duplicate_deals (identifier, last_channel_name, last_price, last_discount_percent, last_seen_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(identifier) DO UPDATE SET
                last_channel_name = excluded.last_channel_name,
                last_price = excluded.last_price,
                last_discount_percent = excluded.last_discount_percent,
                last_seen_at = excluded.last_seen_at
            """,
            (identifier, channel_name, price, discount_percent, seen_at),
        )


def count_active_global_duplicate_records(cutoff_iso: str) -> int:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM global_duplicate_deals WHERE last_seen_at >= ?", (cutoff_iso,)
        ).fetchone()
        return row["n"]


def delete_expired_global_duplicate_records(cutoff_iso: str) -> int:
    with get_connection() as conn:
        cursor = conn.execute("DELETE FROM global_duplicate_deals WHERE last_seen_at < ?", (cutoff_iso,))
        return cursor.rowcount


def delete_expired_duplicate_records(cutoff_iso: str) -> int:
    """Deprecated alongside duplicate_deals (see schema comment) — no longer
    called; kept only so any lingering reference doesn't hard-fail.
    """
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
        available=bool(row["available"]),
        created_at=row["created_at"],
    )


# --- Product Family detection (listener/family.py) -----------------------


def get_family_member(asin: str) -> sqlite3.Row | None:
    with get_connection() as conn:
        return conn.execute(
            "SELECT * FROM family_members WHERE asin = ?", (asin,)
        ).fetchone()


def get_product_family(family_id: str) -> sqlite3.Row | None:
    with get_connection() as conn:
        return conn.execute(
            "SELECT * FROM product_families WHERE family_id = ?", (family_id,)
        ).fetchone()


def get_family_by_deterministic_key(deterministic_key: str) -> sqlite3.Row | None:
    with get_connection() as conn:
        return conn.execute(
            "SELECT * FROM product_families WHERE deterministic_key = ?", (deterministic_key,)
        ).fetchone()


def get_families_by_brand_category(brand: str | None, category: str | None) -> list[sqlite3.Row]:
    """Candidate families for fuzzy/AI title-similarity matching -- narrowed
    to the same brand and category (NULL-safe equality) since brand/category
    are required matching signals, never skipped in favor of title alone.
    """
    with get_connection() as conn:
        return conn.execute(
            """
            SELECT * FROM product_families
            WHERE brand IS ? AND category IS ?
            """,
            (brand, category),
        ).fetchall()


def create_product_family(
    family_id: str,
    brand: str | None,
    normalized_title: str,
    category: str | None,
    deterministic_key: str,
    anchor_asin: str,
    lowest_price: float | None,
    highest_discount_percent: int | None,
    first_seen: str,
    last_seen: str,
    best_variant_label: str | None = None,
) -> None:
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO product_families (
                family_id, brand, normalized_title, category, deterministic_key,
                anchor_asin, lowest_price, highest_discount_percent,
                best_variant_label, best_variant_asin, variant_count, first_seen, last_seen
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?)
            """,
            (
                family_id, brand, normalized_title, category, deterministic_key,
                anchor_asin, lowest_price, highest_discount_percent, best_variant_label, anchor_asin,
                first_seen, last_seen,
            ),
        )


def upsert_family_member(
    asin: str, family_id: str, variant_json: str,
    price: float | None, discount_percent: int | None, seen_at: str,
) -> bool:
    """Returns True the first time this ASIN is recorded in family_members
    (a brand-new distinct member), False on a subsequent update to an
    already-known member -- callers use this to decide whether to bump
    product_families.variant_count.
    """
    with get_connection() as conn:
        is_new = conn.execute(
            "SELECT 1 FROM family_members WHERE asin = ?", (asin,)
        ).fetchone() is None
        conn.execute(
            """
            INSERT INTO family_members (asin, family_id, variant_json, last_price, last_discount_percent, last_seen_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(asin) DO UPDATE SET
                variant_json = excluded.variant_json,
                last_price = excluded.last_price,
                last_discount_percent = excluded.last_discount_percent,
                last_seen_at = excluded.last_seen_at
            """,
            (asin, family_id, variant_json, price, discount_percent, seen_at),
        )
        if is_new:
            conn.execute(
                "UPDATE product_families SET variant_count = variant_count + 1 WHERE family_id = ?",
                (family_id,),
            )
        return is_new


def get_family_member_by_variant(family_id: str, variant_json: str) -> sqlite3.Row | None:
    """Most recently seen member of this family with the exact same variant
    signature -- used for true-duplicate suppression (see family.py). Not
    necessarily the same ASIN: a different reseller listing of the identical
    color/size still counts.
    """
    with get_connection() as conn:
        return conn.execute(
            """
            SELECT * FROM family_members
            WHERE family_id = ? AND variant_json = ?
            ORDER BY last_seen_at DESC LIMIT 1
            """,
            (family_id, variant_json),
        ).fetchone()


def touch_product_family(family_id: str, seen_at: str) -> None:
    with get_connection() as conn:
        conn.execute(
            "UPDATE product_families SET last_seen = ? WHERE family_id = ?", (seen_at, family_id)
        )


def update_product_family_aggregates(
    family_id: str,
    lowest_price: float | None,
    highest_discount_percent: int | None,
    best_verdict_quality: str | None,
    best_variant_label: str | None,
    best_variant_asin: str | None,
    last_seen: str,
) -> None:
    with get_connection() as conn:
        conn.execute(
            """
            UPDATE product_families SET
                lowest_price = ?, highest_discount_percent = ?, best_verdict_quality = ?,
                best_variant_label = ?, best_variant_asin = ?, last_seen = ?
            WHERE family_id = ?
            """,
            (
                lowest_price, highest_discount_percent, best_verdict_quality,
                best_variant_label, best_variant_asin, last_seen, family_id,
            ),
        )


def get_family_ai_decision(asin_a: str, asin_b: str) -> sqlite3.Row | None:
    a, b = sorted((asin_a, asin_b))
    with get_connection() as conn:
        return conn.execute(
            "SELECT * FROM family_ai_decisions WHERE asin_a = ? AND asin_b = ?", (a, b)
        ).fetchone()


def store_family_ai_decision(
    asin_a: str, asin_b: str, same_family: bool, confidence: float, reason: str, decided_at: str,
) -> None:
    a, b = sorted((asin_a, asin_b))
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO family_ai_decisions (asin_a, asin_b, same_family, confidence, reason, decided_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(asin_a, asin_b) DO NOTHING
            """,
            (a, b, int(same_family), confidence, reason, decided_at),
        )


def count_product_families() -> int:
    with get_connection() as conn:
        row = conn.execute("SELECT COUNT(*) AS n FROM product_families").fetchone()
        return row["n"]


def record_family_verdict(
    family_id: str,
    quality: str,
    reason: str,
    suggested_target: int,
    category: str,
    provider: str,
    price: float,
    discount_percent: int | None,
    variant_keys_json: str,
    decided_at: str,
) -> None:
    with get_connection() as conn:
        conn.execute(
            """
            UPDATE product_families SET
                last_verdict_quality = ?, last_verdict_reason = ?, last_verdict_suggested_target = ?,
                last_verdict_category = ?, last_verdict_provider = ?, last_verdict_at = ?,
                last_verdict_price = ?, last_verdict_discount_percent = ?, last_verdict_variant_keys = ?
            WHERE family_id = ?
            """,
            (
                quality, reason, suggested_target, category, provider, decided_at,
                price, discount_percent, variant_keys_json, family_id,
            ),
        )


# --- Daily AI budget manager (listener/budget.py) -------------------------


def record_priority_classification(stat_date: str, priority: int) -> None:
    column = {1: "priority_1", 2: "priority_2", 3: "priority_3"}[priority]
    with get_connection() as conn:
        conn.execute(
            f"""
            INSERT INTO priority_stats (stat_date, {column}) VALUES (?, 1)
            ON CONFLICT(stat_date) DO UPDATE SET {column} = {column} + 1
            """,
            (stat_date,),
        )


def get_priority_counts(stat_date: str) -> dict:
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM priority_stats WHERE stat_date = ?", (stat_date,)).fetchone()
    if row is None:
        return {1: 0, 2: 0, 3: 0}
    return {1: row["priority_1"], 2: row["priority_2"], 3: row["priority_3"]}


def get_alltime_calls_saved() -> int:
    with get_connection() as conn:
        row = conn.execute("SELECT COALESCE(SUM(ai_calls_saved), 0) AS n FROM learning_stats").fetchone()
        return row["n"]


def get_alltime_ai_calls_used() -> int:
    with get_connection() as conn:
        gemini = conn.execute("SELECT COALESCE(SUM(call_count), 0) AS n FROM gemini_quota").fetchone()["n"]
        groq = conn.execute("SELECT COALESCE(SUM(call_count), 0) AS n FROM groq_quota").fetchone()["n"]
        return gemini + groq


# --- Value Score engine (listener/scoring.py) -----------------------------

# Same quality scale everywhere in scoring: worst 0.0 .. best 1.0.
_QUALITY_VALUE = {"skip": 0.0, "average": 1 / 3, "good": 2 / 3, "great": 1.0}

_SCORING_SECONDS_PER_MONTH = 30.44 * 86400  # matches listener/learning.py


# Bumped every time backfill_brand_reputation rewrites the table -- the
# only way brand_reputation ever changes. listener/scoring.py compares this
# against the generation it cached at, so its in-memory copy of the (small)
# table invalidates exactly when the data does, with no import cycle.
brand_reputation_generation = 0


def backfill_brand_reputation(now: "datetime | None" = None) -> int:
    """Full, idempotent rebuild of brand_reputation from `verdicts`, using
    the same monthly-decay style as listener/learning.py's _update_rule:
    each verdict contributes weight RULE_MONTHLY_DECAY ** months_elapsed.
    DELETE + re-insert, so running it twice (or two hundred times) always
    converges to the identical rows -- never duplicates. Returns the number
    of brands written.
    """
    from datetime import datetime as _datetime

    from config import RULE_MONTHLY_DECAY

    now = now or _datetime.now()
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT brand, deal_quality, timestamp FROM verdicts WHERE brand IS NOT NULL"
        ).fetchall()

        totals: dict[str, list[float]] = {}  # brand -> [weighted_quality_sum, weight_sum, count]
        for row in rows:
            quality_value = _QUALITY_VALUE.get(row["deal_quality"])
            if quality_value is None:
                continue
            try:
                verdict_at = _datetime.fromisoformat(row["timestamp"])
            except ValueError:
                continue
            elapsed_months = max(0.0, (now - verdict_at).total_seconds() / _SCORING_SECONDS_PER_MONTH)
            weight = RULE_MONTHLY_DECAY ** elapsed_months
            entry = totals.setdefault(row["brand"], [0.0, 0.0, 0])
            entry[0] += weight * quality_value
            entry[1] += weight
            entry[2] += 1

        conn.execute("DELETE FROM brand_reputation")
        for brand, (weighted_sum, weight_sum, count) in totals.items():
            if weight_sum <= 0:
                continue
            conn.execute(
                """
                INSERT INTO brand_reputation (brand, decayed_quality_mean, decayed_weight, sample_count, last_updated)
                VALUES (?, ?, ?, ?, ?)
                """,
                (brand, weighted_sum / weight_sum, weight_sum, count, now.isoformat()),
            )

    global brand_reputation_generation
    brand_reputation_generation += 1
    return len(totals)


def get_brand_reputation(brand: str, conn: sqlite3.Connection | None = None) -> sqlite3.Row | None:
    if conn is not None:
        return conn.execute(
            "SELECT * FROM brand_reputation WHERE brand = ?", (brand,)
        ).fetchone()
    with get_connection() as owned:
        return owned.execute(
            "SELECT * FROM brand_reputation WHERE brand = ?", (brand,)
        ).fetchone()


def record_price_observation(
    asin: str, family_id: str | None, price: float, discount_percent: float | None, observed_at: str,
) -> None:
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO price_observations (asin, family_id, price, discount_percent, observed_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (asin, family_id, price, discount_percent, observed_at),
        )


# n only ever feeds saturating threshold checks in listener/scoring.py
# (rarity saturation, "fewer than two points has no spread") -- never exact
# arithmetic -- so family counting stops at this cap instead of scanning a
# dense family's entire history on every score.
_OBSERVATION_COUNT_CAP = 20

_ASIN_OBSERVATION_STATS_SQL = (
    "SELECT MIN(price) AS min_price, MAX(price) AS max_price, COUNT(*) AS n "
    "FROM price_observations WHERE asin = ?"
)
# The family branch deliberately uses three tiny statements instead of one
# MIN/MAX/COUNT aggregate: with the covering (family_id, price) index,
# standalone MIN and MAX are each a single index seek, while a combined
# aggregate must scan every entry for the COUNT and loses that optimization
# -- measured several times slower per call on a dense family. (The asin
# branch keeps the single aggregate: one ASIN's history is small enough
# that a single short scan beats three round-trips.)
_FAMILY_MIN_SQL = "SELECT MIN(price) AS v FROM price_observations WHERE family_id = ?"
_FAMILY_MAX_SQL = "SELECT MAX(price) AS v FROM price_observations WHERE family_id = ?"
_FAMILY_COUNT_CAPPED_SQL = (
    "SELECT COUNT(*) AS n FROM (SELECT 1 FROM price_observations WHERE family_id = ? LIMIT ?)"
)


def get_price_observation_stats(
    asin: str, family_id: str | None, conn: sqlite3.Connection | None = None,
) -> dict:
    """MIN/MAX/COUNT over this ASIN's own observations plus (when a family
    is known) every sibling ASIN's observations, all via the covering
    (asin, price)/(family_id, price) indexes -- an `asin = ? OR family_id
    = ?` form would defeat both. A row carrying both this asin and this
    family_id counts twice in n, and the family count saturates at
    _OBSERVATION_COUNT_CAP -- both harmless for n's threshold-only use.
    """
    def _query(c: sqlite3.Connection) -> dict:
        asin_row = c.execute(_ASIN_OBSERVATION_STATS_SQL, (asin,)).fetchone()
        min_price, max_price = asin_row["min_price"], asin_row["max_price"]
        n = asin_row["n"] or 0
        if family_id is not None:
            f_min = c.execute(_FAMILY_MIN_SQL, (family_id,)).fetchone()["v"]
            f_max = c.execute(_FAMILY_MAX_SQL, (family_id,)).fetchone()["v"]
            n += c.execute(_FAMILY_COUNT_CAPPED_SQL, (family_id, _OBSERVATION_COUNT_CAP)).fetchone()["n"]
            if f_min is not None and (min_price is None or f_min < min_price):
                min_price = f_min
            if f_max is not None and (max_price is None or f_max > max_price):
                max_price = f_max
        return {"min_price": min_price, "max_price": max_price, "n": n}

    if conn is not None:
        return _query(conn)
    with get_connection() as owned:
        return _query(owned)


def record_shadow_comparison(stat_date: str, diverged: bool) -> None:
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO priority_stats (stat_date, shadow_total, shadow_divergences) VALUES (?, 1, ?)
            ON CONFLICT(stat_date) DO UPDATE SET
                shadow_total = shadow_total + 1,
                shadow_divergences = shadow_divergences + excluded.shadow_divergences
            """,
            (stat_date, 1 if diverged else 0),
        )


def get_shadow_stats(stat_date: str) -> dict:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT shadow_total, shadow_divergences FROM priority_stats WHERE stat_date = ?",
            (stat_date,),
        ).fetchone()
    if row is None:
        return {"shadow_total": 0, "shadow_divergences": 0}
    return {"shadow_total": row["shadow_total"], "shadow_divergences": row["shadow_divergences"]}


def get_learned_brands_and_categories() -> tuple[set[str], set[str]]:
    """Distinct brand/category names with at least one enabled learned rule
    -- parsed from learned_rules.key, whose shape depends on rule_type (see
    listener/learning.py's module docstring for the four key formats).
    """
    brands: set[str] = set()
    categories: set[str] = set()
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT rule_type, key FROM learned_rules WHERE enabled = 1"
        ).fetchall()
    for row in rows:
        rule_type, key = row["rule_type"], row["key"]
        if rule_type == "brand":
            brands.add(key)
        elif rule_type == "brand_category":
            brand, _, category = key.partition("|")
            brands.add(brand)
            categories.add(category)
        elif rule_type in ("category_price", "category_discount"):
            category, _, _bucket = key.partition("|")
            categories.add(category)
    return brands, categories
