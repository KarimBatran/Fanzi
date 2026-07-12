"""Persistent duplicate-deal detection (backed by the shared SQLite database,
so it survives restarts). The same product (by ASIN, or a normalized title
fingerprint when no ASIN is available) posted again by the same channel
within DUPLICATE_WINDOW_HOURS is treated as a duplicate — *unless* the price
has dropped, the discount has increased, or the window has expired, in which
case it's a new deal worth re-analyzing. Scoped per-channel: the same
product from a *different* channel is always processed independently.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta

import database
from config import DUPLICATE_WINDOW_HOURS

# Outcomes returned by check(): "new" (never seen), "duplicate" (skip),
# "price_changed" (price dropped or discount rose — reprocess), or
# "window_expired" (last seen too long ago — reprocess).
NEW = "new"
DUPLICATE = "duplicate"
PRICE_CHANGED = "price_changed"
WINDOW_EXPIRED = "window_expired"


def _fingerprint(asin: str | None, title: str) -> str:
    if asin:
        return f"asin:{asin}"
    normalized_title = re.sub(r"\s+", " ", title.strip().lower())
    return f"title:{normalized_title}"


def check(channel_name: str, asin: str | None, title: str, price: float, discount_percent: int | None) -> str:
    """Looks up prior state without modifying it. Returns one of NEW,
    DUPLICATE, PRICE_CHANGED, WINDOW_EXPIRED. Callers should call
    `mark_seen(...)` afterwards for every outcome except DUPLICATE (a plain
    duplicate must not refresh the window, or a channel reposting the same
    unchanged deal forever would never expire).
    """
    identifier = _fingerprint(asin, title)
    record = database.get_duplicate_record(channel_name, identifier)
    if record is None:
        return NEW

    last_seen_at = datetime.fromisoformat(record["last_seen_at"])
    window = timedelta(hours=DUPLICATE_WINDOW_HOURS)
    if datetime.now() - last_seen_at >= window:
        return WINDOW_EXPIRED

    last_price = record["last_price"]
    last_discount = record["last_discount_percent"]

    price_decreased = last_price is not None and price < last_price
    discount_increased = discount_percent is not None and (
        last_discount is None or discount_percent > last_discount
    )
    if price_decreased or discount_increased:
        return PRICE_CHANGED

    return DUPLICATE


def mark_seen(channel_name: str, asin: str | None, title: str, price: float, discount_percent: int | None) -> None:
    identifier = _fingerprint(asin, title)
    database.upsert_duplicate_record(
        channel_name, identifier, price, discount_percent, datetime.now().isoformat()
    )


def get_active_count() -> int:
    """Number of duplicate_deals rows not yet expired — for /status."""
    cutoff = (datetime.now() - timedelta(hours=DUPLICATE_WINDOW_HOURS)).isoformat()
    return database.count_active_duplicate_records(cutoff)


def cleanup_expired() -> int:
    """Purges expired rows. Returns the count deleted. Call periodically
    (e.g. once per scheduler cycle) to keep the table from growing forever.
    """
    cutoff = (datetime.now() - timedelta(hours=DUPLICATE_WINDOW_HOURS)).isoformat()
    return database.delete_expired_duplicate_records(cutoff)
