"""Persistent duplicate-deal detection (backed by the shared SQLite database,
so it survives restarts). Global across all monitored channels — the same
canonical product posted by two different channels is the same deal, and
must only ever forward once. Identity priority:

1. Canonical ASIN (the Amazon Standard Identification Number is already the
   canonical Amazon product identifier — there is no separate "product ID"
   concept in this app beyond it).
2. Normalized title + normalized price, used only as a fallback when no
   ASIN could be resolved at all.

The channel a post came from and the original message URL never influence
identity or the lookup — only channel_name is recorded (for observability)
alongside the shared record, never as part of the key.

Within DUPLICATE_WINDOW_HOURS, a repost of the same product is a duplicate
*unless* the price has dropped or the discount has increased, in which case
it's treated as a new deal worth re-analyzing; past the window it's treated
as new regardless.
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


def _fingerprint(asin: str | None, title: str, price: float) -> str:
    """Never derived from the original message URL — only the canonical
    ASIN (once the parser has resolved it) or, failing that, the product's
    own normalized title and price.
    """
    if asin:
        return f"asin:{asin.strip().upper()}"
    normalized_title = re.sub(r"\s+", " ", title.strip().lower())
    return f"title:{normalized_title}|price:{price:.2f}"


def check(asin: str | None, title: str, price: float, discount_percent: int | None, channel_name: str = "") -> str:
    """Looks up prior state without modifying it. Returns one of NEW,
    DUPLICATE, PRICE_CHANGED, WINDOW_EXPIRED. Callers should call
    `mark_seen(...)` afterwards for every outcome except DUPLICATE (a plain
    duplicate must not refresh the window, or a channel reposting the same
    unchanged deal forever would never expire). `channel_name` is only used
    for logging/observability (which channel triggered this check) — it is
    never part of the lookup key, so the same product from any channel maps
    to the same record.
    """
    identifier = _fingerprint(asin, title, price)
    record = database.get_global_duplicate_record(identifier)
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


def mark_seen(asin: str | None, title: str, price: float, discount_percent: int | None, channel_name: str = "") -> None:
    identifier = _fingerprint(asin, title, price)
    database.upsert_global_duplicate_record(
        identifier, channel_name, price, discount_percent, datetime.now().isoformat()
    )


def get_active_count() -> int:
    """Number of global_duplicate_deals rows not yet expired — for /status."""
    cutoff = (datetime.now() - timedelta(hours=DUPLICATE_WINDOW_HOURS)).isoformat()
    return database.count_active_global_duplicate_records(cutoff)


def cleanup_expired() -> int:
    """Purges expired rows. Returns the count deleted. Call periodically
    (e.g. once per scheduler cycle) to keep the table from growing forever.
    """
    cutoff = (datetime.now() - timedelta(hours=DUPLICATE_WINDOW_HOURS)).isoformat()
    return database.delete_expired_global_duplicate_records(cutoff)
