"""In-memory holding area for deals that have been forwarded to the admin
but not yet tracked. Keyed by ASIN. A deal is stored when forwarded and
popped once the user actually creates a tracker for it (or evicted by
simply being overwritten by a newer post of the same ASIN).

Deliberately not persisted: if the bot restarts between a deal being
forwarded and the Track Price button being pressed, the button reports the
deal as expired and the user falls back to /track. That's an acceptable
trade-off for a personal bot — the alternative (a DB table for
short-lived, single-purpose UI state) isn't worth the extra migration.
"""

from __future__ import annotations

_pending: dict[str, dict] = {}


def store(asin: str, title: str, url: str, price: float, discount_percent: int | None) -> None:
    _pending[asin] = {
        "title": title,
        "url": url,
        "price": price,
        "discount_percent": discount_percent,
    }


def get(asin: str) -> dict | None:
    return _pending.get(asin)


def pop(asin: str) -> dict | None:
    return _pending.pop(asin, None)
