"""In-memory duplicate-deal detection. The same product (by ASIN, or a
normalized title fingerprint when no ASIN is available) posted again by the
same channel within DUPLICATE_WINDOW_HOURS is skipped before it ever reaches
Gemini or the forwarding step. The same product posted by a *different*
channel is unaffected — dedup keys are scoped per-channel.
"""

from __future__ import annotations

import re
import time

from config import DUPLICATE_WINDOW_HOURS

_seen: dict[tuple[str, str], float] = {}


def _fingerprint(asin: str | None, title: str) -> str:
    if asin:
        return f"asin:{asin}"
    normalized_title = re.sub(r"\s+", " ", title.strip().lower())
    return f"title:{normalized_title}"


def is_duplicate(channel_name: str, asin: str | None, title: str) -> bool:
    key = (channel_name, _fingerprint(asin, title))
    last_seen = _seen.get(key)
    if last_seen is None:
        return False
    return (time.time() - last_seen) < DUPLICATE_WINDOW_HOURS * 3600


def mark_seen(channel_name: str, asin: str | None, title: str) -> None:
    _seen[(channel_name, _fingerprint(asin, title))] = time.time()
