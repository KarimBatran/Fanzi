from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TrackedProduct:
    id: int
    user_id: int
    asin: str
    title: str | None
    url: str
    current_price: float | None
    target_price: float
    currency: str
    last_checked: str | None
    last_notified_price: float | None
    active: bool
    available: bool
    created_at: str
