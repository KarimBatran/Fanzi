from __future__ import annotations

from dataclasses import dataclass


@dataclass
class User:
    id: int
    telegram_id: int
    username: str | None
    created_at: str
