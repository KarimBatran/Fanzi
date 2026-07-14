"""Lightweight per-channel posting-frequency watchdog. Flags a channel as
silent only relative to its *own* recent historical average posting
interval, never a fixed absolute threshold — a channel that normally posts
once a day isn't "unhealthy" after 12 hours of silence, but one that
normally posts every 10 minutes and has gone quiet for 2 hours is. A
channel with no posting history at all (never posted since monitoring
began) is flagged too, since there's no baseline to say it's "normal".
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta

import database

logger = logging.getLogger("fanzi.listener.watchdog")

_HISTORY_DAYS = 7
# Flagged only once silence stretches to this many multiples of the
# channel's own historical average posting interval.
_SILENCE_MULTIPLIER = 5.0
# Floor so a channel with very few historical posts (a noisy average) isn't
# flagged after a short, unremarkable gap.
_MIN_SILENCE_MINUTES_BEFORE_FLAGGING = 30.0


@dataclass
class ChannelHealth:
    channel: str
    last_post_at: datetime | None
    minutes_since_last_post: float | None
    avg_interval_minutes: float | None
    is_silent_anomaly: bool
    warning: str | None


def _history_date_range() -> tuple[str, str]:
    end = date.today()
    start = end - timedelta(days=_HISTORY_DAYS - 1)
    return start.isoformat(), end.isoformat()


def check_channel(channel: str) -> ChannelHealth:
    last_post_row = database.get_channel_last_post(channel)
    last_post_at = datetime.fromisoformat(last_post_row["last_post_at"]) if last_post_row else None
    minutes_since = (datetime.now() - last_post_at).total_seconds() / 60 if last_post_at is not None else None

    start_date, end_date = _history_date_range()
    total_posts = database.get_channel_stats_range(channel, start_date, end_date)["posts_received"]
    avg_interval_minutes = (_HISTORY_DAYS * 24 * 60) / total_posts if total_posts > 0 else None

    is_anomaly = False
    warning = None
    if last_post_at is None:
        is_anomaly = True
        warning = "No posts received yet"
    elif avg_interval_minutes is not None and minutes_since is not None:
        threshold = max(avg_interval_minutes * _SILENCE_MULTIPLIER, _MIN_SILENCE_MINUTES_BEFORE_FLAGGING)
        if minutes_since > threshold:
            is_anomaly = True
            warning = (
                f"No posts received for {minutes_since:.0f} min "
                f"(expected roughly every {avg_interval_minutes:.0f} min)"
            )

    return ChannelHealth(
        channel=channel,
        last_post_at=last_post_at,
        minutes_since_last_post=minutes_since,
        avg_interval_minutes=avg_interval_minutes,
        is_silent_anomaly=is_anomaly,
        warning=warning,
    )


def check_all_channels(channels: list[str]) -> list[ChannelHealth]:
    """Checks every channel and logs a WARNING for each anomaly found —
    called both proactively (scheduler.py, periodic) and reactively
    (health.py, on every /status)."""
    results = [check_channel(c) for c in channels]
    for result in results:
        if result.is_silent_anomaly:
            logger.warning("channel watchdog: %s — %s", result.channel, result.warning)
    return results
