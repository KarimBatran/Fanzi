"""Automatic message replay: recovers deals from Telegram messages the live
event listener may have missed (downtime, restarts, temporary disconnects)
by fetching recent channel history and replaying anything newer than the
last successfully processed message ID through the exact same _handle_post
pipeline used for live events. Duplicate detection, learned rules, AI
provider selection, and the Track Price flow all apply identically to a
replayed message as to a live one -- replay is just _handle_post called
with an older message's text and real message ID, so it can never produce
a duplicate notification (dedup already guarantees that idempotency).
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import date, datetime

import database
from config import REPLAY_FETCH_LIMIT, REPLAY_RECONNECT_POLL_SECONDS

logger = logging.getLogger("fanzi.listener.replay")

_state = "idle"  # "idle" | "running"
_last_replay_at: datetime | None = None


def get_status() -> dict:
    return {
        "state": _state,
        "last_replay_at": _last_replay_at,
        "recovered_today": database.get_replay_recovered(date.today().isoformat()),
    }


async def replay_channel(client, bot, channel: str, handle_post, fetch_limit: int) -> int:
    """Fetches recent history for one channel and replays anything newer
    than the stored last-processed message ID, oldest first. Returns the
    count actually recovered (processed without raising).
    """
    state_row = database.get_channel_replay_state(channel)
    last_id = state_row["last_message_id"] if state_row is not None else None

    try:
        messages = await client.get_messages(channel, limit=fetch_limit)
    except Exception:
        logger.exception("replay: failed to fetch history for %s", channel)
        return 0

    if not messages:
        return 0

    to_replay = [m for m in messages if last_id is None or m.id > last_id]
    to_replay.sort(key=lambda m: m.id)  # chronological order, oldest first

    if not to_replay:
        return 0

    recovered = 0
    for message in to_replay:
        text = message.message or ""
        try:
            channel_id = getattr(message.peer_id, "channel_id", None)
            await handle_post(bot, text, channel, message_id=message.id, channel_id=channel_id)
            recovered += 1
        except Exception:
            # Do not advance past this message, and stop replaying this
            # channel for this run -- the next replay (next startup or
            # reconnect) will retry from the same checkpoint.
            logger.exception(
                "replay: failed to process message %s in %s — stopping replay for this channel",
                message.id, channel,
            )
            break

    if recovered:
        database.record_replay_recovered(date.today().isoformat(), recovered)
    return recovered


async def replay_all(
    client, bot, channels: list[str], handle_post, *, reason: str, fetch_limit: int | None = None
) -> dict[str, int]:
    """Runs replay_channel for every monitored channel and logs the
    "Replay complete" summary. Safe to call repeatedly (startup, every
    reconnect) — each channel only ever processes what's genuinely new
    since its own last successfully processed message.
    """
    global _state, _last_replay_at

    if fetch_limit is None:
        fetch_limit = REPLAY_FETCH_LIMIT

    _state = "running"
    logger.info("Replay started (%s)", reason)
    start = time.monotonic()

    results: dict[str, int] = {}
    for channel in channels:
        results[channel] = await replay_channel(client, bot, channel, handle_post, fetch_limit)

    duration = time.monotonic() - start
    _last_replay_at = datetime.now()
    _state = "idle"

    total_recovered = sum(results.values())
    summary_lines = ["Replay complete"]
    for channel, count in results.items():
        summary_lines.append(channel)
        summary_lines.append(f"{count} recovered")
    logger.info("\n".join(summary_lines))
    logger.info(
        "Replay finished (%s): %d total recovered across %d channel(s) in %.1fs",
        reason, total_recovered, len(channels), duration,
    )

    return results


async def watch_for_reconnects(client, bot, channels: list[str], handle_post, poll_interval: float | None = None) -> None:
    """Polls the Telethon client's connection state and triggers a replay
    every time it transitions from disconnected back to connected -- covers
    temporary network outages that live events alone might drop updates
    across, in addition to Telethon's own catch_up mechanism.
    """
    if poll_interval is None:
        poll_interval = REPLAY_RECONNECT_POLL_SECONDS

    was_connected = client.is_connected()
    while True:
        await asyncio.sleep(poll_interval)
        is_connected = client.is_connected()
        if is_connected and not was_connected:
            logger.info("Telethon reconnected — running replay to catch up on any missed updates")
            await replay_all(client, bot, channels, handle_post, reason="reconnect")
        was_connected = is_connected
