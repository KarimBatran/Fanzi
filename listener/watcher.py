"""Telethon client watching configured public deal channels. Parses each
post, gets a Gemini verdict, auto-tracks qualifying deals, and forwards a
verdict summary to ADMIN_TELEGRAM_ID via the existing bot.

Two entry points:
- `start_background_listener(bot)` — embedded in bot.py's own event loop.
  Never prompts for interactive login; if no session file exists yet it
  logs instructions and returns None (listener disabled) rather than
  hanging bot.py waiting for console input.
- `main()` / `python listener\\watcher.py` — standalone script. Used for
  the one-time interactive Telethon login (phone + OTP) that creates the
  session file, and can also run the listener stand-alone afterwards.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from urllib.parse import urlsplit, urlunsplit

# Same embeddable-Python sys.path workaround as bot.py (see the comment
# there): needed so `python listener\watcher.py` (standalone) can resolve
# top-level imports (database, config, amazon.*) from the fanzi/ directory,
# two levels up from this file. Harmless no-op on a standard CPython install,
# and a no-op when this module is imported by bot.py (fanzi/ is already on
# sys.path in that case).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from telegram import Bot, InlineKeyboardButton, InlineKeyboardMarkup
from telethon import TelegramClient, events
from telethon.errors import FloodWaitError
from telethon.tl.functions.channels import JoinChannelRequest

import database
import health
from config import (
    ADMIN_TELEGRAM_ID,
    MIN_DEAL_QUALITY,
    TELEGRAM_BOT_TOKEN,
    TELETHON_API_HASH,
    TELETHON_API_ID,
    TELETHON_SESSION_NAME,
)
from listener import channels_store, dedup, pending_deals
from listener.ai_providers import get_manager
from listener.analyzer import analyze_deal, meets_min_quality
from listener.parser import ParsedDeal, extract_from_post

logger = logging.getLogger("fanzi.listener.watcher")

_QUALITY_EMOJI = {"great": "🔥", "good": "✅", "average": "🤷", "skip": "⏭️"}

# Module-level handle on the live client/handler so admin commands (/channels,
# /addchannel, /removechannel) can inspect and update the running listener
# without restarting bot.py. None when the listener isn't running.
_client: TelegramClient | None = None
_bot: Bot | None = None
_message_handler = None
_watched_channels: list[str] = []


def _strip_affiliate_tag(url: str) -> str:
    parts = urlsplit(url)
    query_pairs = [
        pair for pair in parts.query.split("&") if pair and not pair.startswith("tag=")
    ]
    return urlunsplit((parts.scheme, parts.netloc, parts.path, "&".join(query_pairs), ""))


def _format_message(deal: ParsedDeal, verdict_text: str, clean_url: str) -> str:
    discount_badge = f" (-{deal.discount_percent}%)" if deal.discount_percent else ""
    lines = [
        f"🔍 Deal from @{deal.channel_name}",
        "",
        deal.title,
        f"💰 {deal.price:g} EGP{discount_badge}",
        f"📊 Verdict: {verdict_text}",
        "",
        clean_url,
    ]
    return "\n".join(lines)


def _track_button(asin: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [[InlineKeyboardButton("📉 Track Price", callback_data=f"track:{asin}")]]
    )


async def _handle_post(bot: Bot, text: str, channel_name: str) -> None:
    logger.info("[%s] Post received: %s", channel_name, text[:50].replace("\n", " "))

    deal = await extract_from_post(text, channel_name)
    if deal is None:
        logger.info("[%s] No ASIN found — skipping", channel_name)
        return
    logger.info("[%s] ASIN extracted: %s", channel_name, deal.asin)

    dedup_outcome = dedup.check(channel_name, deal.asin, deal.title, deal.price, deal.discount_percent)
    if dedup_outcome == dedup.DUPLICATE:
        logger.info("[%s] duplicate deal skipped", channel_name)
        health.record_duplicate_skipped()
        return
    if dedup_outcome == dedup.PRICE_CHANGED:
        logger.info("[%s] price changed — reprocessing", channel_name)
    elif dedup_outcome == dedup.WINDOW_EXPIRED:
        logger.info("[%s] duplicate window expired — reprocessing", channel_name)
    dedup.mark_seen(channel_name, deal.asin, deal.title, deal.price, deal.discount_percent)

    clean_url = _strip_affiliate_tag(deal.url) if deal.url else f"https://www.amazon.eg/dp/{deal.asin}"
    pending_deals.store(deal.asin, deal.title, clean_url, deal.price, deal.discount_percent)

    price_history = database.get_latest_price_for_asin(deal.asin)
    verdict = await analyze_deal(deal, price_history)

    if verdict is None:
        logger.info("[%s] AI providers unavailable — forwarding without verdict", channel_name)
        # Both providers failed/skipped for one of several reasons — forward
        # the raw deal without a verdict rather than dropping it, but
        # distinguish "both providers' daily quota is used up" (expected, not
        # an app bug) from an actual API/parser failure so the admin isn't
        # misled into thinking something is broken every time quota runs out.
        if get_manager().both_quota_exhausted():
            verdict_text = "unavailable (daily AI quota reached on both providers)"
        else:
            verdict_text = "unavailable (analysis failed)"
        message = _format_message(deal, verdict_text, clean_url)
        await bot.send_message(chat_id=ADMIN_TELEGRAM_ID, text=message, reply_markup=_track_button(deal.asin))
        logger.info("[%s] Forwarding to admin", channel_name)
        return

    logger.info(
        "[%s] AI verdict (%s): %s — %s", channel_name, verdict.provider, verdict.deal_quality, verdict.reason
    )
    health.record_deal_analyzed()

    if not meets_min_quality(verdict.deal_quality, MIN_DEAL_QUALITY):
        logger.info("[%s] Filtered out (%s) — skipping", channel_name, verdict.deal_quality)
        return

    emoji = _QUALITY_EMOJI.get(verdict.deal_quality, "🤷")
    verdict_text = f"{emoji} {verdict.reason}"
    message = _format_message(deal, verdict_text, clean_url)
    logger.info("[%s] Forwarding to admin", channel_name)
    await bot.send_message(chat_id=ADMIN_TELEGRAM_ID, text=message, reply_markup=_track_button(deal.asin))


def _make_message_handler(bot: Bot):
    async def _on_message(event: events.NewMessage.Event) -> None:
        channel = await event.get_chat()
        channel_name = getattr(channel, "username", None) or getattr(channel, "title", "unknown")
        text = event.message.message or ""
        try:
            await _handle_post(bot, text, channel_name)
        except FloodWaitError as exc:
            logger.warning("FloodWaitError — sleeping %ds", exc.seconds)
            await asyncio.sleep(exc.seconds)
        except Exception:
            logger.exception("error handling post from @%s", channel_name)

    return _on_message


def _session_file_path() -> str:
    return f"{TELETHON_SESSION_NAME}.session"


async def start_background_listener(bot: Bot) -> TelegramClient | None:
    """Starts the listener as a connection on the caller's existing asyncio
    event loop. Returns the connected TelegramClient (caller owns its
    lifetime — disconnect it on shutdown), or None if the listener is
    disabled/unavailable. Never raises — every failure is logged and
    treated as "listener unavailable", so it can never take the bot down.
    """
    global _client, _bot, _message_handler, _watched_channels

    channels = channels_store.get_effective_channels()
    if not TELETHON_API_ID or not channels:
        logger.info("deal listener disabled (TELETHON_API_ID/DEAL_CHANNELS not configured)")
        return None

    if not os.path.isfile(_session_file_path()):
        logger.warning(
            "No Telethon session found for the deal listener. Run "
            r"'python listener\watcher.py' once to complete Telegram login, "
            "then restart bot.py. Deal listener is disabled for this run."
        )
        return None

    # catch_up=True: on connect, replay any updates missed while disconnected
    # (and any pts-gap-recovered updates on busy channels) as real events
    # instead of silently absorbing them into internal state with no
    # dispatch to our handler. This SDK version takes it on the client
    # constructor, not on .start().
    client = TelegramClient(TELETHON_SESSION_NAME, TELETHON_API_ID, TELETHON_API_HASH, catch_up=True)
    handler = _make_message_handler(bot)
    client.add_event_handler(handler, events.NewMessage(chats=channels))

    try:
        await client.start()
    except Exception:
        logger.exception("deal listener failed to connect — continuing without it")
        return None

    _client = client
    _bot = bot
    _message_handler = handler
    _watched_channels = channels

    statuses = await _check_channel_statuses(client, channels)
    active_count = sum(1 for _, ok in statuses if ok)
    health.set_channels_status(active=active_count, configured=len(channels))

    logger.info("deal listener running in background, watching: %s", ", ".join(channels))
    return client


async def _check_channel_statuses(client: TelegramClient, channels: list[str]) -> list[tuple[str, bool]]:
    """Live-resolves each channel and returns [(channel, ok), ...], logging as it goes."""
    results: list[tuple[str, bool]] = []
    for channel in channels:
        try:
            await client.get_entity(channel)
            logger.info("deal listener joined channel: %s", channel)
            results.append((channel, True))
        except Exception:
            logger.warning("deal listener could not resolve channel: %s", channel)
            results.append((channel, False))
    return results


async def get_channel_statuses() -> list[tuple[str, bool]]:
    """Live status for /channels. Empty list if the listener isn't running."""
    if _client is None:
        return []
    return await _check_channel_statuses(_client, _watched_channels)


async def _resubscribe() -> None:
    """Re-registers the message handler with the current effective channel
    list — how add/remove take effect immediately on the running client.
    """
    global _message_handler, _watched_channels
    assert _client is not None and _bot is not None
    if _message_handler is not None:
        _client.remove_event_handler(_message_handler)
    channels = channels_store.get_effective_channels()
    handler = _make_message_handler(_bot)
    _client.add_event_handler(handler, events.NewMessage(chats=channels))
    _message_handler = handler
    _watched_channels = channels

    statuses = await _check_channel_statuses(_client, channels)
    active_count = sum(1 for _, ok in statuses if ok)
    health.set_channels_status(active=active_count, configured=len(channels))


async def add_channel_runtime(channel: str) -> tuple[bool, str]:
    """Validates, joins, persists, and subscribes to a new channel on the
    live listener — no restart needed. Returns (success, reply_text).
    """
    if _client is None:
        return False, "Listener isn't running — check the logs."

    channel = channel.lstrip("@").strip()
    try:
        entity = await _client.get_entity(channel)
    except Exception:
        logger.warning("addchannel: could not resolve %s", channel)
        return False, "❌ Couldn't find that channel — check the username and try again"

    try:
        await _client(JoinChannelRequest(entity))
    except Exception:
        # Already joined, or the channel restricts joining — either way we
        # can still receive updates if the entity resolved; don't fail here.
        logger.warning("addchannel: JoinChannelRequest failed for %s — continuing anyway", channel)

    channels_store.add_channel(channel)
    await _resubscribe()
    logger.info("addchannel: now watching %s", channel)
    return True, f"✅ Now watching @{channel}"


async def remove_channel_runtime(channel: str) -> tuple[bool, str]:
    if _client is None:
        return False, "Listener isn't running — check the logs."

    channel = channel.lstrip("@").strip()
    channels_store.remove_channel(channel)
    await _resubscribe()
    logger.info("removechannel: stopped watching %s", channel)
    return True, f"✅ Stopped watching @{channel}"


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    channels = channels_store.get_effective_channels()
    if not channels:
        logger.warning("no deal channels configured — listener has nothing to watch")

    database.init_db()
    bot = Bot(token=TELEGRAM_BOT_TOKEN)
    client = TelegramClient(TELETHON_SESSION_NAME, TELETHON_API_ID, TELETHON_API_HASH, catch_up=True)
    client.add_event_handler(_make_message_handler(bot), events.NewMessage(chats=channels))

    # Standalone run — this is the one place allowed to prompt interactively
    # for phone/OTP when no session file exists yet. catch_up=True: see the
    # comment in start_background_listener() above.
    await client.start()
    logger.info("Fanzi deal listener started, watching: %s", ", ".join(channels))
    await client.run_until_disconnected()


if __name__ == "__main__":
    asyncio.run(main())
