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
import time
from datetime import date, datetime

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
    AI_SOFT_TIMEOUT_ENABLED,
    AI_SOFT_TIMEOUT_SECONDS,
    MIN_DEAL_QUALITY,
    TELEGRAM_BOT_TOKEN,
    TELETHON_API_HASH,
    TELETHON_API_ID,
    TELETHON_SESSION_NAME,
)
from listener import channels_store, dedup, family, learning, pending_deals, replay
from listener.ai_providers import get_manager
from listener.analyzer import DealVerdict, analyze_deal, meets_min_quality
from listener.family import FamilyDecision
from listener.parser import ParseDiagnostics, ParsedDeal, extract_from_post
from listener.timing import DealTiming

logger = logging.getLogger("fanzi.listener.watcher")

_QUALITY_EMOJI = {"great": "🔥", "good": "✅", "average": "🤷", "skip": "⏭️"}

# Module-level handle on the live client/handler so admin commands (/channels,
# /addchannel, /removechannel) can inspect and update the running listener
# without restarting bot.py. None when the listener isn't running.
_client: TelegramClient | None = None
_bot: Bot | None = None
_message_handler = None
_reconnect_task: asyncio.Task | None = None
_watched_channels: list[str] = []


def _format_message(deal: ParsedDeal, verdict_text: str, clean_url: str, *, verdict_is_explanation: bool = False) -> str:
    discount_badge = f" (-{deal.discount_percent}%)" if deal.discount_percent else ""
    verdict_line = verdict_text if verdict_is_explanation else f"📊 Verdict: {verdict_text}"
    lines = [
        f"🔍 Deal from @{deal.channel_name}",
        "",
        deal.title,
        f"💰 {deal.price:g} EGP{discount_badge}",
        verdict_line,
        "",
        clean_url,
    ]
    return "\n".join(lines)


def _track_button(asin: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [[InlineKeyboardButton("📉 Track Price", callback_data=f"track:{asin}")]]
    )


def _unavailable_verdict_text() -> str:
    # Both providers failed/skipped for one of several reasons — forward the
    # raw deal without a verdict rather than dropping it, but distinguish
    # "both providers' daily quota is used up" (expected, not an app bug)
    # from an actual API/parser failure so the admin isn't misled into
    # thinking something is broken every time quota runs out.
    if get_manager().both_quota_exhausted():
        return "unavailable (daily AI quota reached on both providers)"
    return "unavailable (analysis failed)"


def _build_message_for_verdict(deal: ParsedDeal, verdict: DealVerdict | None, clean_url: str) -> str:
    if verdict is None:
        return _format_message(deal, _unavailable_verdict_text(), clean_url)
    if verdict.provider == "learned":
        # Honest about the source — never presented as if AI just analyzed
        # this deal (see listener/learning.py's format_explanation()).
        return _format_message(deal, verdict.reason, clean_url, verdict_is_explanation=True)
    emoji = _QUALITY_EMOJI.get(verdict.deal_quality, "🤷")
    return _format_message(deal, f"{emoji} {verdict.reason}", clean_url)


_BETTER_VARIANT_TEMPLATE = (
    "🔥 Better Variant Found\n\n"
    "{title}\n\n"
    "Variant:\n{variant_label}\n\n"
    "Current:\n{price:g} EGP\n\n"
    "Previous family best:\n{previous_price:g} EGP ({previous_label})\n\n"
    "Savings:\n{savings:g} EGP\n\n"
    "This is now the cheapest variant in the family.\n\n"
    "{url}"
)

_NEW_VARIANT_TEMPLATE = (
    "🎨 New Variant Available\n\n"
    "{title}\n\n"
    "Variant:\n{variant_label}\n\n"
    "Price:\n{price:g} EGP\n\n"
    "Current family best:\n{best_label} ({best_price:g} EGP)\n\n"
    "{url}"
)


def _build_message_for_family_variant(deal: ParsedDeal, clean_url: str, decision: FamilyDecision) -> str:
    """Only ever called for the "better_variant"/"new_variant" outcomes of
    listener.family.finalize() -- a brand-new family still uses the normal
    _build_message_for_verdict() path, unaffected by Product Family logic.
    """
    label = family.variant_label(decision.variant)
    if decision.notify_kind == "better_variant":
        previous_price = decision.previous_best_price if decision.previous_best_price is not None else deal.price
        return _BETTER_VARIANT_TEMPLATE.format(
            title=deal.title,
            variant_label=label,
            price=deal.price,
            previous_price=previous_price,
            previous_label=decision.previous_best_label or "previous variant",
            savings=decision.savings if decision.savings is not None else 0.0,
            url=clean_url,
        )
    best_price = decision.previous_best_price if decision.previous_best_price is not None else deal.price
    return _NEW_VARIANT_TEMPLATE.format(
        title=deal.title,
        variant_label=label,
        price=deal.price,
        best_label=decision.previous_best_label or label,
        best_price=best_price,
        url=clean_url,
    )


async def _handle_post(
    bot: Bot, text: str, channel_name: str, message_id: int | None = None, channel_id: int | None = None,
) -> None:
    logger.info("[%s] Post received: %s", channel_name, text[:50].replace("\n", " "))
    stat_date = date.today().isoformat()
    now = datetime.now()
    database.record_channel_post_received(channel_name, stat_date)
    database.record_channel_last_post(channel_name, now.isoformat(), message_id)

    timing = DealTiming(channel=channel_name)

    try:
        await _process_post(bot, text, channel_name, stat_date, timing)
    except Exception:
        database.record_channel_failure(channel_name, stat_date)
        raise
    finally:
        database.record_channel_latency(channel_name, stat_date, timing.total_ms())

    # Only advances the replay checkpoint once processing has completed
    # without raising — an unexpected exception above must not advance it,
    # so the next replay retries this same message.
    if message_id is not None:
        database.set_channel_replay_state(channel_name, channel_id, message_id, now.isoformat())


async def _process_post(bot: Bot, text: str, channel_name: str, stat_date: str, timing: DealTiming) -> None:
    diagnostics = ParseDiagnostics()
    parse_start = time.perf_counter()
    deal = await extract_from_post(text, channel_name, diagnostics=diagnostics)
    timing.record("parser", (time.perf_counter() - parse_start) * 1000)
    if deal is None:
        if diagnostics.reason == "no_price":
            database.record_channel_no_price(channel_name, stat_date)
        elif diagnostics.reason == "non_amazon_link":
            database.record_channel_non_amazon_link(channel_name, stat_date)
        else:
            database.record_channel_no_asin(channel_name, stat_date)
        logger.info("[%s] No ASIN found — skipping", channel_name)
        return
    database.record_channel_parsed(channel_name, stat_date)
    timing.asin = deal.asin
    timing.title = deal.title
    if deal.redirect_ms:
        timing.record("redirect", deal.redirect_ms)
    logger.info("[%s] ASIN extracted: %s", channel_name, deal.asin)

    dedup_start = time.perf_counter()
    dedup_outcome = dedup.check(deal.asin, deal.title, deal.price, deal.discount_percent, channel_name=channel_name)
    if dedup_outcome == dedup.DUPLICATE:
        timing.record("dedup", (time.perf_counter() - dedup_start) * 1000)
        logger.info("[%s] duplicate deal skipped (already forwarded from another channel or earlier)", channel_name)
        health.record_duplicate_skipped()
        database.record_channel_duplicate(channel_name, stat_date)
        return
    if dedup_outcome == dedup.PRICE_CHANGED:
        logger.info("[%s] price changed — reprocessing", channel_name)
    elif dedup_outcome == dedup.WINDOW_EXPIRED:
        logger.info("[%s] duplicate window expired — reprocessing", channel_name)
    dedup.mark_seen(deal.asin, deal.title, deal.price, deal.discount_percent, channel_name=channel_name)
    timing.record("dedup", (time.perf_counter() - dedup_start) * 1000)

    # Product Family detection: recognizes color/size/capacity/pack variants
    # of the same underlying product as related, not automatically
    # duplicates (see listener/family.py). Runs *before* the deal-quality AI
    # call so a true same-variant repost never costs an AI request.
    brand = learning.extract_brand(deal.title)
    guessed_category = learning.guess_category(deal.title)
    family_start = time.perf_counter()
    family_decision = await family.pre_check(
        deal.asin, deal.title, deal.price, deal.discount_percent, brand, guessed_category
    )
    timing.record("family", (time.perf_counter() - family_start) * 1000)

    if family_decision.notify_kind == "duplicate":
        logger.info(
            "[%s] true family-variant duplicate skipped (%s, family=%s)",
            channel_name, deal.asin, family_decision.family_id,
        )
        health.record_duplicate_skipped()
        database.record_channel_duplicate(channel_name, stat_date)
        return

    # Always the canonical Amazon product URL -- never the original
    # channel's short/tracking link (see amazon.parser.normalize_product_url
    # and listener.parser.extract_from_post, which now always sets deal.url
    # this way once the ASIN has been resolved).
    clean_url = deal.url
    pending_deals.store(deal.asin, deal.title, clean_url, deal.price, deal.discount_percent)

    # Append-only price history for listener/scoring.py -- the
    # deal-forwarding half of price_observations (the tracked-product half
    # is written by database.update_price_check). Recorded here, after the
    # family pre_check, so the observation carries its family_id.
    database.record_price_observation(
        deal.asin, family_decision.family_id, deal.price, deal.discount_percent,
        datetime.now().isoformat(),
    )

    price_history = database.get_latest_price_for_asin(deal.asin)

    # Daily AI budget manager (listener/budget.py) priority signals: a
    # brand-new family is always Priority 1, and "cheaper than every
    # previous variant" is known immediately from pre_check's own price
    # comparison, without waiting for a verdict.
    is_new_family = family_decision.notify_kind == "new_family"
    is_new_family_low_price = is_new_family or (
        family_decision.previous_best_price is None or deal.price < family_decision.previous_best_price
    )
    analyze_kwargs = dict(
        family_id=family_decision.family_id, variant=family_decision.variant,
        is_new_family=is_new_family, is_new_family_low_price=is_new_family_low_price,
    )

    if not AI_SOFT_TIMEOUT_ENABLED:
        verdict = await analyze_deal(deal, price_history, timing=timing, **analyze_kwargs)
        await _forward_verdict(bot, deal, verdict, clean_url, channel_name, timing, stat_date, family_decision)
        return

    analysis_task = asyncio.create_task(analyze_deal(deal, price_history, timing=timing, **analyze_kwargs))
    done, _pending = await asyncio.wait({analysis_task}, timeout=AI_SOFT_TIMEOUT_SECONDS)

    if analysis_task in done:
        verdict = analysis_task.result()
        await _forward_verdict(bot, deal, verdict, clean_url, channel_name, timing, stat_date, family_decision)
        return

    # AI hasn't answered within the soft timeout — user experience over
    # waiting indefinitely: forward now with a placeholder, keep analyzing in
    # the background, and edit this same message in place once it finishes.
    logger.info(
        "[%s] AI soft timeout (%.1fs) — forwarding placeholder, analysis continues in background",
        channel_name, AI_SOFT_TIMEOUT_SECONDS,
    )
    placeholder_text = _format_message(deal, "analyzing...", clean_url)
    send_start = time.perf_counter()
    message = await bot.send_message(
        chat_id=ADMIN_TELEGRAM_ID, text=placeholder_text, reply_markup=_track_button(deal.asin)
    )
    timing.record("telegram_send", (time.perf_counter() - send_start) * 1000)
    timing.note(f"AI timed out after {AI_SOFT_TIMEOUT_SECONDS * 1000:.0f} ms — forwarded with placeholder")
    timing.log_summary()

    asyncio.create_task(
        _finish_analysis_in_background(
            bot, analysis_task, deal, clean_url, channel_name, stat_date, message.chat_id, message.message_id,
            family_decision,
        )
    )


def _record_verdict_stats(verdict: DealVerdict, channel_name: str, stat_date: str) -> None:
    if verdict.provider == "learned":
        database.record_channel_rule_hit(channel_name, stat_date)
    elif verdict.provider in ("gemini", "groq"):
        database.record_channel_ai_analysis(channel_name, stat_date)


async def _forward_verdict(
    bot: Bot, deal: ParsedDeal, verdict: DealVerdict | None, clean_url: str, channel_name: str,
    timing: DealTiming, stat_date: str, family_decision: FamilyDecision,
) -> None:
    """Shared by both the normal (AI answered in time) and soft-timeout
    (AI answered late, called from the immediate path when it happens to
    finish inside the timeout window) code paths.
    """
    if verdict is None:
        logger.info("[%s] AI providers unavailable — forwarding without verdict", channel_name)
    else:
        logger.info(
            "[%s] AI verdict (%s): %s — %s", channel_name, verdict.provider, verdict.deal_quality, verdict.reason
        )
        health.record_deal_analyzed()
        _record_verdict_stats(verdict, channel_name, stat_date)

    # A brand-new family behaves exactly as before this feature existed --
    # standard message, standard MIN_DEAL_QUALITY filter. An existing
    # family's variant is finalized only now (a verdict, or None, is known)
    # and is never subject to the quality filter: "New Variant Available"/
    # "Better Variant Found" are informational about family state, not a
    # deal-quality judgment, and price/discount alone can already justify
    # notifying regardless of what the AI verdict says (see family.py).
    final_decision = family_decision
    if family_decision.notify_kind == "pending":
        final_decision = family.finalize(
            family_decision.family_id, deal.asin, family_decision.variant,
            deal.price, deal.discount_percent, verdict.deal_quality if verdict is not None else None,
        )

    if final_decision.notify_kind == "new_family" and verdict is not None:
        if not meets_min_quality(verdict.deal_quality, MIN_DEAL_QUALITY):
            logger.info("[%s] Filtered out (%s) — skipping", channel_name, verdict.deal_quality)
            timing.log_summary()
            return

    if final_decision.notify_kind in ("better_variant", "new_variant"):
        message_text = _build_message_for_family_variant(deal, clean_url, final_decision)
    else:
        message_text = _build_message_for_verdict(deal, verdict, clean_url)
    send_start = time.perf_counter()
    await bot.send_message(chat_id=ADMIN_TELEGRAM_ID, text=message_text, reply_markup=_track_button(deal.asin))
    timing.record("telegram_send", (time.perf_counter() - send_start) * 1000)
    database.record_channel_forwarded(channel_name, stat_date)
    if timing.ai_total_ms():
        database.record_channel_ai_latency(channel_name, stat_date, timing.ai_total_ms())
    logger.info("[%s] Forwarding to admin", channel_name)
    timing.log_summary()


async def _finish_analysis_in_background(
    bot: Bot, analysis_task: asyncio.Task, deal: ParsedDeal, clean_url: str, channel_name: str, stat_date: str,
    chat_id: int, message_id: int, family_decision: FamilyDecision,
) -> None:
    """Awaits the AI analysis that outran the soft timeout, then edits the
    already-forwarded placeholder message in place with the real verdict.
    If AI ultimately fails, the placeholder is left unchanged — no duplicate
    notification is ever sent for the same deal.
    """
    try:
        verdict = await analysis_task
    except Exception:
        logger.exception("[%s] background analysis failed for %s — leaving message unchanged", channel_name, deal.asin)
        database.record_channel_failure(channel_name, stat_date)
        return

    if verdict is None:
        logger.info(
            "[%s] background analysis unavailable for %s — leaving message unchanged", channel_name, deal.asin
        )
        return

    health.record_deal_analyzed()
    _record_verdict_stats(verdict, channel_name, stat_date)

    final_decision = family_decision
    if family_decision.notify_kind == "pending":
        final_decision = family.finalize(
            family_decision.family_id, deal.asin, family_decision.variant,
            deal.price, deal.discount_percent, verdict.deal_quality,
        )

    # Already forwarded via the placeholder — the deal is visible either way,
    # so a low-quality verdict still gets an honest edit rather than leaving
    # "analyzing..." stuck forever (that's reserved for the AI-failed case).
    if final_decision.notify_kind in ("better_variant", "new_variant"):
        new_text = _build_message_for_family_variant(deal, clean_url, final_decision)
    else:
        new_text = _build_message_for_verdict(deal, verdict, clean_url)
    try:
        await bot.edit_message_text(chat_id=chat_id, message_id=message_id, text=new_text)
        database.record_channel_forwarded(channel_name, stat_date)
        logger.info("[%s] Edited placeholder with delayed verdict (%s) for %s", channel_name, verdict.provider, deal.asin)
    except Exception:
        logger.exception("[%s] failed to edit placeholder message for %s", channel_name, deal.asin)
        database.record_channel_failure(channel_name, stat_date)


def _make_message_handler(bot: Bot):
    async def _on_message(event: events.NewMessage.Event) -> None:
        channel = await event.get_chat()
        channel_name = getattr(channel, "username", None) or getattr(channel, "title", "unknown")
        text = event.message.message or ""
        try:
            await _handle_post(bot, text, channel_name, message_id=event.message.id, channel_id=channel.id)
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

    await replay.replay_all(client, bot, channels, _handle_post, reason="startup")

    global _reconnect_task
    _reconnect_task = asyncio.create_task(replay.watch_for_reconnects(client, bot, channels, _handle_post))

    logger.info("deal listener running in background, watching: %s", ", ".join(channels))
    return client


def stop_reconnect_watcher() -> None:
    global _reconnect_task
    if _reconnect_task is not None:
        _reconnect_task.cancel()
        _reconnect_task = None


async def _check_channel_statuses(client: TelegramClient, channels: list[str]) -> list[tuple[str, bool]]:
    """Live-resolves each channel and returns [(channel, ok), ...], logging as it goes."""
    results: list[tuple[str, bool]] = []
    for channel in channels:
        try:
            entity = await client.get_entity(channel)
            # Logs the resolved Telegram entity ID alongside the configured
            # name — makes "Got difference for channel <id>" telethon log
            # lines attributable to a specific monitored channel without
            # guessing, which matters for auditing per-channel health.
            logger.info("deal listener joined channel: %s (id=%s)", channel, entity.id)
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
