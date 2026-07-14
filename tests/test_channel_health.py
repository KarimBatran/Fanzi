"""Covers the per-channel health audit: persisted counters (database.py),
the silence watchdog (listener/watchdog.py), the /status channel summary
(health.py), and that listener/watcher.py's _handle_post actually increments
the right counters for real (mocked-AI) post outcomes. Zero real Gemini/Groq
calls anywhere.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import database
from listener import watchdog
from listener.analyzer import DealVerdict

TODAY = date.today().isoformat()


def test_channel_stats_accumulate_and_persist():
    database.record_channel_post_received("ba3bou3_deals", TODAY)
    database.record_channel_post_received("ba3bou3_deals", TODAY)
    database.record_channel_parsed("ba3bou3_deals", TODAY)
    database.record_channel_forwarded("ba3bou3_deals", TODAY)
    database.record_channel_duplicate("ba3bou3_deals", TODAY)

    stats = database.get_channel_stats("ba3bou3_deals", TODAY)
    assert stats["posts_received"] == 2
    assert stats["parsed"] == 1
    assert stats["forwarded"] == 1
    assert stats["duplicates"] == 1

    # Simulates a restart: a fresh read of the same (test) DB sees the same values.
    stats_again = database.get_channel_stats("ba3bou3_deals", TODAY)
    assert stats_again == stats


def test_channel_stats_failure_reasons_are_distinct():
    database.record_channel_no_price("CouponsEgypt", TODAY)
    database.record_channel_no_asin("CouponsEgypt", TODAY)
    database.record_channel_no_asin("CouponsEgypt", TODAY)
    database.record_channel_non_amazon_link("CouponsEgypt", TODAY)

    stats = database.get_channel_stats("CouponsEgypt", TODAY)
    assert stats["no_price_failures"] == 1
    assert stats["no_asin_failures"] == 2
    assert stats["non_amazon_links"] == 1


def test_channel_last_post_tracked():
    now_iso = datetime.now().isoformat()
    database.record_channel_last_post("Mego_Reviews", now_iso, 12345)
    row = database.get_channel_last_post("Mego_Reviews")
    assert row["last_post_at"] == now_iso
    assert row["last_post_id"] == 12345


def test_watchdog_flags_channel_with_zero_history():
    result = watchdog.check_channel("NeverPostedChannel")
    assert result.is_silent_anomaly is True
    assert "No posts received" in result.warning


def test_watchdog_flags_channel_silent_relative_to_its_own_history(monkeypatch):
    # Historical average: 2000 posts over the last 7 days -> ~1 every ~5 min,
    # so a 2-hour silence is a real anomaly (well past 5x that interval).
    for i in range(7):
        d = (date.today() - timedelta(days=i)).isoformat()
        database._bump_channel_stat("BusyChannel", d, "posts_received", 2000 / 7)

    stale_time = (datetime.now() - timedelta(hours=2)).isoformat()
    database.record_channel_last_post("BusyChannel", stale_time, 1)

    result = watchdog.check_channel("BusyChannel")
    assert result.is_silent_anomaly is True
    assert "No posts received for" in result.warning


def test_watchdog_does_not_flag_a_genuinely_quiet_channel():
    # Historical average: only a handful of posts over 7 days -> a long
    # natural gap between posts, so a recent-ish post must NOT be flagged.
    for i in range(3):
        d = (date.today() - timedelta(days=i * 2)).isoformat()
        database.record_channel_post_received("QuietChannel", d)

    recent_time = (datetime.now() - timedelta(minutes=20)).isoformat()
    database.record_channel_last_post("QuietChannel", recent_time, 1)

    result = watchdog.check_channel("QuietChannel")
    assert result.is_silent_anomaly is False


def test_get_channel_health_includes_every_configured_channel():
    import health

    channels = ["ba3bou3_deals", "CouponsEgypt", "Mego_Reviews", "NeverSeenChannel"]
    with patch("health.channels_store.get_effective_channels", return_value=channels):
        results = health.get_channel_health()

    assert [r["channel"] for r in results] == channels
    # A channel with zero activity still appears, correctly flagged.
    never_seen = next(r for r in results if r["channel"] == "NeverSeenChannel")
    assert never_seen["is_silent_anomaly"] is True


def test_status_message_includes_channel_section():
    import health

    with patch("health.channels_store.get_effective_channels", return_value=["ba3bou3_deals"]):
        message = health.format_status_message()

    assert "📡 Channels" in message
    assert "ba3bou3_deals" in message


POST_TEXT = "Test deal price: 500 EGP\nhttps://www.amazon.eg/dp/B0HEALTHTEST"


def _fake_bot():
    bot = AsyncMock()
    msg = MagicMock()
    msg.chat_id, msg.message_id = 1, 2
    bot.send_message = AsyncMock(return_value=msg)
    bot.edit_message_text = AsyncMock()
    return bot


@pytest.mark.asyncio
async def test_handle_post_increments_posts_parsed_and_forwarded():
    import listener.watcher as watcher_module

    bot = _fake_bot()
    verdict = DealVerdict(
        deal_quality="great", reason="x", suggested_target=450, category="other", provider="gemini"
    )
    with patch.object(watcher_module, "analyze_deal", new=AsyncMock(return_value=verdict)):
        await watcher_module._handle_post(bot, POST_TEXT, "counter_test_channel", message_id=1)

    stats = database.get_channel_stats("counter_test_channel", TODAY)
    assert stats["posts_received"] == 1
    assert stats["parsed"] == 1
    assert stats["forwarded"] == 1
    assert stats["ai_analyses"] == 1


@pytest.mark.asyncio
async def test_handle_post_increments_no_price_failure_counter():
    import listener.watcher as watcher_module

    bot = _fake_bot()
    # A post with an ASIN link but no price anywhere in the text. (Note: the
    # ASIN deliberately avoids containing the substring "price" -- the parser's
    # price regex matches the literal word "price" case-insensitively, so an
    # ASIN like B0NOPRICE1 would false-positive-match its own "PRICE" substring.)
    text = "Check this out\nhttps://www.amazon.eg/dp/B0TESTXYZ9"
    await watcher_module._handle_post(bot, text, "no_price_channel", message_id=1)

    stats = database.get_channel_stats("no_price_channel", TODAY)
    assert stats["posts_received"] == 1
    assert stats["no_price_failures"] == 1
    assert stats["parsed"] == 0
    bot.send_message.assert_not_called()


@pytest.mark.asyncio
async def test_handle_post_increments_no_asin_failure_counter():
    import listener.watcher as watcher_module

    bot = _fake_bot()
    text = "Just some random channel chatter, price 500 EGP, no product link at all."
    await watcher_module._handle_post(bot, text, "no_asin_channel", message_id=1)

    stats = database.get_channel_stats("no_asin_channel", TODAY)
    assert stats["posts_received"] == 1
    assert stats["no_asin_failures"] == 1


@pytest.mark.asyncio
async def test_handle_post_increments_duplicate_counter():
    import listener.watcher as watcher_module

    bot = _fake_bot()
    verdict = DealVerdict(
        deal_quality="great", reason="x", suggested_target=450, category="other", provider="gemini"
    )
    with patch.object(watcher_module, "analyze_deal", new=AsyncMock(return_value=verdict)):
        await watcher_module._handle_post(bot, POST_TEXT, "dup_test_channel", message_id=1)
        await watcher_module._handle_post(bot, POST_TEXT, "dup_test_channel", message_id=2)

    stats = database.get_channel_stats("dup_test_channel", TODAY)
    assert stats["posts_received"] == 2
    assert stats["duplicates"] == 1
    assert stats["forwarded"] == 1  # only the first one forwarded
