"""Covers listener/watcher.py's forwarding-latency instrumentation and the
AI soft-timeout/background-edit flow. Uses a real parser (no network needed
for a direct /dp/ link) and a fake Telegram Bot; analyze_deal itself is
mocked so these tests make zero real Gemini/Groq calls and don't depend on
timing-sensitive real AI latency.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from listener.analyzer import DealVerdict

POST_TEXT = "Test deal price: 500 EGP\nhttps://www.amazon.eg/dp/B0TESTDEAL"


def _fake_message(chat_id=111, message_id=222):
    msg = MagicMock()
    msg.chat_id = chat_id
    msg.message_id = message_id
    return msg


def _fake_bot():
    bot = AsyncMock()
    bot.send_message = AsyncMock(return_value=_fake_message())
    bot.edit_message_text = AsyncMock()
    return bot


@pytest.mark.asyncio
async def test_fast_deal_forwards_normally_with_timing_logged(caplog):
    import logging

    import listener.watcher as watcher

    bot = _fake_bot()
    fast_verdict = DealVerdict(
        deal_quality="great", reason="Great deal.", suggested_target=450, category="other", provider="gemini"
    )
    with patch.object(watcher, "analyze_deal", new=AsyncMock(return_value=fast_verdict)), caplog.at_level(
        logging.INFO, logger="fanzi.listener.timing"
    ):
        await watcher._handle_post(bot, POST_TEXT, "test_channel")

    bot.send_message.assert_called_once()
    bot.edit_message_text.assert_not_called()
    assert any("Deal timing" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_ai_soft_timeout_forwards_placeholder_then_edits_on_completion(monkeypatch):
    import listener.watcher as watcher

    monkeypatch.setattr(watcher, "AI_SOFT_TIMEOUT_ENABLED", True)
    monkeypatch.setattr(watcher, "AI_SOFT_TIMEOUT_SECONDS", 0.05)

    slow_verdict = DealVerdict(
        deal_quality="great", reason="Great deal (slow).", suggested_target=450, category="other", provider="gemini"
    )

    async def _slow_analyze(*args, **kwargs):
        await asyncio.sleep(0.2)  # well past the 0.05s soft timeout
        return slow_verdict

    bot = _fake_bot()
    with patch.object(watcher, "analyze_deal", new=_slow_analyze):
        await watcher._handle_post(bot, POST_TEXT, "test_channel")

    # Forwarded immediately with the placeholder — must not have waited for AI.
    bot.send_message.assert_called_once()
    placeholder_text = bot.send_message.call_args.kwargs["text"]
    assert "analyzing..." in placeholder_text
    bot.edit_message_text.assert_not_called()

    # Let the background task finish and edit the message.
    await asyncio.sleep(0.3)
    bot.edit_message_text.assert_called_once()
    edited_text = bot.edit_message_text.call_args.kwargs["text"]
    assert "analyzing..." not in edited_text
    assert "Great deal (slow)" in edited_text
    assert bot.edit_message_text.call_args.kwargs["chat_id"] == 111
    assert bot.edit_message_text.call_args.kwargs["message_id"] == 222


@pytest.mark.asyncio
async def test_ai_soft_timeout_leaves_message_unchanged_if_ai_ultimately_fails(monkeypatch):
    import listener.watcher as watcher

    monkeypatch.setattr(watcher, "AI_SOFT_TIMEOUT_ENABLED", True)
    monkeypatch.setattr(watcher, "AI_SOFT_TIMEOUT_SECONDS", 0.05)

    async def _slow_failing_analyze(*args, **kwargs):
        await asyncio.sleep(0.2)
        return None  # both providers ultimately unavailable

    bot = _fake_bot()
    with patch.object(watcher, "analyze_deal", new=_slow_failing_analyze):
        await watcher._handle_post(bot, POST_TEXT, "test_channel")

    bot.send_message.assert_called_once()
    await asyncio.sleep(0.3)
    bot.edit_message_text.assert_not_called()  # no duplicate notification, message left as-is


@pytest.mark.asyncio
async def test_background_task_does_not_increase_forwarding_latency(monkeypatch):
    """The whole point of the soft timeout: _handle_post must return almost
    immediately (well under the AI's real duration) once the timeout fires.
    """
    import time

    import listener.watcher as watcher

    monkeypatch.setattr(watcher, "AI_SOFT_TIMEOUT_ENABLED", True)
    monkeypatch.setattr(watcher, "AI_SOFT_TIMEOUT_SECONDS", 0.05)

    async def _slow_analyze(*args, **kwargs):
        await asyncio.sleep(1.0)
        return DealVerdict(
            deal_quality="good", reason="x", suggested_target=450, category="other", provider="groq"
        )

    bot = _fake_bot()
    start = time.perf_counter()
    with patch.object(watcher, "analyze_deal", new=_slow_analyze):
        await watcher._handle_post(bot, POST_TEXT, "test_channel")
    elapsed = time.perf_counter() - start

    assert elapsed < 0.5  # nowhere near the AI's 1.0s duration
    bot.send_message.assert_called_once()


@pytest.mark.asyncio
async def test_slow_request_triggers_warning_log(caplog, monkeypatch):
    import logging

    import listener.timing as timing_module

    # Lower the threshold rather than faking a stage duration — slow-request
    # detection is based on real measured wall-clock total time (what
    # actually matters in production), so the test must genuinely take
    # longer than the threshold, not just claim to via an injected number.
    monkeypatch.setattr(timing_module, "SLOW_REQUEST_THRESHOLD_SECONDS", 0.01)

    import listener.watcher as watcher

    bot = _fake_bot()
    fast_verdict = DealVerdict(
        deal_quality="great", reason="x", suggested_target=450, category="other", provider="gemini"
    )

    async def _analyze_slower_than_threshold(deal, price_history, *, timing=None, **_kwargs):
        await asyncio.sleep(0.05)
        return fast_verdict

    with patch.object(watcher, "analyze_deal", new=_analyze_slower_than_threshold), caplog.at_level(
        logging.WARNING, logger="fanzi.listener.timing"
    ):
        await watcher._handle_post(bot, POST_TEXT, "test_channel")

    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert any("SLOW DEAL" in r.message for r in warnings)
    assert any("B0TESTDEAL" in r.message for r in warnings)
