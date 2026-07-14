"""Covers the production fix for cross-channel duplicate notifications,
canonical (never-original) product URLs, and AI payload sanitization.
Zero real Gemini/Groq calls.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from listener.ai_providers import AIVerdict
from listener.analyzer import DealVerdict

# Real production case: the same product (Tornado TOF-49Y fan) posted by two
# different channels using two different link.amazon short codes.
TORNADO_TEXT_CHANNEL_A = (
    "مروحة تورنيدو TOF-49Y سعر 1799 جنيه\nhttps://link.amazon/B0AAAAAAAA"
)
TORNADO_TEXT_CHANNEL_B = (
    "مروحة تورنيدو TOF-49Y سعر 1799 جنيه\nhttps://link.amazon/B0BBBBBBBB"
)


def _fake_bot():
    bot = AsyncMock()
    bot.send_message = AsyncMock(return_value=MagicMock(chat_id=1, message_id=2))
    bot.edit_message_text = AsyncMock()
    return bot


def _fake_verdict(quality="great"):
    return DealVerdict(deal_quality=quality, reason="x", suggested_target=1600, category="appliance", provider="gemini")


async def _resolve_asin_same_product(url, client=None):
    # Both short links resolve to the exact same real product ASIN --
    # simulates two different tracking/shortener links for one product.
    return "B0TORNADO1"


@pytest.mark.asyncio
async def test_same_asin_from_two_channels_forwards_exactly_once():
    import listener.watcher as watcher_module

    bot = _fake_bot()
    verdict = _fake_verdict()

    with patch("listener.parser.extract_asin", new=AsyncMock(side_effect=_resolve_asin_same_product)), patch.object(
        watcher_module, "analyze_deal", new=AsyncMock(return_value=verdict)
    ):
        await watcher_module._handle_post(bot, TORNADO_TEXT_CHANNEL_A, "OffersEgyptofficial", message_id=1)
        await watcher_module._handle_post(bot, TORNADO_TEXT_CHANNEL_B, "Mego_Reviews", message_id=2)

    # Exactly one notification for the same real-world product, even though
    # it arrived from two different channels via two different links.
    bot.send_message.assert_called_once()


@pytest.mark.asyncio
async def test_same_asin_lower_price_from_another_channel_is_forwarded():
    import listener.watcher as watcher_module

    bot = _fake_bot()

    with patch("listener.parser.extract_asin", new=AsyncMock(side_effect=_resolve_asin_same_product)), patch.object(
        watcher_module, "analyze_deal", new=AsyncMock(return_value=_fake_verdict())
    ):
        await watcher_module._handle_post(bot, TORNADO_TEXT_CHANNEL_A, "OffersEgyptofficial", message_id=1)
        lower_price_text = "مروحة تورنيدو TOF-49Y سعر 1599 جنيه\nhttps://link.amazon/B0CCCCCCCC"
        await watcher_module._handle_post(bot, lower_price_text, "Mego_Reviews", message_id=2)

    assert bot.send_message.call_count == 2  # price drop -> forwarded again, not suppressed


@pytest.mark.asyncio
async def test_same_asin_after_window_expires_is_forwarded():
    import database
    import listener.watcher as watcher_module
    from datetime import datetime, timedelta

    bot = _fake_bot()
    # Pre-seed a stale dedup record (well past DUPLICATE_WINDOW_HOURS).
    stale_time = (datetime.now() - timedelta(hours=100)).isoformat()
    database.upsert_global_duplicate_record("asin:B0TORNADO1", "OldChannel", 1799.0, 15, stale_time)

    with patch("listener.parser.extract_asin", new=AsyncMock(side_effect=_resolve_asin_same_product)), patch.object(
        watcher_module, "analyze_deal", new=AsyncMock(return_value=_fake_verdict())
    ):
        await watcher_module._handle_post(bot, TORNADO_TEXT_CHANNEL_A, "OffersEgyptofficial", message_id=1)

    bot.send_message.assert_called_once()


@pytest.mark.asyncio
async def test_replay_across_channels_cannot_duplicate_notify():
    """The same product replayed from two different channels' history must
    still forward only once -- replay uses the exact same _handle_post
    pipeline, so global dedup applies identically.
    """
    import listener.watcher as watcher_module
    from listener import replay

    class _FakeMessage:
        def __init__(self, msg_id, text):
            self.id = msg_id
            self.message = text
            self.peer_id = MagicMock(channel_id=999)

    class _FakeClient:
        def __init__(self, messages):
            self._messages = messages

        async def get_messages(self, channel, limit=50):
            return list(self._messages)

    bot = _fake_bot()
    with patch("listener.parser.extract_asin", new=AsyncMock(side_effect=_resolve_asin_same_product)), patch.object(
        watcher_module, "analyze_deal", new=AsyncMock(return_value=_fake_verdict())
    ):
        await replay.replay_channel(
            _FakeClient([_FakeMessage(1, TORNADO_TEXT_CHANNEL_A)]), bot, "channel_x", watcher_module._handle_post, 50
        )
        await replay.replay_channel(
            _FakeClient([_FakeMessage(1, TORNADO_TEXT_CHANNEL_B)]), bot, "channel_y", watcher_module._handle_post, 50
        )

    bot.send_message.assert_called_once()


@pytest.mark.asyncio
async def test_forwarded_message_uses_canonical_url_not_original_link():
    import listener.watcher as watcher_module

    bot = _fake_bot()
    with patch("listener.parser.extract_asin", new=AsyncMock(side_effect=_resolve_asin_same_product)), patch.object(
        watcher_module, "analyze_deal", new=AsyncMock(return_value=_fake_verdict())
    ):
        await watcher_module._handle_post(bot, TORNADO_TEXT_CHANNEL_A, "OffersEgyptofficial", message_id=1)

    sent_text = bot.send_message.call_args.kwargs["text"]
    assert "https://www.amazon.eg/dp/B0TORNADO1" in sent_text
    assert "link.amazon" not in sent_text
    assert "B0AAAAAAAA" not in sent_text


@pytest.mark.asyncio
async def test_no_asin_means_no_message_and_no_product_url():
    import listener.watcher as watcher_module

    bot = _fake_bot()
    text = "Just some channel chatter with no product link or ASIN at all."
    await watcher_module._handle_post(bot, text, "no_asin_channel_2", message_id=1)

    bot.send_message.assert_not_called()  # structurally impossible to include a URL for a deal that never existed


def test_ai_payload_contains_no_url_asin_or_tracking_parameters():
    from listener.ai_providers import _build_user_content

    content = _build_user_content(
        title="Tornado TOF-49Y fan",
        price=1799.0,
        discount_percent=15,
        channel_name="OffersEgyptofficial",
        price_history=None,
        brand="Tornado",
        category_hint="appliance",
    )

    assert "http://" not in content
    assert "https://" not in content
    assert "link.amazon" not in content
    assert "tinyurl" not in content
    assert "bit.ly" not in content
    assert "tag=" not in content
    assert "asin" not in content.lower() or "B0" not in content  # no ASIN value leaked
    assert "brand=Tornado" in content
    assert "category_hint=appliance" in content
    assert "title='Tornado TOF-49Y fan'" in content


def test_ai_payload_strips_url_even_if_it_leaks_into_title():
    from listener.ai_providers import _build_user_content

    content = _build_user_content(
        title="Deal here https://link.amazon/B0SNEAKY1 grab it",
        price=500.0,
        discount_percent=10,
        channel_name="chan",
        price_history=None,
        brand=None,
        category_hint=None,
    )
    assert "http" not in content
    assert "link.amazon" not in content


@pytest.mark.asyncio
async def test_get_verdict_sends_only_structured_fields_to_provider():
    """End-to-end: whatever the real Gemini/Groq generate() call actually
    receives as user_content must be clean, even when the raw post text
    (which analyze_deal never even passes anymore) contained tracking links.
    """
    from listener.ai_providers import AIProviderManager, GeminiProvider, GroqProvider

    manager = AIProviderManager(GeminiProvider(), GroqProvider())
    captured = {}

    async def _capture_generate(user_content, strict=False):
        captured["content"] = user_content
        return (
            '{"deal_quality": "great", "reason": "x", "suggested_target": 1600, "category": "appliance"}'
        )

    with patch.object(manager.gemini, "generate", new=_capture_generate):
        await manager.get_verdict(
            title="Tornado TOF-49Y fan",
            price=1799.0,
            discount_percent=15,
            channel_name="OffersEgyptofficial",
            price_history=None,
            brand="Tornado",
            category_hint="appliance",
        )

    assert "link.amazon" not in captured["content"]
    assert "http" not in captured["content"]
    assert "B0" not in captured["content"]
