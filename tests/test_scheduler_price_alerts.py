"""Covers scheduler.py's price-change state machine: general price-change
notifications fire on ANY movement (independent of target price), target
alerts keep working independently, availability transitions notify, and
deal-forwarding dedup is never consulted for tracked products. Zero real
network calls (amazon.tracker.fetch_product is mocked); zero real
Gemini/Groq calls (nothing here touches AI at all).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

import database
import scheduler
from amazon.tracker import PriceNotFoundError


def _make_product(price: float, target: float = 500.0, available: bool = True) -> int:
    user = database.get_or_create_user(111, "tester")
    product = database.add_tracked_product(
        user_id=user.id, asin="B0PRICEALERT", title="Kenwood Sandwich Maker",
        url="https://www.amazon.eg/dp/B0PRICEALERT", current_price=price, target_price=target,
    )
    database.update_price_check(product.id, price, available=available)
    return product.id


def _fake_bot():
    return AsyncMock()


@pytest.mark.asyncio
async def test_price_decrease_1000_to_999_notifies():
    _make_product(1000.0)
    bot = _fake_bot()
    with patch.object(scheduler, "fetch_product", new=AsyncMock(return_value=("Kenwood Sandwich Maker", 999.0))):
        await scheduler.run_check_cycle(bot)
    bot.send_message.assert_called_once()
    assert "Price Update" in bot.send_message.call_args.kwargs["text"]
    assert "-1 EGP" in bot.send_message.call_args.kwargs["text"]


@pytest.mark.asyncio
async def test_price_decrease_999_to_998_notifies():
    _make_product(999.0)
    bot = _fake_bot()
    with patch.object(scheduler, "fetch_product", new=AsyncMock(return_value=("Kenwood Sandwich Maker", 998.0))):
        await scheduler.run_check_cycle(bot)
    bot.send_message.assert_called_once()


@pytest.mark.asyncio
async def test_price_increase_998_to_999_notifies():
    _make_product(998.0)
    bot = _fake_bot()
    with patch.object(scheduler, "fetch_product", new=AsyncMock(return_value=("Kenwood Sandwich Maker", 999.0))):
        await scheduler.run_check_cycle(bot)
    bot.send_message.assert_called_once()
    text = bot.send_message.call_args.kwargs["text"]
    assert "Price Update" in text
    assert "+1 EGP" in text


@pytest.mark.asyncio
async def test_identical_price_999_to_999_does_not_notify():
    _make_product(999.0, target=500.0)
    bot = _fake_bot()
    with patch.object(scheduler, "fetch_product", new=AsyncMock(return_value=("Kenwood Sandwich Maker", 999.0))):
        await scheduler.run_check_cycle(bot)
    bot.send_message.assert_not_called()


@pytest.mark.asyncio
async def test_unavailable_to_1200_notifies_back_in_stock():
    _make_product(1200.0, available=False)
    bot = _fake_bot()
    with patch.object(scheduler, "fetch_product", new=AsyncMock(return_value=("Kenwood Sandwich Maker", 1200.0))):
        await scheduler.run_check_cycle(bot)
    bot.send_message.assert_called_once()
    assert "Back In Stock" in bot.send_message.call_args.kwargs["text"]


@pytest.mark.asyncio
async def test_1200_to_unavailable_notifies():
    _make_product(1200.0, available=True)
    bot = _fake_bot()
    with patch.object(scheduler, "fetch_product", new=AsyncMock(side_effect=PriceNotFoundError("no price element"))):
        await scheduler.run_check_cycle(bot)
    bot.send_message.assert_called_once()
    assert "Unavailable" in bot.send_message.call_args.kwargs["text"]


@pytest.mark.asyncio
async def test_target_reached_alert_still_fires():
    _make_product(1600.0, target=1500.0)
    bot = _fake_bot()
    with patch.object(scheduler, "fetch_product", new=AsyncMock(return_value=("Kenwood Sandwich Maker", 1498.0))):
        await scheduler.run_check_cycle(bot)
    # Both the general price-change alert AND the target-reached alert fire
    # -- "in addition to", not instead of.
    assert bot.send_message.call_count == 2
    texts = [call.kwargs["text"] for call in bot.send_message.call_args_list]
    assert any("Price Update" in t for t in texts)
    assert any("Target Reached" in t for t in texts)


@pytest.mark.asyncio
async def test_deal_dedup_cannot_suppress_tracked_product_notifications():
    """Even if the exact same ASIN is already marked as a forwarded deal in
    global_duplicate_deals, the tracked-product price-change notification
    must still fire -- the two systems share nothing.
    """
    from listener import dedup

    product_id = _make_product(1000.0)
    row = database.get_active_products(database.get_or_create_user(111, "tester").id)[0]
    assert row.asin == "B0PRICEALERT"

    # Pre-mark this exact ASIN as an already-forwarded deal (global dedup).
    dedup.mark_seen("B0PRICEALERT", "Kenwood Sandwich Maker", 999.0, None, channel_name="some_channel")
    assert dedup.check("B0PRICEALERT", "Kenwood Sandwich Maker", 999.0, None, channel_name="some_channel") == dedup.DUPLICATE

    bot = _fake_bot()
    with patch.object(scheduler, "fetch_product", new=AsyncMock(return_value=("Kenwood Sandwich Maker", 999.0))):
        await scheduler.run_check_cycle(bot)

    bot.send_message.assert_called_once()  # price-change notification still sent


@pytest.mark.asyncio
async def test_run_check_cycle_never_calls_dedup_check_or_mark_seen():
    _make_product(1000.0)
    bot = _fake_bot()
    with patch.object(scheduler, "fetch_product", new=AsyncMock(return_value=("Kenwood Sandwich Maker", 999.0))), patch(
        "listener.dedup.check"
    ) as dedup_check, patch("listener.dedup.mark_seen") as dedup_mark_seen:
        await scheduler.run_check_cycle(bot)

    dedup_check.assert_not_called()
    dedup_mark_seen.assert_not_called()


@pytest.mark.asyncio
async def test_fetch_failure_does_not_change_state_or_notify():
    from amazon.tracker import FetchTimeoutError

    product_id = _make_product(1000.0)
    bot = _fake_bot()
    with patch.object(scheduler, "fetch_product", new=AsyncMock(side_effect=FetchTimeoutError("timed out"))):
        await scheduler.run_check_cycle(bot)

    bot.send_message.assert_not_called()
    user = database.get_or_create_user(111, "tester")
    product = database.get_active_products(user.id)[0]
    assert product.current_price == 1000.0  # unchanged -- retried next cycle
    assert product.available is True


@pytest.mark.asyncio
async def test_replay_cannot_generate_duplicate_tracked_product_alerts():
    """Tracked-product alerts are only ever produced by run_check_cycle --
    listener.replay never touches tracked_products/scheduler at all, so
    replaying channel history structurally cannot duplicate a price alert.
    """
    import inspect

    from listener import replay

    source = inspect.getsource(replay)
    assert "tracked_product" not in source.lower()
    assert "scheduler" not in source.lower()
