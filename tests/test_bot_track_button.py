"""Covers the 📉 Track Price button flow in bot.py. Regression test for a
real bug: the flow used to be a ConversationHandler, which only tracks one
active state per (chat, user) — pressing Track Price on a second deal while
an earlier deal's target-price choice was still unresolved matched no state
handler and silently did nothing. Plain handlers + pending_deals (keyed by
ASIN) fixed it; this test reproduces the exact concurrent scenario.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

import bot as bot_module
import database
from listener import pending_deals

ADMIN_ID = 777


class _FakeMessage:
    def __init__(self) -> None:
        self.replies: list[tuple[str, object]] = []

    async def reply_text(self, text, reply_markup=None):
        self.replies.append((text, reply_markup))


class _FakeQuery:
    def __init__(self, data: str) -> None:
        self.data = data
        self.message = _FakeMessage()

    async def answer(self):
        pass


class _FakeUpdate:
    def __init__(self, callback_query=None, message=None, user_id: int = ADMIN_ID) -> None:
        self.callback_query = callback_query
        self.message = message
        self.effective_user = MagicMock(id=user_id, username="admin")


class _FakeContext:
    def __init__(self) -> None:
        self.user_data: dict = {}


@pytest.mark.asyncio
async def test_second_track_button_still_responds_while_first_is_pending():
    pending_deals.store("B0DEALA001", "Deal A", "https://www.amazon.eg/dp/B0DEALA001", 1000.0, 20)
    context = _FakeContext()

    query_a = _FakeQuery("track:B0DEALA001")
    await bot_module.track_button_pressed(_FakeUpdate(callback_query=query_a), context)
    assert len(query_a.message.replies) == 1
    assert query_a.message.replies[0][1] is not None  # target-choice keyboard attached

    # Deal B's button is pressed before deal A's target choice was ever made.
    pending_deals.store("B0DEALB002", "Deal B", "https://www.amazon.eg/dp/B0DEALB002", 500.0, 15)
    query_b = _FakeQuery("track:B0DEALB002")
    await bot_module.track_button_pressed(_FakeUpdate(callback_query=query_b), context)
    assert len(query_b.message.replies) == 1
    assert query_b.message.replies[0][1] is not None  # must still show the keyboard, not be dropped

    # Finishing deal A afterwards must use deal A's own data, not deal B's.
    query_a2 = _FakeQuery("target:B0DEALA001:10")
    await bot_module.target_percentage_chosen(_FakeUpdate(callback_query=query_a2), context)
    user = database.get_or_create_user(ADMIN_ID, "admin")
    tracked_a = database.get_tracked_product_by_asin(user.id, "B0DEALA001")
    assert tracked_a is not None
    assert tracked_a.target_price == pytest.approx(900.0)

    # Finishing deal B afterwards (custom price) must use deal B's own data.
    query_b2 = _FakeQuery("target:B0DEALB002:custom")
    await bot_module.target_custom_prompt(_FakeUpdate(callback_query=query_b2), context)

    text_message = _FakeMessage()
    text_message.text = "321"

    class _FakeTextUpdate:
        def __init__(self) -> None:
            self.message = text_message
            self.effective_user = MagicMock(id=ADMIN_ID, username="admin")
            self.callback_query = None

    await bot_module.maybe_custom_target_received(_FakeTextUpdate(), context)
    tracked_b = database.get_tracked_product_by_asin(user.id, "B0DEALB002")
    assert tracked_b is not None
    assert tracked_b.target_price == 321.0


@pytest.mark.asyncio
async def test_track_button_already_tracked_shows_message_not_duplicate():
    user = database.get_or_create_user(ADMIN_ID, "admin")
    database.add_tracked_product(
        user_id=user.id,
        asin="B0ALREADY1",
        title="Already tracked",
        url="https://www.amazon.eg/dp/B0ALREADY1",
        current_price=100.0,
        target_price=90.0,
    )
    pending_deals.store("B0ALREADY1", "Already tracked", "https://www.amazon.eg/dp/B0ALREADY1", 100.0, 20)

    query = _FakeQuery("track:B0ALREADY1")
    context = _FakeContext()
    await bot_module.track_button_pressed(_FakeUpdate(callback_query=query), context)

    assert len(query.message.replies) == 1
    assert "already tracking" in query.message.replies[0][0].lower()

    with database.get_connection() as conn:
        count = conn.execute(
            "SELECT COUNT(*) AS n FROM tracked_products WHERE asin = ?", ("B0ALREADY1",)
        ).fetchone()["n"]
    assert count == 1


@pytest.mark.asyncio
async def test_maybe_custom_target_received_ignores_unrelated_text():
    """Must not react to ordinary text when no Custom price prompt is
    outstanding — otherwise it would interfere with /track's own flow.
    """
    context = _FakeContext()
    text_message = _FakeMessage()
    text_message.text = "hello"

    class _FakeTextUpdate:
        def __init__(self) -> None:
            self.message = text_message
            self.effective_user = MagicMock(id=ADMIN_ID, username="admin")
            self.callback_query = None

    await bot_module.maybe_custom_target_received(_FakeTextUpdate(), context)
    assert text_message.replies == []
