"""Covers Product Family detection (listener/family.py): variant extraction,
deterministic/fuzzy/AI family matching, true-duplicate suppression vs.
"Better Variant Found"/"New Variant Available" notifications, permanent AI
decision caching, and that listener.replay's shared _handle_post pipeline
applies this logic identically to replayed messages. Zero real Gemini/Groq
calls -- every AI similarity check is explicitly mocked; anything reaching
a real provider client raises (see tests/conftest.py's
block_real_ai_providers safety net).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import database
from listener import family
from listener.analyzer import DealVerdict


# --- Variant extraction ----------------------------------------------------


def test_extract_color_variant():
    assert family.extract_variant_attributes("Mintra Winter Slippers Red")["color"] == "Red"


def test_extract_size_variant_with_anchor_word():
    assert family.extract_variant_attributes("Nike Shoes Size 42")["size"] == "42"


def test_extract_capacity_variant():
    assert family.extract_variant_attributes("Samsung Galaxy 128GB")["capacity"] == "128GB"


def test_extract_pack_variant():
    assert family.extract_variant_attributes("AA Batteries Pack of 6")["pack"] == "6 Pack"


def test_normalize_strips_variant_tokens_so_colors_match():
    a = family.normalize_title_for_family("Mintra Winter Slippers Red")
    b = family.normalize_title_for_family("Mintra Winter Slippers Black")
    assert a == b


# --- Deterministic matching + duplicate suppression / better-or-new -------


@pytest.mark.asyncio
async def test_new_family_created_on_first_sighting():
    decision = await family.pre_check(
        "B0RED00001", "Mintra Winter Slippers Red", 84.0, 15, brand=None, category="accessory"
    )
    assert decision.notify_kind == "new_family"
    assert decision.variant == {"color": "Red"}
    fam = database.get_product_family(decision.family_id)
    assert fam["lowest_price"] == 84.0
    assert fam["variant_count"] == 1


@pytest.mark.asyncio
async def test_same_family_same_variant_same_price_is_suppressed_as_duplicate():
    await family.pre_check("B0RED00001", "Mintra Winter Slippers Red", 84.0, 15, brand=None, category="accessory")
    # Same color, price within +-1 EGP tolerance, same discount -- a
    # different ASIN (e.g. a reseller duplicate listing) posting the exact
    # same variant must be suppressed.
    decision = await family.pre_check(
        "B0RED00002", "Mintra Winter Slippers Red", 84.5, 15, brand=None, category="accessory"
    )
    assert decision.notify_kind == "duplicate"


@pytest.mark.asyncio
async def test_same_family_different_color_cheaper_is_better_variant():
    await family.pre_check("B0RED00001", "Mintra Winter Slippers Red", 99.0, 10, brand=None, category="accessory")
    decision = await family.pre_check(
        "B0BLK00001", "Mintra Winter Slippers Black", 84.0, 10, brand=None, category="accessory"
    )
    assert decision.notify_kind == "pending"
    final = family.finalize(decision.family_id, "B0BLK00001", decision.variant, 84.0, 10, verdict_quality="good")
    assert final.notify_kind == "better_variant"
    assert final.previous_best_price == 99.0
    assert final.savings == 15.0
    fam = database.get_product_family(decision.family_id)
    assert fam["lowest_price"] == 84.0
    assert fam["best_variant_label"] == "Black"


@pytest.mark.asyncio
async def test_same_family_different_color_pricier_is_new_variant():
    await family.pre_check("B0RED00001", "Mintra Winter Slippers Red", 84.0, 15, brand=None, category="accessory")
    decision = await family.pre_check(
        "B0BLU00001", "Mintra Winter Slippers Blue", 99.0, 5, brand=None, category="accessory"
    )
    final = family.finalize(decision.family_id, "B0BLU00001", decision.variant, 99.0, 5, verdict_quality="average")
    assert final.notify_kind == "new_variant"
    assert final.previous_best_price == 84.0
    # Family's best price/label are unchanged -- this variant didn't win.
    fam = database.get_product_family(decision.family_id)
    assert fam["lowest_price"] == 84.0
    assert fam["best_variant_label"] == "Red"


@pytest.mark.asyncio
async def test_lower_absolute_price_wins_even_with_lower_discount():
    """Acceptance test 5: lower discount but lower absolute price must still
    notify as a better variant -- price alone is sufficient.
    """
    await family.pre_check("B0RED00001", "Mintra Winter Slippers Red", 100.0, 50, brand=None, category="accessory")
    decision = await family.pre_check(
        "B0BLK00001", "Mintra Winter Slippers Black", 90.0, 5, brand=None, category="accessory"
    )
    final = family.finalize(decision.family_id, "B0BLK00001", decision.variant, 90.0, 5, verdict_quality=None)
    assert final.notify_kind == "better_variant"


@pytest.mark.asyncio
async def test_new_family_for_unrelated_product():
    slippers = await family.pre_check(
        "B0RED00001", "Mintra Winter Slippers Red", 84.0, 15, brand=None, category="accessory"
    )
    power_bank = await family.pre_check(
        "B0OTHER001", "Anker Power Bank 20000mAh", 500.0, 20, brand="Anker", category="accessory"
    )
    assert power_bank.notify_kind == "new_family"
    assert power_bank.family_id != slippers.family_id
    assert database.count_product_families() == 2


# --- AI similarity: only in the ambiguous confidence band, cached forever -


def _fake_similarity_provider(same_family=True, confidence=0.9):
    async def _generate(user_content, *, strict=False, system_prompt=None):
        return (
            '{"same_family": %s, "confidence": %.2f, "reason": "same product, different variant"}'
            % ("true" if same_family else "false", confidence)
        )
    return _generate


@pytest.mark.asyncio
async def test_ai_similarity_consulted_only_in_ambiguous_band(monkeypatch):
    # Force everything that isn't an exact/near-exact title match into the
    # ambiguous band, so a moderately-similar title must go through AI.
    monkeypatch.setattr(family, "FAMILY_FUZZY_MATCH_THRESHOLD", 0.999)
    monkeypatch.setattr(family, "FAMILY_AI_MATCH_FLOOR", 0.01)

    await family.pre_check("B0ANCHOR01", "Mintra Winter Slippers", 84.0, 15, brand=None, category="accessory")

    from listener.ai_providers import get_manager

    manager = get_manager()
    with patch.object(manager.gemini, "generate", new=_fake_similarity_provider(same_family=True)):
        decision = await family.pre_check(
            "B0CANDID01", "Mintra Winter Slipper Shoes", 90.0, 10, brand=None, category="accessory"
        )

    assert decision.matched_via == "ai"
    assert decision.notify_kind == "pending"


@pytest.mark.asyncio
async def test_ai_not_consulted_for_clearly_different_brand_category():
    """Different brand/category never even reaches similarity scoring --
    the candidate pool itself is empty, so AI is structurally never asked.
    block_real_ai_providers (conftest) would raise if it somehow were.
    """
    await family.pre_check("B0ANCHOR01", "Mintra Winter Slippers", 84.0, 15, brand=None, category="accessory")
    decision = await family.pre_check(
        "B0OTHERBR1", "Samsung Galaxy Phone", 5000.0, 5, brand="Samsung", category="phone"
    )
    assert decision.notify_kind == "new_family"
    assert decision.matched_via == "new"


@pytest.mark.asyncio
async def test_ai_decision_cached_permanently_never_asked_twice(monkeypatch):
    monkeypatch.setattr(family, "FAMILY_FUZZY_MATCH_THRESHOLD", 0.999)
    monkeypatch.setattr(family, "FAMILY_AI_MATCH_FLOOR", 0.01)

    anchor_family_row = {
        "family_id": "fam_test_anchor", "brand": None, "normalized_title": "widget alpha",
        "category": "accessory", "anchor_asin": "B0ANCHOR99",
    }
    database.create_product_family(
        family_id="fam_test_anchor", brand=None, normalized_title="widget alpha", category="accessory",
        deterministic_key="test-key-anchor", anchor_asin="B0ANCHOR99",
        lowest_price=50.0, highest_discount_percent=10,
        first_seen="2026-01-01T00:00:00", last_seen="2026-01-01T00:00:00",
    )

    from listener.ai_providers import get_manager
    manager = get_manager()
    call_count = {"n": 0}

    async def _counting_generate(user_content, *, strict=False, system_prompt=None):
        call_count["n"] += 1
        return '{"same_family": true, "confidence": 0.9, "reason": "same widget"}'

    with patch.object(manager.gemini, "generate", new=_counting_generate):
        first = await family._fuzzy_or_ai_match(
            "B0CANDIDATE1", "Widget Alpha Variant", "widget alpha variant", None, "accessory"
        )
        second = await family._fuzzy_or_ai_match(
            "B0CANDIDATE1", "Widget Alpha Variant", "widget alpha variant", None, "accessory"
        )

    assert call_count["n"] == 1  # second call served entirely from the permanent cache
    assert first[0] is not None and second[0] is not None
    assert first[0]["family_id"] == second[0]["family_id"]


# --- Replay respects Product Family logic (test 9) -------------------------


def _fake_bot():
    bot = AsyncMock()
    bot.send_message = AsyncMock(return_value=MagicMock(chat_id=1, message_id=2))
    bot.edit_message_text = AsyncMock()
    return bot


def _fake_verdict(quality="good"):
    return DealVerdict(deal_quality=quality, reason="x", suggested_target=80, category="accessory", provider="gemini")


@pytest.mark.asyncio
async def test_replay_suppresses_true_variant_duplicate_across_channels():
    import listener.watcher as watcher_module
    from listener import replay

    text_red_a = "Mintra Winter Slippers Red سعر 84 جنيه\nhttps://link.amazon/B0REDVARA"
    text_red_b = "Mintra Winter Slippers Red سعر 84 جنيه\nhttps://link.amazon/B0REDVARB"

    async def _resolve(url, client=None):
        return "B0REDVARA" if "B0REDVARA" in url else "B0REDVARB2"

    bot = _fake_bot()

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

    with patch("listener.parser.extract_asin", new=AsyncMock(side_effect=_resolve)), patch.object(
        watcher_module, "analyze_deal", new=AsyncMock(return_value=_fake_verdict())
    ):
        await replay.replay_channel(
            _FakeClient([_FakeMessage(1, text_red_a)]), bot, "channel_x", watcher_module._handle_post, 50
        )
        await replay.replay_channel(
            _FakeClient([_FakeMessage(1, text_red_b)]), bot, "channel_y", watcher_module._handle_post, 50
        )

    # Both replayed messages describe the identical variant at the identical
    # price -- family-level true-duplicate suppression must catch what
    # global ASIN dedup alone would have let through as "new" (different
    # ASINs), exactly like it would for a live message.
    bot.send_message.assert_called_once()
