"""Covers the self-improving knowledge engine in listener/learning.py and its
integration into listener/analyzer.py: rule creation from repeated AI
verdicts, rule firing (AI call saved), validation sampling, outlier bypass,
provider-agnostic learning, and rule persistence/rebuild. No real Gemini or
Groq call is ever made — analyze_deal's AI path is exercised via a mocked
AIProviderManager, exactly like tests/test_analyzer.py.
"""

from __future__ import annotations

from datetime import date
from unittest.mock import AsyncMock, patch

import pytest

import database
from listener import learning
from listener.ai_providers import AIVerdict
from listener.analyzer import analyze_deal
from listener.parser import ParsedDeal


def _anker_charger_deal(price: float = 300.0, discount: int = 25) -> ParsedDeal:
    return ParsedDeal(
        asin="B0ANKERCHG1",
        title="Anker PowerLine III شاحن سريع لاجهزة ابل",
        price=price,
        discount_percent=discount,
        channel_name="test_channel",
        raw_text="test post",
        url="https://www.amazon.eg/dp/B0ANKERCHG1",
    )


def _fake_manager(deal_quality: str, provider: str = "gemini") -> AsyncMock:
    manager = AsyncMock()
    manager.get_verdict = AsyncMock(
        return_value=AIVerdict(
            provider=provider,
            deal_quality=deal_quality,
            reason="A solid Anker charger deal.",
            suggested_target=int(300 * 0.9),
            category="cable",
        )
    )
    return manager


async def _run_deal_through_ai(deal_quality: str = "great", provider: str = "gemini", price: float = 300.0, discount: int = 25):
    deal = _anker_charger_deal(price=price, discount=discount)
    manager = _fake_manager(deal_quality, provider)
    with patch("listener.analyzer.get_manager", return_value=manager):
        verdict = await analyze_deal(deal, None)
    # Give the fire-and-forget learning task a chance to run.
    import asyncio
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    return verdict, manager


@pytest.mark.asyncio
async def test_five_anker_charger_deals_create_a_rule():
    for _ in range(5):
        verdict, manager = await _run_deal_through_ai("great")
        manager.get_verdict.assert_called_once()

    rule = database.get_learned_rule("brand", "Anker")
    assert rule is not None
    assert rule["sample_count"] == 5
    assert rule["predicted_quality"] == "great"
    assert rule["enabled"] == 1

    brand_category_rule = database.get_learned_rule("brand_category", "Anker|cable")
    assert brand_category_rule is not None
    assert brand_category_rule["sample_count"] == 5


@pytest.mark.asyncio
async def test_sixth_similar_deal_fires_rule_without_ai_call(monkeypatch):
    for _ in range(5):
        await _run_deal_through_ai("great")

    # At 100% confidence there's still a small (2%) chance of a random
    # validation call by design — pin the roll so this test deterministically
    # exercises the "rule fires" path, not the "validation" path (which has
    # its own dedicated test below).
    monkeypatch.setattr("listener.learning.random.random", lambda: 0.99)

    today = date.today().isoformat()
    saved_before = database.get_learning_stats(today)["ai_calls_saved"]

    deal = _anker_charger_deal()
    manager = _fake_manager("great")
    with patch("listener.analyzer.get_manager", return_value=manager):
        verdict = await analyze_deal(deal, None)

    manager.get_verdict.assert_not_called()
    assert verdict is not None
    assert verdict.provider == "learned"
    assert verdict.deal_quality == "great"
    assert "Learned Pattern" in verdict.reason
    assert "Anker" in verdict.reason

    saved_after = database.get_learning_stats(today)["ai_calls_saved"]
    assert saved_after == saved_before + 1


@pytest.mark.asyncio
async def test_random_validation_triggers_ai_despite_confident_rule(monkeypatch):
    for _ in range(5):
        await _run_deal_through_ai("great")

    # Force the validation roll to always pick AI, and confirm confidence
    # updates correctly when the fresh AI verdict disagrees with the rule.
    monkeypatch.setattr("listener.learning.random.random", lambda: 0.0)

    deal = _anker_charger_deal()
    manager = _fake_manager("average")  # disagrees with the learned "great"
    with patch("listener.analyzer.get_manager", return_value=manager):
        verdict = await analyze_deal(deal, None)
    manager.get_verdict.assert_called_once()
    assert verdict is not None
    assert verdict.deal_quality == "average"  # real AI answer, not the rule's

    import asyncio
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    today = date.today().isoformat()
    stats = database.get_learning_stats(today)
    assert stats["validation_calls"] == 1
    assert stats["validation_disagreements"] == 1

    # brand_category has higher priority than brand, so it's the rule that
    # was actually checked/validated here.
    rule = database.get_learned_rule("brand_category", "Anker|cable")
    assert rule["last_validated"] is not None
    # 5 "great" + 1 "average" vote → confidence should have dropped from 100%.
    assert rule["confidence"] < 1.0

    # Both rule types learn from every real verdict regardless of which one
    # triggered the validation, so "brand" also reflects the disagreement.
    brand_rule = database.get_learned_rule("brand", "Anker")
    assert brand_rule["confidence"] < 1.0


@pytest.mark.asyncio
async def test_unknown_brand_always_uses_ai():
    deal = ParsedDeal(
        asin="B0UNKNOWN1",
        title="Generic no-name charger cable",
        price=150.0,
        discount_percent=20,
        channel_name="test_channel",
        raw_text="test post",
        url="https://www.amazon.eg/dp/B0UNKNOWN1",
    )
    manager = _fake_manager("good")
    with patch("listener.analyzer.get_manager", return_value=manager):
        verdict = await analyze_deal(deal, None)
    manager.get_verdict.assert_called_once()
    assert verdict.provider == "gemini"


@pytest.mark.asyncio
async def test_groq_verdicts_contribute_to_learning_like_gemini():
    for _ in range(5):
        await _run_deal_through_ai("great", provider="groq")

    rule = database.get_learned_rule("brand", "Anker")
    assert rule is not None
    assert rule["sample_count"] == 5
    assert rule["predicted_quality"] == "great"


@pytest.mark.asyncio
async def test_outlier_discount_forces_ai_bypassing_rule():
    for _ in range(5):
        await _run_deal_through_ai("great")

    # >50% discount is an outlier — AI must be forced even though a
    # confident Anker rule now exists.
    deal = _anker_charger_deal(discount=60)
    manager = _fake_manager("great")
    with patch("listener.analyzer.get_manager", return_value=manager):
        verdict = await analyze_deal(deal, None)
    manager.get_verdict.assert_called_once()
    assert verdict.provider == "gemini"


@pytest.mark.asyncio
async def test_rules_persist_and_are_visible_from_a_fresh_read():
    """Simulates a restart: rules written via analyze_deal must be visible
    from a plain database read, not just in-memory.
    """
    for _ in range(5):
        await _run_deal_through_ai("great")

    rows = database.list_learned_rules(enabled_only=True)
    assert any(r["rule_type"] == "brand" and r["key"] == "Anker" for r in rows)

    today = date.today().isoformat()
    stats = database.get_learning_stats(today)
    assert stats["ai_calls_saved"] == 0  # no rule fired yet in this test


def test_rules_ordered_by_confidence_descending():
    now = "2026-01-01T00:00:00"
    database.upsert_learned_rule(
        rule_type="brand", key="Baseus", predicted_quality="good", confidence=0.84,
        sample_count=9, last_updated=now, last_validated=None, enabled=True, rule_version=1,
    )
    database.upsert_learned_rule(
        rule_type="brand_category", key="Anker|cable", predicted_quality="great", confidence=0.92,
        sample_count=18, last_updated=now, last_validated=None, enabled=True, rule_version=1,
    )
    rows = database.list_learned_rules(enabled_only=True)
    assert [r["key"] for r in rows] == ["Anker|cable", "Baseus"]


@pytest.mark.asyncio
async def test_status_reflects_learning_statistics(monkeypatch):
    for _ in range(5):
        await _run_deal_through_ai("great")
    monkeypatch.setattr("listener.learning.random.random", lambda: 0.99)
    deal = _anker_charger_deal()
    manager = _fake_manager("great")
    with patch("listener.analyzer.get_manager", return_value=manager):
        await analyze_deal(deal, None)  # this one should fire the rule

    import health
    message = health.format_status_message()
    assert "Learned rules" in message
    assert "AI calls saved today: 1" in message


@pytest.mark.asyncio
async def test_rebuildrules_reconstructs_from_verdict_history():
    for _ in range(5):
        await _run_deal_through_ai("great")

    database.clear_learned_rules()
    assert database.get_learned_rule("brand", "Anker") is None

    verdict_count, rule_count = learning.rebuild_rules_from_history()
    assert verdict_count == 5
    assert rule_count > 0
    rule = database.get_learned_rule("brand", "Anker")
    assert rule is not None
    assert rule["sample_count"] == 5
    assert rule["predicted_quality"] == "great"


def test_extract_brand_case_insensitive_and_no_guessing():
    assert learning.extract_brand("ANKER powerbank 10000mAh") == "Anker"
    assert learning.extract_brand("some anker product") == "Anker"
    assert learning.extract_brand("Totally unknown gadget brand") is None
    assert learning.extract_brand(None) is None


def test_price_and_discount_buckets():
    assert learning.price_bucket(50) == "0-100"
    assert learning.price_bucket(150) == "100-200"
    assert learning.price_bucket(1500) == "1000+"
    assert learning.discount_bucket(5) == "0-10%"
    assert learning.discount_bucket(45) == "30-50%"
    assert learning.discount_bucket(None) == "unknown"
