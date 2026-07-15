"""Covers the daily AI budget manager (listener/budget.py): priority
classification, budget-gated spending (including the end-of-day reserve),
adaptive validation-probability scaling (listener/learning.py's
validation_multiplier), and the Product Family verdict cache short-circuit
in listener/analyzer.py. Zero real Gemini/Groq calls -- every path that
would reach a real provider is either never exercised (family cache /
budget skip / rule) or explicitly mocked; conftest.py's
block_real_ai_providers raises if anything slips through.
"""

from __future__ import annotations

import random
from datetime import date
from unittest.mock import AsyncMock, patch

import pytest

import database
from listener import budget, family, learning
from listener.analyzer import DealVerdict, analyze_deal
from listener.parser import ParsedDeal


def _today() -> str:
    return date.today().isoformat()


def _seed_calls_used(n: int) -> None:
    for _ in range(n):
        database.increment_gemini_quota_count(_today())


def _make_deal(asin="B0BUDGET01", price=500.0, discount=15, title="Anker Power Bank") -> ParsedDeal:
    return ParsedDeal(
        asin=asin, title=title, price=price, discount_percent=discount,
        channel_name="test_channel", raw_text=title, url=f"https://www.amazon.eg/dp/{asin}",
    )


# --- Priority classification -----------------------------------------------


def test_priority_1_for_high_discount():
    assert budget.classify_priority(
        discount_percent=45, is_new_family=False, is_unknown_brand=False, is_new_category=False,
        is_new_lowest_family_price=False, rule_confidence=0.99,
    ) == 1


def test_priority_1_for_new_family():
    assert budget.classify_priority(
        discount_percent=5, is_new_family=True, is_unknown_brand=False, is_new_category=False,
        is_new_lowest_family_price=False, rule_confidence=0.99,
    ) == 1


def test_priority_1_for_low_rule_confidence():
    assert budget.classify_priority(
        discount_percent=5, is_new_family=False, is_unknown_brand=False, is_new_category=False,
        is_new_lowest_family_price=False, rule_confidence=0.50,
    ) == 1


def test_priority_2_for_moderate_discount():
    assert budget.classify_priority(
        discount_percent=25, is_new_family=False, is_unknown_brand=False, is_new_category=False,
        is_new_lowest_family_price=False, rule_confidence=0.99,
    ) == 2


def test_priority_3_for_low_discount_well_understood_rule():
    assert budget.classify_priority(
        discount_percent=8, is_new_family=False, is_unknown_brand=False, is_new_category=False,
        is_new_lowest_family_price=False, rule_confidence=0.97,
    ) == 3


# --- Budget gating / end-of-day reserve ------------------------------------


def test_priority_1_always_spends_even_at_zero_remaining():
    _seed_calls_used(10_000)  # far past any plausible daily budget
    snapshot = budget.get_snapshot()
    assert snapshot.remaining == 0
    assert budget.should_spend_ai_call(1, snapshot) is True


def test_priority_3_skipped_when_budget_tight():
    """Acceptance test 3 + 7 (end-of-day reserve preserved): once remaining
    is down near the reserve floor, Priority 3 must not spend, while
    Priority 1 still can.
    """
    from config import DAILY_ANALYSIS_CAP

    # Leave just over the reserve floor (10% of budget) remaining.
    reserve_floor = int(DAILY_ANALYSIS_CAP * 0.10)
    _seed_calls_used(DAILY_ANALYSIS_CAP - reserve_floor - 1)
    snapshot = budget.get_snapshot()
    assert snapshot.remaining <= reserve_floor + 5

    assert budget.should_spend_ai_call(3, snapshot) is False
    assert budget.should_spend_ai_call(1, snapshot) is True


def test_heavy_morning_traffic_still_leaves_budget_for_evening():
    """Acceptance test 1: even after a big early-day burst of Priority-1
    spending, remaining budget above the reserve floor is still correctly
    tracked and available -- the reserve itself is what protects the
    evening, and it only shrinks when *actually* spent, not merely
    projected.
    """
    from config import DAILY_ANALYSIS_CAP

    _seed_calls_used(DAILY_ANALYSIS_CAP // 2)
    snapshot = budget.get_snapshot()
    assert snapshot.remaining == DAILY_ANALYSIS_CAP - DAILY_ANALYSIS_CAP // 2
    assert snapshot.remaining > snapshot.reserve_floor
    assert budget.should_spend_ai_call(2, snapshot) is True


# --- Adaptive confidence / validation multiplier ----------------------------


def test_validation_multiplier_lower_when_budget_tight():
    from config import DAILY_ANALYSIS_CAP

    healthy = budget.get_snapshot()
    assert healthy.validation_multiplier == pytest.approx(1.0, abs=0.01)

    _seed_calls_used(DAILY_ANALYSIS_CAP - int(DAILY_ANALYSIS_CAP * 0.10))
    tight = budget.get_snapshot()
    assert tight.validation_multiplier < healthy.validation_multiplier
    assert tight.confidence_threshold_display < healthy.confidence_threshold_display


def test_learning_decide_validation_multiplier_reduces_ai_usage(monkeypatch):
    """Acceptance test 4: a lower validation_multiplier must strictly reduce
    how often an already-confident rule still gets validated with a real AI
    call, for the exact same random draw.
    """
    from listener.learning import Decision, RuleMatch

    # Confidence 0.80 falls in the [0.70, 0.85) band, whose base AI
    # validation probability is a fixed 0.50 (see
    # learning._confidence_band_ai_probability) -- distinct from the
    # RULE_VALIDATION_RATE_MEDIUM/HIGH bands used at higher confidence.
    match = RuleMatch(rule_type="brand", key="Anker", predicted_quality="good", confidence=0.80, sample_count=20)
    monkeypatch.setattr("listener.learning._candidate_rules", lambda *a, **kw: [match])
    monkeypatch.setattr("listener.learning.is_outlier", lambda *a, **kw: False)
    monkeypatch.setattr(random, "random", lambda: 0.15)  # < 0.50 (full) but > 0.50*0.05 (scaled)

    full = learning.decide("Anker", "accessory", 500.0, 10, validation_multiplier=1.0)
    scaled = learning.decide("Anker", "accessory", 500.0, 10, validation_multiplier=0.05)

    assert full.kind == "validate"
    assert scaled.kind == "rule"  # same random draw, but scaled probability no longer clears it


# --- Family verdict cache short-circuits a real AI call (acceptance 5) -----


@pytest.mark.asyncio
async def test_family_cache_hit_avoids_real_ai_call():
    now = database.get_connection  # sanity import check, unused
    family_id = "fam_budget_test"
    database.create_product_family(
        family_id=family_id, brand=None, normalized_title="widget", category=None,
        deterministic_key="key123", anchor_asin="B0ANCHORX1", lowest_price=500.0,
        highest_discount_percent=15, first_seen="2026-01-01T00:00:00",
        last_seen="2026-01-01T00:00:00", best_variant_label="Standard",
    )
    from datetime import datetime
    database.record_family_verdict(
        family_id=family_id, quality="good", reason="cached reason", suggested_target=480,
        category="accessory", provider="gemini", price=500.0, discount_percent=15,
        variant_keys_json="[]", decided_at=datetime.now().isoformat(),
    )

    deal = _make_deal(asin="B0VARIANT02", price=502.0, discount=15, title="Unbranded Widget")
    verdict = await analyze_deal(deal, price_history=None, family_id=family_id, variant={})

    assert verdict is not None
    assert verdict.provider == "family_cache"
    assert verdict.deal_quality == "good"


@pytest.mark.asyncio
async def test_family_cache_miss_falls_through_to_real_ai_when_price_moved_a_lot():
    family_id = "fam_budget_test2"
    database.create_product_family(
        family_id=family_id, brand=None, normalized_title="widget2", category=None,
        deterministic_key="key456", anchor_asin="B0ANCHORX2", lowest_price=500.0,
        highest_discount_percent=15, first_seen="2026-01-01T00:00:00",
        last_seen="2026-01-01T00:00:00", best_variant_label="Standard",
    )
    from datetime import datetime
    database.record_family_verdict(
        family_id=family_id, quality="good", reason="cached reason", suggested_target=480,
        category="accessory", provider="gemini", price=500.0, discount_percent=15,
        variant_keys_json="[]", decided_at=datetime.now().isoformat(),
    )

    fresh_verdict = DealVerdict(deal_quality="great", reason="fresh", suggested_target=300, category="accessory", provider="gemini")
    deal = _make_deal(asin="B0VARIANT03", price=350.0, discount=15, title="Unbranded Widget")  # 30% cheaper -- well past the threshold

    from listener.ai_providers import get_manager
    manager = get_manager()
    with patch.object(manager.gemini, "generate", new=AsyncMock(
        return_value='{"deal_quality": "great", "reason": "fresh", "suggested_target": 300, "category": "accessory"}'
    )):
        verdict = await analyze_deal(deal, price_history=None, family_id=family_id, variant={})

    assert verdict is not None
    assert verdict.provider == "gemini"  # cache correctly rejected -- real call was made
