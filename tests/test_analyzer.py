"""Covers the pre-provider "cheap checks" in listener/analyzer.py: no
Gemini/Groq call for a missing/invalid price or a low-discount post, and that
a real (mocked) verdict from the provider manager is wrapped into
DealVerdict correctly. Provider-selection/retry/circuit-breaker behavior
lives in tests/test_ai_providers.py.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from listener.analyzer import DealVerdict, analyze_deal
from listener.parser import ParsedDeal


def _make_deal(price: float = 5150.0, discount_percent: int | None = 43) -> ParsedDeal:
    return ParsedDeal(
        asin="B0B1B64D1R",
        title="قلاية رقمية سعة 5.8 لتر من بلاك اند ديكر",
        price=price,
        discount_percent=discount_percent,
        channel_name="ba3bou3_deals",
        raw_text="test post",
        url="https://www.amazon.eg/dp/B0B1B64D1R",
    )


@pytest.mark.asyncio
async def test_no_price_skips_ai_providers():
    deal = _make_deal(price=0)
    with patch("listener.analyzer.get_manager") as get_manager:
        verdict = await analyze_deal(deal, None)
        get_manager.assert_not_called()
    assert verdict is None


@pytest.mark.asyncio
async def test_low_discount_skips_ai_providers():
    """Low-discount posts get a synthetic *average* verdict (not "skip") so
    skipping analysis doesn't automatically suppress the deal — it flows
    through the normal MIN_DEAL_QUALITY filter like any other verdict.
    """
    deal = _make_deal(discount_percent=5)
    with patch("listener.analyzer.get_manager") as get_manager:
        verdict = await analyze_deal(deal, None)
        get_manager.assert_not_called()
    assert verdict is not None
    assert verdict.deal_quality == "average"
    assert verdict.provider == "none"


@pytest.mark.asyncio
async def test_verdict_wraps_manager_result_with_provider():
    from listener.ai_providers import AIVerdict

    deal = _make_deal()
    fake_manager = AsyncMock()
    fake_manager.get_verdict = AsyncMock(
        return_value=AIVerdict(
            provider="groq", deal_quality="good", reason="solid deal", suggested_target=4800, category="appliance"
        )
    )
    with patch("listener.analyzer.get_manager", return_value=fake_manager):
        verdict = await analyze_deal(deal, None)

    assert verdict == DealVerdict(
        deal_quality="good", reason="solid deal", suggested_target=4800, category="appliance", provider="groq"
    )


@pytest.mark.asyncio
async def test_both_providers_unavailable_returns_none():
    deal = _make_deal()
    fake_manager = AsyncMock()
    fake_manager.get_verdict = AsyncMock(return_value=None)
    with patch("listener.analyzer.get_manager", return_value=fake_manager):
        verdict = await analyze_deal(deal, None)
    assert verdict is None
