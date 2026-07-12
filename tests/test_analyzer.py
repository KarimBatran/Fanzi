"""Covers the Gemini quota-aware skip logic in listener/analyzer.py: no
Gemini call for a missing/invalid price, a low-discount post, or when the
daily quota is exhausted; a real (mocked) Gemini response is parsed
correctly when none of those pre-checks apply.
"""

from __future__ import annotations

from datetime import date
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import database
import listener.analyzer as analyzer_module
from listener.analyzer import DealVerdict, QuotaGuard, analyze_deal
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


class _FakeResponse:
    def __init__(self, text: str) -> None:
        self.text = text


def _fake_client(response=None) -> MagicMock:
    client = MagicMock()
    client.aio.models.generate_content = AsyncMock(return_value=response)
    return client


@pytest.mark.asyncio
async def test_no_price_skips_gemini():
    deal = _make_deal(price=0)
    fake_client = _fake_client()
    with patch.object(analyzer_module, "_get_client", return_value=fake_client):
        verdict = await analyze_deal(deal, None)
        fake_client.aio.models.generate_content.assert_not_called()
    assert verdict is None


@pytest.mark.asyncio
async def test_low_discount_skips_gemini():
    """Low-discount posts get a synthetic *average* verdict (not "skip") so
    skipping Gemini doesn't automatically suppress the deal — it flows
    through the normal MIN_DEAL_QUALITY filter like any other verdict.
    """
    deal = _make_deal(discount_percent=5)
    fake_client = _fake_client()
    with patch.object(analyzer_module, "_get_client", return_value=fake_client):
        verdict = await analyze_deal(deal, None)
        fake_client.aio.models.generate_content.assert_not_called()
    assert verdict is not None
    assert verdict.deal_quality == "average"


@pytest.mark.asyncio
async def test_daily_quota_reached_skips_gemini():
    today = date.today().isoformat()
    with database.get_connection() as conn:
        conn.execute(
            "INSERT INTO gemini_quota (quota_date, call_count) VALUES (?, ?) "
            "ON CONFLICT(quota_date) DO UPDATE SET call_count = excluded.call_count",
            (today, analyzer_module._quota_guard.daily_cap),
        )

    deal = _make_deal()
    fake_client = _fake_client()
    with patch.object(analyzer_module, "_get_client", return_value=fake_client):
        verdict = await analyze_deal(deal, None)
        fake_client.aio.models.generate_content.assert_not_called()
    assert verdict is None


@pytest.mark.asyncio
async def test_gemini_response_parsed_into_verdict():
    deal = _make_deal()
    fake_json = (
        '{"deal_quality": "great", "reason": "43% off a well-known brand appliance.", '
        '"suggested_target": 4800, "category": "appliance"}'
    )
    fake_client = _fake_client(_FakeResponse(fake_json))
    with patch.object(analyzer_module, "_get_client", return_value=fake_client):
        verdict = await analyze_deal(deal, None)
        fake_client.aio.models.generate_content.assert_called_once()
    assert verdict == DealVerdict(
        deal_quality="great", reason="43% off a well-known brand appliance.", suggested_target=4800, category="appliance"
    )


@pytest.mark.asyncio
async def test_gemini_response_with_markdown_fence_still_parses():
    deal = _make_deal()
    fake_json = (
        '```json\n{"deal_quality": "good", "reason": "Solid discount.", '
        '"suggested_target": 4900, "category": "appliance"}\n```'
    )
    fake_client = _fake_client(_FakeResponse(fake_json))
    with patch.object(analyzer_module, "_get_client", return_value=fake_client):
        verdict = await analyze_deal(deal, None)
    assert verdict is not None
    assert verdict.deal_quality == "good"
    assert verdict.suggested_target == 4900


def test_quota_guard_counts_calls():
    import asyncio

    guard = QuotaGuard(rate_limit_per_min=5, daily_cap=100)

    async def run():
        await guard.acquire()
        await guard.acquire()

    asyncio.run(run())
    assert guard.minute_count() == 2
    assert guard.daily_count() == 2
    assert guard.daily_quota_reached() is False
