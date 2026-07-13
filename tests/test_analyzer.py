"""Covers the Gemini quota-aware skip logic in listener/analyzer.py: no
Gemini call for a missing/invalid price, a low-discount post, or when the
daily quota is exhausted; a real (mocked) Gemini response is parsed
correctly when none of those pre-checks apply.
"""

from __future__ import annotations

from datetime import date
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from google.genai import errors as genai_errors

import database
import health
import listener.analyzer as analyzer_module
from listener.analyzer import DealVerdict, QuotaGuard, analyze_deal, get_quota_status
from listener.parser import ParsedDeal


def _api_error(status: str, message: str = "test error") -> genai_errors.APIError:
    return genai_errors.APIError(500, {"status": status, "message": message})


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


def test_unmocked_gemini_call_is_blocked():
    """Proves the tests/conftest.py safety net actually works: any attempt
    to reach the real Gemini API without an explicit patch.object(...,
    "_get_client", ...) fails loudly instead of silently spending real
    quota. This is what guarantees the full suite consumes zero real
    Gemini requests. (analyze_deal itself catches this — like any other
    client error — and forwards without a verdict rather than crashing;
    calling _get_client() directly is what proves the block is in place.)
    """
    with pytest.raises(AssertionError, match="real Gemini API"):
        analyzer_module._get_client()


@pytest.mark.asyncio
async def test_analyze_deal_never_reaches_real_gemini_if_unmocked():
    """analyze_deal degrades to "unavailable" (returns None) rather than
    ever completing a real network call when _get_client isn't mocked —
    the safety net fails closed, not open.
    """
    deal = _make_deal()
    verdict = await analyze_deal(deal, None)
    assert verdict is None


@pytest.mark.asyncio
async def test_resource_exhausted_stops_further_calls():
    """RESOURCE_EXHAUSTED means Google's own quota is spent — the app must
    record that and stop making further doomed requests, but still fall
    back (return None) so the caller keeps forwarding deals.
    """
    deal = _make_deal()
    fake_client = _fake_client()
    fake_client.aio.models.generate_content = AsyncMock(side_effect=_api_error("RESOURCE_EXHAUSTED"))

    assert analyzer_module._quota_guard.external_quota_exhausted() is False
    with patch.object(analyzer_module, "_get_client", return_value=fake_client):
        verdict = await analyze_deal(deal, None)
    assert verdict is None
    assert analyzer_module._quota_guard.external_quota_exhausted() is True

    # A second deal must not even attempt a Gemini call now.
    other_deal = _make_deal(price=999.0)
    fake_client_2 = _fake_client(_FakeResponse('{"deal_quality": "good", "reason": "x", "suggested_target": 900, "category": "other"}'))
    with patch.object(analyzer_module, "_get_client", return_value=fake_client_2):
        verdict2 = await analyze_deal(other_deal, None)
        fake_client_2.aio.models.generate_content.assert_not_called()
    assert verdict2 is None


@pytest.mark.asyncio
async def test_unavailable_retries_with_backoff_then_succeeds():
    """UNAVAILABLE (transient overload) is retried with exponential backoff;
    a later retry succeeding must return the real verdict, not fall back.
    """
    deal = _make_deal()
    success_response = _FakeResponse(
        '{"deal_quality": "great", "reason": "y", "suggested_target": 4800, "category": "appliance"}'
    )
    fake_client = _fake_client()
    fake_client.aio.models.generate_content = AsyncMock(
        side_effect=[_api_error("UNAVAILABLE"), _api_error("UNAVAILABLE"), success_response]
    )

    with patch.object(analyzer_module, "_get_client", return_value=fake_client), patch.object(
        analyzer_module.asyncio, "sleep", new=AsyncMock()
    ) as mock_sleep:
        verdict = await analyze_deal(deal, None)

    assert verdict is not None
    assert verdict.deal_quality == "great"
    assert fake_client.aio.models.generate_content.call_count == 3
    assert mock_sleep.call_count == 2  # one backoff before each of the 2 retries


@pytest.mark.asyncio
async def test_unavailable_exhausts_retries_then_falls_back():
    """If every retry also fails, analyze_deal gives up and returns None
    (the existing forward-without-verdict fallback) rather than retrying
    forever.
    """
    deal = _make_deal()
    fake_client = _fake_client()
    fake_client.aio.models.generate_content = AsyncMock(side_effect=_api_error("UNAVAILABLE"))

    with patch.object(analyzer_module, "_get_client", return_value=fake_client), patch.object(
        analyzer_module.asyncio, "sleep", new=AsyncMock()
    ):
        verdict = await analyze_deal(deal, None)

    assert verdict is None
    # 1 initial attempt + GEMINI_RETRY_COUNT retries.
    assert fake_client.aio.models.generate_content.call_count == analyzer_module.GEMINI_RETRY_COUNT + 1


@pytest.mark.asyncio
async def test_non_retryable_error_fails_immediately_without_retry():
    """Auth failures / invalid requests must not be retried — only
    UNAVAILABLE is treated as transient.
    """
    deal = _make_deal()
    fake_client = _fake_client()
    fake_client.aio.models.generate_content = AsyncMock(side_effect=_api_error("PERMISSION_DENIED"))

    with patch.object(analyzer_module, "_get_client", return_value=fake_client), patch.object(
        analyzer_module.asyncio, "sleep", new=AsyncMock()
    ) as mock_sleep:
        verdict = await analyze_deal(deal, None)

    assert verdict is None
    fake_client.aio.models.generate_content.assert_called_once()
    mock_sleep.assert_not_called()


def test_status_reports_external_quota_state():
    today = date.today().isoformat()
    assert get_quota_status()["external_quota_exhausted"] is False
    assert "External quota: AVAILABLE" in health.format_status_message()

    database.mark_gemini_external_quota_exhausted(today)

    assert get_quota_status()["external_quota_exhausted"] is True
    assert "External quota: EXHAUSTED" in health.format_status_message()


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
