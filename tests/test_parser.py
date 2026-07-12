"""Covers all link formats listener.parser.extract_from_post must handle:
direct amazon.eg links, amzn.eu/amzn.to/a.co short links, link.amazon (both
a directly-usable path and one needing a redirect), a generic shortener
(tinyurl.com), and the bare-ASIN-in-text fallback. No real network calls —
httpx.AsyncClient.head/get are mocked.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import httpx
import pytest

from listener.parser import extract_from_post

POST_1_ADIDAS = (
    "عرض على كوتشي Adidas Galaxy Step مقاس 45 وتلت بـ2515 جنية\n"
    "https://link.amazon/B0e48L1wU\n"
    "البائع امازون.. شحن مجاني"
)

POST_2_VIZIO = (
    "▪️فيو تلفزيون LED ذكي بدون اطار مقاس 43 بوصة - L43VIEWA425\n"
    "بــــــــــسعر 7777 🔥\n"
    "https://tinyurl.com/NGMM-MEGO"
)


class _RedirectedResponse:
    """Minimal stand-in exposing only the `.url` attribute the resolver reads."""

    def __init__(self, url: str) -> None:
        self.url = url


@pytest.mark.asyncio
async def test_direct_amazon_dp_link():
    text = "السعر: 300 جنيه\nhttps://www.amazon.eg/dp/B0ABCDEFGH"
    deal = await extract_from_post(text, "test_channel")
    assert deal is not None
    assert deal.asin == "B0ABCDEFGH"
    assert deal.price == 300.0


@pytest.mark.asyncio
async def test_amzn_to_short_link():
    with patch.object(
        httpx.AsyncClient,
        "get",
        new=AsyncMock(return_value=_RedirectedResponse("https://www.amazon.eg/dp/B0SHORTLNK")),
    ):
        text = "Price: 199 EGP\nhttps://amzn.to/3xyzABC"
        deal = await extract_from_post(text, "test_channel")
    assert deal is not None
    assert deal.asin == "B0SHORTLNK"
    assert deal.price == 199.0


@pytest.mark.asyncio
async def test_link_amazon_direct_path():
    """Post 1 — the link.amazon path segment is used directly, no redirect."""
    with patch.object(httpx.AsyncClient, "head", new=AsyncMock()) as mock_head:
        deal = await extract_from_post(POST_1_ADIDAS, "Mego_Reviews")
        mock_head.assert_not_called()
    assert deal is not None
    assert deal.asin == "B0E48L1WU"
    assert deal.price == 2515.0


@pytest.mark.asyncio
async def test_link_amazon_needs_redirect():
    """A link.amazon URL whose path isn't a plain alnum code falls back to a redirect."""
    with patch.object(
        httpx.AsyncClient,
        "head",
        new=AsyncMock(return_value=_RedirectedResponse("https://www.amazon.eg/dp/B0ZZZZZZZZ")),
    ):
        text = "السعر: 999 جنيه\nhttps://link.amazon/re-direct?x=1"
        deal = await extract_from_post(text, "test_channel")
    assert deal is not None
    assert deal.asin == "B0ZZZZZZZZ"
    assert deal.price == 999.0


@pytest.mark.asyncio
async def test_generic_shortener_tinyurl():
    """Post 2 — a tinyurl.com link resolved via a mocked HEAD redirect."""
    with patch.object(
        httpx.AsyncClient,
        "head",
        new=AsyncMock(return_value=_RedirectedResponse("https://www.amazon.eg/dp/B0VIZIOTV1")),
    ):
        deal = await extract_from_post(POST_2_VIZIO, "OffersCommunityEG")
    assert deal is not None
    assert deal.asin == "B0VIZIOTV1"
    assert deal.price == 7777.0


@pytest.mark.asyncio
async def test_bare_asin_fallback_no_url():
    """No URL in the post at all — falls back to scanning the raw text."""
    text = "عرض على كوتشي Adidas Galaxy Step B0E48L1WU مقاس 45 وتلت بـ2515 جنية"
    deal = await extract_from_post(text, "test_channel")
    assert deal is not None
    assert deal.asin == "B0E48L1WU"
    assert deal.price == 2515.0
