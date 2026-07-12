"""Covers all link formats listener.parser.extract_from_post must handle:
direct amazon.eg links, amzn.eu/amzn.to/a.co short links, link.amazon (both
a directly-usable path and one needing a redirect), a generic shortener
(tinyurl.com), and the bare-ASIN-in-text fallback. No real network calls —
httpx.AsyncClient.get (and .get for amzn.to-style short links) are mocked.
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
async def test_link_amazon_9char_b0_code_follows_redirect():
    """Post 1 — a 9-char B0-prefixed link.amazon code is a truncated/opaque
    short code, not a real ASIN on its own (real ASINs are always 10 chars) —
    it must always be resolved via redirect rather than guessed directly.
    """
    with patch.object(
        httpx.AsyncClient,
        "get",
        new=AsyncMock(return_value=_RedirectedResponse("https://www.amazon.eg/dp/B0REALASIN")),
    ) as mock_get:
        deal = await extract_from_post(POST_1_ADIDAS, "Mego_Reviews")
        mock_get.assert_called_once()
    assert deal is not None
    assert deal.asin == "B0REALASIN"
    assert deal.price == 2515.0


@pytest.mark.asyncio
async def test_link_amazon_needs_redirect():
    """A link.amazon URL whose path isn't a plain alnum code falls back to a redirect."""
    with patch.object(
        httpx.AsyncClient,
        "get",
        new=AsyncMock(return_value=_RedirectedResponse("https://www.amazon.eg/dp/B0ZZZZZZZZ")),
    ):
        text = "السعر: 999 جنيه\nhttps://link.amazon/re-direct?x=1"
        deal = await extract_from_post(text, "test_channel")
    assert deal is not None
    assert deal.asin == "B0ZZZZZZZZ"
    assert deal.price == 999.0


@pytest.mark.asyncio
async def test_link_amazon_redirect_asin_in_query_param():
    """Some Amazon promo/redirect pages (e.g. /promotion/psp/...) carry the
    real ASIN in a `redirectAsin` query param instead of the URL path.
    """
    with patch.object(
        httpx.AsyncClient,
        "get",
        new=AsyncMock(
            return_value=_RedirectedResponse(
                "https://www.amazon.eg/promotion/psp/PROMO123?redirectAsin=B07RJNSTZ&tag=egypt06-21"
            )
        ),
    ):
        text = "السعر: 250 جنيه\nhttps://link.amazon/re-direct?x=1"
        deal = await extract_from_post(text, "test_channel")
    assert deal is not None
    assert deal.asin == "B07RJNSTZ"
    assert deal.price == 250.0


@pytest.mark.asyncio
async def test_generic_shortener_tinyurl():
    """Post 2 — a tinyurl.com link resolved via a mocked GET redirect."""
    with patch.object(
        httpx.AsyncClient,
        "get",
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


@pytest.mark.asyncio
async def test_price_bare_preposition_no_currency_word():
    """"ب<number>" with no "سعر" and no currency word at all (e.g. "ب37428")
    is extremely common on real deal channels and was previously dropped
    entirely — a live-channel diagnosis found real Amazon deals (valid ASIN
    links) on LQoffers/Mego_Reviews silently failing at the parse stage
    purely because of this phrasing.
    """
    text = "تلاجة من Ocean بسعة 625 لتر ب37428\nhttps://www.amazon.eg/dp/B0FRIDGE01"
    deal = await extract_from_post(text, "LQoffers")
    assert deal is not None
    assert deal.asin == "B0FRIDGE01"
    assert deal.price == 37428.0


@pytest.mark.asyncio
async def test_price_bare_preposition_with_tatweel_and_space():
    """Same "ب<number>" phrasing but with elongation characters (تطويل) and
    a space before the digits: "بــــ 129".
    """
    text = "حلاوة طحينية الرشيدي الميزان 900جم بــــ 129\nhttps://www.amazon.eg/dp/B0HALAWA01"
    deal = await extract_from_post(text, "Mego_Reviews")
    assert deal is not None
    assert deal.asin == "B0HALAWA01"
    assert deal.price == 129.0


@pytest.mark.asyncio
async def test_price_bare_preposition_does_not_false_positive_on_ordinary_words():
    """"ب" followed by a letter (بسعة, بدون) must not be mistaken for a price
    — only "ب" directly followed by a digit qualifies.
    """
    text = "خصم 50% بدون حد اقصى على جميع احذية Anta - Skechers - Umbro"
    deal = await extract_from_post(text, "LQoffers")
    assert deal is None
