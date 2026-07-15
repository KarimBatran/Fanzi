"""Covers amazon.tracker price extraction after the buy-box-scoping fix.
Regression for a real production bug: on amazon.eg a product with no buy box
of its own renders sponsored-product carousels whose `.a-price .a-offscreen`
elements appear first in the DOM and belong to DIFFERENT ASINs, so the old
page-wide selector reported a rotating ad's price (the "price keeps
changing / doesn't match when I click" reports). No Playwright/browser is
launched -- _extract_price is driven by a fake page mapping selectors to
element text, exactly as query_selector_all would return.
"""

from __future__ import annotations

import pytest

from amazon.tracker import _extract_price, _parse_price_text


class _FakeElement:
    def __init__(self, text: str) -> None:
        self._text = text

    async def inner_text(self) -> str:
        return self._text


class _FakePage:
    def __init__(self, mapping: dict[str, list[str]]) -> None:
        self._mapping = mapping

    async def query_selector_all(self, selector: str):
        return [_FakeElement(t) for t in self._mapping.get(selector, [])]


def test_parse_price_handles_nbsp_and_thousands():
    assert _parse_price_text("EGP\xa02,170.00") == 2170.0
    assert _parse_price_text("EGP238.90") == 238.90
    assert _parse_price_text("EGP\xa020,699.00") == 20699.0


def test_parse_price_none_for_junk():
    assert _parse_price_text("") is None
    assert _parse_price_text("Currently unavailable") is None


@pytest.mark.asyncio
async def test_extract_price_from_core_price_container():
    page = _FakePage(
        {
            "#corePrice_feature_div .apex-pricetopay-value .a-offscreen": ["EGP238.90"],
        }
    )
    assert await _extract_price(page) == 238.90


@pytest.mark.asyncio
async def test_extract_price_skips_empty_offscreen_spans():
    # First matching container has an empty a-offscreen (the empty
    # .priceToPay span seen live on amazon.eg); must fall through to the
    # first element that actually parses, not return None or "".
    page = _FakePage(
        {
            "#corePrice_feature_div .priceToPay .a-offscreen": [""],
            "#corePrice_feature_div .a-offscreen": ["", "EGP950.00"],
        }
    )
    assert await _extract_price(page) == 950.0


@pytest.mark.asyncio
async def test_extract_price_ignores_sponsored_carousel_prices():
    """The core regression: a product with NO buy box. The only prices on
    the page are sponsored-carousel/compare-similar prices for OTHER ASINs,
    exposed only via the page-wide `.a-price .a-offscreen` selector the new
    code no longer consults. Extraction must return None -> product reported
    unavailable, never a stranger's rotating ad price.
    """
    page = _FakePage(
        {
            ".a-price .a-offscreen": ["EGP1,949.98", "EGP3,999.99", "EGP2,399.00"],
            # No buy-box / core-price container exists for the tracked item.
        }
    )
    assert await _extract_price(page) is None


@pytest.mark.asyncio
async def test_extract_price_prefers_price_to_pay_over_list_price():
    # If a container ever holds both a struck-through list price and the
    # price to pay, the pricetopay selector (queried first) must win.
    page = _FakePage(
        {
            "#corePrice_feature_div .apex-pricetopay-value .a-offscreen": ["EGP180.00"],
            "#corePrice_feature_div .a-offscreen": ["EGP250.00", "EGP180.00"],
        }
    )
    assert await _extract_price(page) == 180.0
