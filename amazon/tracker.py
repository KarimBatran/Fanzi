"""Price/title fetching from amazon.eg product pages, with retry+backoff
and specific exceptions so the bot can give the user a meaningful error
instead of a generic one.
"""

from __future__ import annotations

import asyncio
import logging
import re

from playwright.async_api import Page
from playwright.async_api import TimeoutError as PlaywrightTimeoutError

from amazon.browser import get_context

logger = logging.getLogger("fanzi.tracker")

MAX_ATTEMPTS = 3  # initial attempt + 2 retries, per spec
BACKOFF_SECONDS = (3, 7)  # delay before retry 1, retry 2 — modest pacing, not hammering the page

_PAGE_LOAD_TIMEOUT_MS = 20_000

_TITLE_SELECTORS = ("#productTitle", "#title")
_PRICE_SELECTORS = (
    ".a-price .a-offscreen",
    "#corePrice_feature_div .a-offscreen",
    "#priceblock_ourprice",
    "#priceblock_dealprice",
)
_BLOCKED_MARKERS = ("enter the characters you see below", "captcha", "robot check")
_PRICE_RE = re.compile(r"[\d,]+(?:\.\d+)?")


class ProductFetchError(Exception):
    """Base class for all product-fetch failures."""


class FetchTimeoutError(ProductFetchError):
    """The product page didn't finish loading within the timeout."""


class PageBlockedError(ProductFetchError):
    """Amazon presented a CAPTCHA/verification challenge instead of the product page."""


class PriceNotFoundError(ProductFetchError):
    """The page loaded but no title/price element could be found."""


async def fetch_product(url: str) -> tuple[str, float]:
    """Returns (title, price) for an amazon.eg product URL. Retries up to
    MAX_ATTEMPTS times with backoff on any ProductFetchError; raises the
    last error if every attempt fails.
    """
    last_error: ProductFetchError | None = None
    for attempt in range(MAX_ATTEMPTS):
        try:
            return await _fetch_once(url)
        except ProductFetchError as exc:
            last_error = exc
            logger.warning("fetch attempt %d/%d failed for %s: %s", attempt + 1, MAX_ATTEMPTS, url, exc)
            if attempt < MAX_ATTEMPTS - 1:
                await asyncio.sleep(BACKOFF_SECONDS[attempt])
    assert last_error is not None
    raise last_error


async def _fetch_once(url: str) -> tuple[str, float]:
    context = await get_context()
    page = await context.new_page()
    try:
        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=_PAGE_LOAD_TIMEOUT_MS)
        except PlaywrightTimeoutError as exc:
            raise FetchTimeoutError(f"timed out loading {url}") from exc

        if await _is_blocked(page):
            raise PageBlockedError("Amazon presented a verification/CAPTCHA challenge")

        title = await _extract_first_text(page, _TITLE_SELECTORS)
        if title is None:
            raise PriceNotFoundError("could not find product title on page")

        price = await _extract_price(page)
        if price is None:
            raise PriceNotFoundError("could not find a price on page")

        return title, price
    finally:
        await page.close()


async def _is_blocked(page: Page) -> bool:
    content = (await page.content()).lower()
    return any(marker in content for marker in _BLOCKED_MARKERS)


async def _extract_first_text(page: Page, selectors: tuple[str, ...]) -> str | None:
    for selector in selectors:
        el = await page.query_selector(selector)
        if el:
            text = (await el.inner_text()).strip()
            if text:
                return text
    return None


async def _extract_price(page: Page) -> float | None:
    for selector in _PRICE_SELECTORS:
        el = await page.query_selector(selector)
        if el:
            price = _parse_price_text((await el.inner_text()).strip())
            if price is not None:
                return price
    return None


def _parse_price_text(text: str) -> float | None:
    """amazon.eg prices are formatted like '6,299.00' (comma thousands
    separator, dot decimal)."""
    match = _PRICE_RE.search(text)
    if not match:
        return None
    try:
        return float(match.group(0).replace(",", ""))
    except ValueError:
        return None


def format_price(value: float) -> str:
    if value == int(value):
        return f"{int(value):,}"
    return f"{value:,.2f}"
