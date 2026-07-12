"""Extract ASIN, price, title, and discount % from a deal-channel post.
Posts mix Arabic and English (e.g. "السعر: 499 جنيه", "-7%", "Price: 499 EGP")
and use a mix of link formats — direct amazon.eg links, amzn.eu/amzn.to/a.co
short links, Amazon's own link.amazon shortener, and generic shorteners
(tinyurl.com, bit.ly, shorturl.at, ...).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from urllib.parse import urlsplit

import httpx

from amazon.parser import extract_asin, normalize_product_url

logger = logging.getLogger("fanzi.listener.parser")

_URL_RE = re.compile(r"https?://\S+")

# "السعر: 499 جنيه" / "بسعر 7777" / "Price: 499 EGP" / "499 جنية" — comma
# thousands, optional decimals. "سعر"/"السعر" both accepted (with/without
# the definite article), and "جنية" alongside the more common "جنيه" spelling.
_PRICE_RE = re.compile(
    r"(?:(?:ال)?سعر|price)\s*[:\-]?\s*([\d,]+(?:\.\d+)?)\s*(?:جنيه|جنية|ج\.م|egp|le)?"
    r"|([\d,]+(?:\.\d+)?)\s*(?:جنيه|جنية|ج\.م|egp|le)\b",
    re.IGNORECASE,
)

# "-7%", "(٧٪ خصم)", "خصم 7%", "7% off"
_DISCOUNT_RE = re.compile(r"[-(]?\s*(\d{1,2})\s*%")

_ARABIC_INDIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")

_EMOJI_PREFIX_RE = re.compile(r"^[\W_]+", re.UNICODE)

# link.amazon path segments are sometimes the ASIN itself and sometimes an
# opaque short code of a different length — accept any alnum-only segment in
# a plausible ASIN-ish length range and use it directly; anything else (query
# strings, slashes, etc.) falls through to a redirect follow.
_ASIN_RE = re.compile(r"^[A-Za-z0-9]{6,12}$")

# Last-resort scan of the raw post text for something ASIN-shaped (Amazon
# ASINs issued in recent years start with "B0"), used only when no URL in
# the post could be resolved to an ASIN.
_BARE_ASIN_IN_TEXT_RE = re.compile(r"\bB0[A-Z0-9]{7,8}\b")

_REDIRECT_TIMEOUT_SECONDS = 5.0


@dataclass
class ParsedDeal:
    asin: str
    title: str
    price: float
    discount_percent: int | None
    channel_name: str
    raw_text: str
    url: str


async def extract_from_post(text: str, channel_name: str = "") -> ParsedDeal | None:
    """Returns a ParsedDeal, or None if no valid Amazon.eg ASIN / price found."""
    if not text:
        return None

    normalized = text.translate(_ARABIC_INDIC_DIGITS)

    urls = _URL_RE.findall(normalized)
    asin: str | None = None
    matched_url = ""
    async with httpx.AsyncClient(follow_redirects=True, timeout=_REDIRECT_TIMEOUT_SECONDS) as client:
        for url in urls:
            found = await _extract_asin_from_url(url, client)
            if found:
                asin = found
                matched_url = url
                break

    if asin is None:
        # Scan prose only — a URL that already failed extraction/redirect
        # resolution must not be re-matched here just because its path
        # segment happens to look ASIN-shaped (e.g. a dead link.amazon code).
        text_without_urls = _URL_RE.sub(" ", normalized)
        text_match = _BARE_ASIN_IN_TEXT_RE.search(text_without_urls.upper())
        if text_match:
            asin = text_match.group(0)

    if asin is None:
        return None

    price = _extract_price(normalized)
    if price is None:
        return None

    return ParsedDeal(
        asin=asin,
        title=_extract_title(normalized),
        price=price,
        discount_percent=_extract_discount(normalized),
        channel_name=channel_name,
        raw_text=text,
        url=matched_url or normalize_product_url(asin),
    )


async def _extract_asin_from_url(url: str, client: httpx.AsyncClient) -> str | None:
    """Priority order: direct amazon.eg /dp//gp/product and amzn.eu/amzn.to/a.co
    short links (via the shared amazon.parser.extract_asin), then Amazon's own
    link.amazon shortener, then any other (generic) shortener.
    """
    asin = await extract_asin(url, client)
    if asin:
        return asin

    host = urlsplit(url).netloc.lower()

    if "link.amazon" in host:
        path_segment = urlsplit(url).path.strip("/").split("/")[0] if urlsplit(url).path else ""
        # Real ASINs are always exactly 10 chars. A 9-char code starting with
        # "B0" is a truncated/opaque short code, not a usable ASIN on its own
        # — always resolve those via redirect rather than guessing.
        looks_like_truncated_b0_code = len(path_segment) == 9 and path_segment.upper().startswith("B0")
        if _ASIN_RE.match(path_segment) and not looks_like_truncated_b0_code:
            return path_segment.upper()
        return await _resolve_via_redirect(url, client)

    if "amazon" not in host:
        # Generic shortener (tinyurl.com, bit.ly, shorturl.at, ...).
        return await _resolve_via_redirect(url, client)

    return None


async def _resolve_via_redirect(url: str, client: httpx.AsyncClient) -> str | None:
    try:
        # GET, not HEAD: link.amazon (and possibly other shorteners) reject
        # HEAD requests with a 404 but resolve correctly on GET.
        response = await client.get(url, timeout=_REDIRECT_TIMEOUT_SECONDS)
    except httpx.HTTPError:
        logger.warning("redirect resolution failed for %s", url)
        return None

    resolved_url = str(response.url)
    if "amazon" not in resolved_url.lower():
        logger.warning("redirect for %s did not resolve to an Amazon URL: %s", url, resolved_url)
        return None

    return await extract_asin(resolved_url, client)


def _extract_price(text: str) -> float | None:
    match = _PRICE_RE.search(text)
    if not match:
        return None
    raw = match.group(1) or match.group(2)
    try:
        return float(raw.replace(",", ""))
    except ValueError:
        return None


def _extract_discount(text: str) -> int | None:
    match = _DISCOUNT_RE.search(text)
    if not match:
        return None
    return int(match.group(1))


def _extract_title(text: str) -> str:
    first_line = text.strip().splitlines()[0] if text.strip() else ""
    # Strip a leading run of emoji/punctuation (e.g. "📢 عرض على ...").
    cleaned = _EMOJI_PREFIX_RE.sub("", first_line).strip()
    return cleaned or first_line.strip()
