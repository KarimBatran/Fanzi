"""Extract ASIN, price, title, and discount % from a deal-channel post.
Posts mix Arabic and English (e.g. "السعر: 499 جنيه", "-7%", "Price: 499 EGP")
and use a mix of link formats — direct amazon.eg links, amzn.eu/amzn.to/a.co
short links, Amazon's own link.amazon shortener, and generic shorteners
(tinyurl.com, bit.ly, shorturl.at, ...).
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from urllib.parse import parse_qs, urlsplit

import httpx

from amazon.parser import extract_asin, normalize_product_url
from config import REDIRECT_CACHE_TTL_SECONDS

logger = logging.getLogger("fanzi.listener.parser")

# Short-TTL cache for resolved link.amazon/generic-shortener redirects — the
# same deal link is often crossposted across multiple channels within
# minutes, and this avoids a redundant real network round-trip for an
# identical URL seen again inside the TTL window. Maps url -> (asin_or_None,
# expiry_monotonic_time). A negative-cache entry (asin=None) is cached too,
# for the same reason (a dead link stays dead for the TTL window).
_redirect_cache: dict[str, tuple[str | None, float]] = {}


def _redirect_cache_get(url: str) -> tuple[bool, str | None]:
    entry = _redirect_cache.get(url)
    if entry is None:
        return False, None
    asin, expiry = entry
    if time.monotonic() >= expiry:
        del _redirect_cache[url]
        return False, None
    return True, asin


def _redirect_cache_set(url: str, asin: str | None) -> None:
    _redirect_cache[url] = (asin, time.monotonic() + REDIRECT_CACHE_TTL_SECONDS)

_URL_RE = re.compile(r"https?://\S+")

# "السعر: 499 جنيه" / "بسعر 7777" / "Price: 499 EGP" / "499 جنية" — comma
# thousands, optional decimals. "سعر"/"السعر" both accepted (with/without
# the definite article), and "جنية" alongside the more common "جنيه" spelling.
#
# Third alternative: "ب37428" / "بـ 480" — the preposition "ب" ("for"/"at")
# directly followed by a number, with no "سعر" and no currency word at all.
# Extremely common phrasing on Egyptian deal channels ("تلاجة ... ب37428")
# that the first two alternatives miss entirely, silently dropping otherwise
# perfectly valid deals at the parse stage. Requires \b before "ب" and a
# digit shortly after it (through any run of tatweel/elongation characters
# and at most one space) so it can't match ordinary words starting with "ب"
# followed by a letter (بسعة, بعد, بدون, ...).
_PRICE_RE = re.compile(
    r"(?:(?:ال)?سعر|price)\s*[:\-]?\s*([\d,]+(?:\.\d+)?)\s*(?:جنيه|جنية|ج\.م|egp|le)?"
    r"|([\d,]+(?:\.\d+)?)\s*(?:جنيه|جنية|ج\.م|egp|le)\b"
    r"|\bب[ـ]*\s?([\d,]+(?:\.\d+)?)\b",
    re.IGNORECASE,
)

# "-7%", "(٧٪ خصم)", "خصم 7%", "7% off"
_DISCOUNT_RE = re.compile(r"[-(]?\s*(\d{1,2})\s*%")

# Also folds the Arabic thousands separator "،" (U+060C) to the ASCII comma
# the price regex's [\d,] character class expects — without this, a price
# like "19،999" (Arabic-formatted thousands) parses as just "19", silently
# dropping the rest of the number once the regex hits the untranslated "،".
_ARABIC_INDIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩،", "0123456789,")

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
    redirect_ms: float = 0.0


@dataclass
class ParseDiagnostics:
    """Optional out-parameter for extract_from_post — set only when it
    returns None, so callers (listener/watcher.py) can distinguish *why* for
    per-channel health metrics without extract_from_post's return type
    changing for existing callers/tests.
    """
    reason: str | None = None  # "no_asin" | "non_amazon_link" | "no_price"


async def extract_from_post(
    text: str, channel_name: str = "", *, diagnostics: ParseDiagnostics | None = None
) -> ParsedDeal | None:
    """Returns a ParsedDeal, or None if no valid Amazon.eg ASIN / price found."""
    if not text:
        return None

    normalized = text.translate(_ARABIC_INDIC_DIGITS)

    urls = _URL_RE.findall(normalized)
    asin: str | None = None
    matched_url = ""
    redirect_ms = 0.0
    async with httpx.AsyncClient(follow_redirects=True, timeout=_REDIRECT_TIMEOUT_SECONDS) as client:
        for url in urls:
            redirect_start = time.perf_counter()
            found = await _extract_asin_from_url(url, client)
            redirect_ms += (time.perf_counter() - redirect_start) * 1000
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
        if diagnostics is not None:
            # Links were present but none resolved to a usable Amazon ASIN
            # (a different site, a dead shortener, ...) vs. no link/bare-ASIN
            # pattern in the post at all.
            diagnostics.reason = "non_amazon_link" if urls else "no_asin"
        return None

    price = _extract_price(normalized)
    if price is None:
        if diagnostics is not None:
            diagnostics.reason = "no_price"
        return None

    return ParsedDeal(
        asin=asin,
        title=_extract_title(normalized),
        price=price,
        discount_percent=_extract_discount(normalized),
        channel_name=channel_name,
        raw_text=text,
        url=matched_url or normalize_product_url(asin),
        redirect_ms=redirect_ms,
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
    hit, cached_asin = _redirect_cache_get(url)
    if hit:
        return cached_asin

    asin = await _resolve_via_redirect_uncached(url, client)
    _redirect_cache_set(url, asin)
    return asin


async def _resolve_via_redirect_uncached(url: str, client: httpx.AsyncClient) -> str | None:
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

    asin = await extract_asin(resolved_url, client)
    if asin:
        return asin

    # Some Amazon promo/redirect pages (e.g. /promotion/psp/...) don't carry
    # the ASIN in the path at all — it's in a `redirectAsin` query param instead.
    query_params = parse_qs(urlsplit(resolved_url).query)
    redirect_asin_values = query_params.get("redirectAsin")
    if redirect_asin_values and _ASIN_RE.match(redirect_asin_values[0]):
        return redirect_asin_values[0].upper()

    return None


def _extract_price(text: str) -> float | None:
    match = _PRICE_RE.search(text)
    if not match:
        return None
    raw = match.group(1) or match.group(2) or match.group(3)
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
