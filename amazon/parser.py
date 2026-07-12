"""ASIN extraction from any amazon.eg URL shape. Single entry point — every
caller (bot.py, scheduler.py, tests) goes through `extract_asin`, so the
extraction rule never gets duplicated.
"""

from __future__ import annotations

import logging
import re

import httpx

logger = logging.getLogger("fanzi.amazon.parser")

_BARE_ASIN_RE = re.compile(r"^[A-Z0-9]{10}$")
_PATH_ASIN_RE = re.compile(r"/(?:dp|gp/product)/([A-Z0-9]{10})")
_SHORT_LINK_HOSTS = ("amzn.eu", "amzn.to", "a.co")


def normalize_product_url(asin: str) -> str:
    return f"https://www.amazon.eg/dp/{asin}"


async def extract_asin(text: str, http_client: httpx.AsyncClient | None = None) -> str | None:
    """Extract a 10-char ASIN from a bare ASIN, a full amazon.eg URL
    (/dp/ASIN, /gp/product/ASIN, with any slug/query trailing), or an
    amzn.eu/amzn.to short link (resolved via a real HTTP redirect follow —
    the only network call this module makes).
    """
    candidate = text.strip()

    if _BARE_ASIN_RE.match(candidate):
        return candidate

    match = _PATH_ASIN_RE.search(candidate)
    if match:
        return match.group(1)

    if any(host in candidate for host in _SHORT_LINK_HOSTS):
        resolved_url = await _resolve_short_link(candidate, http_client)
        if resolved_url is None:
            return None
        match = _PATH_ASIN_RE.search(resolved_url)
        if match:
            return match.group(1)

    return None


async def _resolve_short_link(url: str, http_client: httpx.AsyncClient | None) -> str | None:
    client = http_client or httpx.AsyncClient(follow_redirects=True, timeout=10.0)
    owns_client = http_client is None
    try:
        response = await client.get(url)
        return str(response.url)
    except httpx.HTTPError as exc:
        logger.warning("short-link resolution failed for %s: %s", url, exc)
        return None
    finally:
        if owns_client:
            await client.aclose()
