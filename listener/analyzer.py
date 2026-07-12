"""Gemini API call to assess deal quality and produce a verdict, with
quota-aware request management (rate limiting, daily budget, and cheap
pre-Gemini skips) so the app stays comfortably within the Gemini free tier.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass
from datetime import date

from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types

import database
from config import (
    DAILY_ANALYSIS_CAP,
    GEMINI_API_KEY,
    MIN_DISCOUNT_FOR_ANALYSIS,
    RATE_LIMIT_PER_MIN,
)
from listener.parser import ParsedDeal

logger = logging.getLogger("fanzi.listener.analyzer")

_MODEL = "gemini-flash-latest"

_SYSTEM_PROMPT = (
    "You are a deal analyst for Amazon Egypt. You receive a product deal post "
    "and optional price history. Return ONLY a JSON object with these exact "
    "keys: deal_quality (one of: great/good/average/skip), reason (one "
    "sentence, English, max 15 words), suggested_target (integer EGP, 5-10% "
    "below current price for good deals, same as current for average), "
    "category (phone/headphones/laptop/accessory/cable/appliance/other). "
    "Judge based on: discount percentage mentioned, product category "
    "prestige, price vs history if available. Be conservative — most deals "
    "are average."
)

_QUALITY_RANK = {"skip": 0, "average": 1, "good": 2, "great": 3}

_JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)


@dataclass
class DealVerdict:
    deal_quality: str
    reason: str
    suggested_target: int
    category: str


def meets_min_quality(deal_quality: str, min_quality: str) -> bool:
    return _QUALITY_RANK.get(deal_quality, 0) >= _QUALITY_RANK.get(min_quality, 2)


class QuotaGuard:
    """Rate-limits and daily-budgets Gemini calls. Rate limiting waits for a
    slot rather than dropping the request; the daily cap is a hard stop —
    callers must check `daily_quota_reached()` themselves and skip Gemini
    entirely once it's True (this class does not enforce the cap inside
    `acquire()`, so a checked-then-acquire race can't spin forever waiting
    for a slot that will never come).

    The per-minute count is in-memory only (a 60s window losing its history
    on restart is harmless). The daily count is persisted in the
    gemini_quota table, keyed by date, so restarts don't reset it and it
    naturally resets at local midnight (a new day is just a fresh row).
    """

    def __init__(self, rate_limit_per_min: int, daily_cap: int) -> None:
        self.rate_limit_per_min = rate_limit_per_min
        self.daily_cap = daily_cap
        self._call_times: list[float] = []
        self._lock = asyncio.Lock()

    @staticmethod
    def _today() -> str:
        return date.today().isoformat()

    def daily_count(self) -> int:
        return database.get_gemini_quota_count(self._today())

    def minute_count(self) -> int:
        cutoff = time.monotonic() - 60
        return sum(1 for t in self._call_times if t > cutoff)

    def daily_quota_reached(self) -> bool:
        return self.daily_count() >= self.daily_cap

    async def acquire(self) -> None:
        """Waits until a call slot is available under the per-minute rate
        limit, then reserves it and increments the persisted daily counter.
        """
        async with self._lock:
            while True:
                cutoff = time.monotonic() - 60
                self._call_times = [t for t in self._call_times if t > cutoff]
                if len(self._call_times) < self.rate_limit_per_min:
                    break
                wait_seconds = 60 - (time.monotonic() - min(self._call_times)) + 0.05
                await asyncio.sleep(max(wait_seconds, 0.05))

            self._call_times.append(time.monotonic())
            database.increment_gemini_quota_count(self._today())


_quota_guard = QuotaGuard(RATE_LIMIT_PER_MIN, DAILY_ANALYSIS_CAP)
_client: genai.Client | None = None


def _get_client() -> genai.Client:
    """Lazy singleton — constructing genai.Client() raises immediately if
    GEMINI_API_KEY is empty, so this must not run at import time (that would
    crash the whole app on startup instead of degrading to "unavailable").
    """
    global _client
    if _client is None:
        _client = genai.Client(api_key=GEMINI_API_KEY)
    return _client


def get_quota_status() -> dict:
    """Snapshot for /status — {daily_count, daily_cap, remaining, minute_count}."""
    daily_count = _quota_guard.daily_count()
    return {
        "daily_count": daily_count,
        "daily_cap": _quota_guard.daily_cap,
        "remaining": max(0, _quota_guard.daily_cap - daily_count),
        "minute_count": _quota_guard.minute_count(),
    }


def _strip_json_fence(text: str) -> str:
    return _JSON_FENCE_RE.sub("", text.strip()).strip()


async def analyze_deal(deal: ParsedDeal, price_history: float | None) -> DealVerdict | None:
    """Returns a DealVerdict, or None if analysis couldn't be completed (API
    failure, timeout, or daily quota exhausted) — callers fall back to
    forwarding the raw deal with an "unavailable" verdict rather than
    dropping it. A low-discount post gets a synthetic "average" verdict
    instead — a neutral, non-biased quality assessment (not a hard-coded
    "skip") that goes through the exact same MIN_DEAL_QUALITY filter as a
    real Gemini verdict would, so skipping Gemini never *automatically*
    suppresses the deal on its own.
    """
    if deal.price is None or deal.price <= 0:
        logger.info("skipped analysis (no price)")
        return None

    if deal.discount_percent is not None and deal.discount_percent < MIN_DISCOUNT_FOR_ANALYSIS:
        logger.info("skipped analysis (low discount)")
        return DealVerdict(
            deal_quality="average",
            reason=f"Discount ({deal.discount_percent}%) is below the {MIN_DISCOUNT_FOR_ANALYSIS}% analysis threshold — not sent to Gemini.",
            suggested_target=int(deal.price * 0.95),
            category="other",
        )

    if _quota_guard.daily_quota_reached():
        logger.warning("daily quota reached, forwarding without analysis")
        return None

    await _quota_guard.acquire()

    history_line = (
        f"Previously tracked price: {price_history:g} EGP." if price_history is not None else "No price history available."
    )
    user_content = (
        f"{deal.raw_text}\n\n"
        f"Parsed: title={deal.title!r}, price={deal.price:g} EGP, "
        f"discount={deal.discount_percent}%, channel={deal.channel_name}. "
        f"{history_line}"
    )

    try:
        client = _get_client()
        response = await client.aio.models.generate_content(
            model=_MODEL,
            contents=user_content,
            config=genai_types.GenerateContentConfig(
                system_instruction=_SYSTEM_PROMPT,
                response_mime_type="application/json",
                max_output_tokens=256,
                # This is a simple structured-extraction task — extended
                # thinking just burns quota/tokens for no benefit (it was
                # consuming the entire budget before any visible output).
                thinking_config=genai_types.ThinkingConfig(thinking_budget=0),
            ),
        )
    except genai_errors.APIError as exc:
        logger.error(
            "deal analysis failed for ASIN %s: model=%s status=%s message=%s",
            deal.asin,
            _MODEL,
            exc.status,
            exc.message,
        )
        return None
    except Exception:
        logger.exception("deal analysis failed for ASIN %s (model=%s)", deal.asin, _MODEL)
        return None

    text = response.text
    if not text:
        logger.warning("no content in analysis response for ASIN %s", deal.asin)
        return None

    try:
        data = json.loads(_strip_json_fence(text))
        return DealVerdict(
            deal_quality=str(data["deal_quality"]).strip().lower(),
            reason=str(data["reason"]).strip(),
            suggested_target=int(float(data["suggested_target"])),
            category=str(data["category"]).strip().lower(),
        )
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        logger.exception("malformed analysis response for ASIN %s: %s", deal.asin, text)
        return None
