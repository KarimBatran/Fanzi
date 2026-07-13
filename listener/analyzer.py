"""Public entry point for deal-quality analysis. Everything else in the app
calls analyze_deal()/meets_min_quality() exactly as before — the actual
Gemini-then-Groq provider selection, retries, and circuit breaking now live
in listener.ai_providers, and this module never leaks which provider
answered into any caller's control flow (only DealVerdict.provider, an
informational field callers are free to ignore).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from config import MIN_DISCOUNT_FOR_ANALYSIS
from listener.ai_providers import get_manager
from listener.parser import ParsedDeal

logger = logging.getLogger("fanzi.listener.analyzer")

_QUALITY_RANK = {"skip": 0, "average": 1, "good": 2, "great": 3}


@dataclass
class DealVerdict:
    deal_quality: str
    reason: str
    suggested_target: int
    category: str
    provider: str = "none"


def meets_min_quality(deal_quality: str, min_quality: str) -> bool:
    return _QUALITY_RANK.get(deal_quality, 0) >= _QUALITY_RANK.get(min_quality, 2)


async def analyze_deal(deal: ParsedDeal, price_history: float | None) -> DealVerdict | None:
    """Returns a DealVerdict, or None if analysis couldn't be completed by
    either provider — callers fall back to forwarding the raw deal with an
    "unavailable" verdict rather than dropping it. A low-discount post gets a
    synthetic "average" verdict instead of ever calling an AI provider — a
    neutral, non-biased quality assessment that still goes through the exact
    same MIN_DEAL_QUALITY filter as a real verdict would.

    Cheap checks (no price, low discount) run before any provider is touched,
    exactly as before this module started delegating to AIProviderManager.
    """
    if deal.price is None or deal.price <= 0:
        logger.info("skipped analysis (no price)")
        return None

    if deal.discount_percent is not None and deal.discount_percent < MIN_DISCOUNT_FOR_ANALYSIS:
        logger.info("skipped analysis (low discount)")
        return DealVerdict(
            deal_quality="average",
            reason=f"Discount ({deal.discount_percent}%) is below the {MIN_DISCOUNT_FOR_ANALYSIS}% analysis threshold — not sent to an AI provider.",
            suggested_target=int(deal.price * 0.95),
            category="other",
            provider="none",
        )

    manager = get_manager()
    verdict = await manager.get_verdict(
        raw_text=deal.raw_text,
        title=deal.title,
        price=deal.price,
        discount_percent=deal.discount_percent,
        channel_name=deal.channel_name,
        price_history=price_history,
    )
    if verdict is None:
        return None

    return DealVerdict(
        deal_quality=verdict.deal_quality,
        reason=verdict.reason,
        suggested_target=verdict.suggested_target,
        category=verdict.category,
        provider=verdict.provider,
    )
