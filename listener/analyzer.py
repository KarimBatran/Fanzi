"""Public entry point for deal-quality analysis. Everything else in the app
calls analyze_deal()/meets_min_quality() exactly as before — the actual
Gemini-then-Groq provider selection, retries, and circuit breaking now live
in listener.ai_providers, and this module never leaks which provider
answered into any caller's control flow (only DealVerdict.provider, an
informational field callers are free to ignore).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import date, datetime

import database
from config import MIN_DISCOUNT_FOR_ANALYSIS
from listener import learning
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


async def analyze_deal(
    deal: ParsedDeal, price_history: float | None, *, timing=None,
) -> DealVerdict | None:
    """Returns a DealVerdict, or None if analysis couldn't be completed by
    either provider — callers fall back to forwarding the raw deal with an
    "unavailable" verdict rather than dropping it. A low-discount post gets a
    synthetic "average" verdict instead of ever calling an AI provider — a
    neutral, non-biased quality assessment that still goes through the exact
    same MIN_DEAL_QUALITY filter as a real verdict would.

    Cheap checks (no price, low discount) run before any provider is touched,
    exactly as before this module started delegating to AIProviderManager.

    `timing` (a listener.timing.DealTiming, optional, keyword-only) records
    stage durations for the per-deal timing summary — the required
    (deal, price_history) positional signature and return value are
    unchanged, so this is purely additive instrumentation.
    """
    start = time.monotonic()
    try:
        return await _analyze_deal(deal, price_history, timing=timing)
    finally:
        logger.debug("End-to-end analysis duration: %.0f ms", (time.monotonic() - start) * 1000)


async def _analyze_deal(deal: ParsedDeal, price_history: float | None, *, timing=None) -> DealVerdict | None:
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

    rule_lookup_start = time.monotonic()
    brand = learning.extract_brand(deal.title)
    guessed_category = learning.guess_category(deal.title)
    decision = learning.decide(brand, guessed_category, deal.price, deal.discount_percent)
    if timing:
        timing.record("rule_lookup", (time.monotonic() - rule_lookup_start) * 1000)
    stat_date = date.today().isoformat()

    if decision.kind == "rule":
        database.record_rule_hit(stat_date)
        database.record_ai_call_saved(stat_date)
        rule = decision.rule
        target_multiplier = 0.95 if rule.predicted_quality in ("good", "great") else 1.0
        logger.info(
            "learned rule fired (%s=%s, confidence=%.0f%%) — AI call saved",
            rule.rule_type, rule.key, rule.confidence * 100,
        )
        return DealVerdict(
            deal_quality=rule.predicted_quality,
            reason=learning.format_explanation(rule, brand, guessed_category),
            suggested_target=int(deal.price * target_multiplier),
            category=guessed_category or "other",
            provider="learned",
        )

    if decision.kind == "validate":
        database.record_validation_call(stat_date)
        logger.info(
            "validating learned rule (%s=%s, confidence=%.0f%%) with a real AI call",
            decision.rule.rule_type, decision.rule.key, decision.rule.confidence * 100,
        )
    elif decision.had_candidate:
        database.record_rule_miss(stat_date)

    manager = get_manager()
    verdict = await manager.get_verdict(
        title=deal.title,
        price=deal.price,
        discount_percent=deal.discount_percent,
        channel_name=deal.channel_name,
        price_history=price_history,
        brand=brand,
        category_hint=guessed_category,
        timing=timing,
    )
    if verdict is None:
        return None

    if decision.kind == "validate":
        now = datetime.now()
        learning.mark_validated(decision.rule.rule_type, decision.rule.key, now)
        if verdict.deal_quality == decision.rule.predicted_quality:
            database.record_validation_agreement(stat_date)
        else:
            database.record_validation_disagreement(stat_date)
            logger.info(
                "validation disagreement: rule predicted %s, AI (%s) said %s",
                decision.rule.predicted_quality, verdict.provider, verdict.deal_quality,
            )

    # Every successful real AI verdict becomes training data — fire-and-forget
    # so this never delays forwarding the deal.
    enqueue_start = time.monotonic()
    learning.spawn_learning_task(
        asin=deal.asin, provider=verdict.provider, brand=brand, category=verdict.category,
        title=deal.title, price=deal.price, discount_percent=deal.discount_percent,
        deal_quality=verdict.deal_quality, reason=verdict.reason,
        suggested_target=verdict.suggested_target, channel=deal.channel_name,
    )
    if timing:
        timing.record("learning_enqueue", (time.monotonic() - enqueue_start) * 1000)

    return DealVerdict(
        deal_quality=verdict.deal_quality,
        reason=verdict.reason,
        suggested_target=verdict.suggested_target,
        category=verdict.category,
        provider=verdict.provider,
    )
