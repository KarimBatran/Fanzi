"""Public entry point for deal-quality analysis. Everything else in the app
calls analyze_deal()/meets_min_quality() exactly as before — the actual
Gemini-then-Groq provider selection, retries, and circuit breaking now live
in listener.ai_providers, and this module never leaks which provider
answered into any caller's control flow (only DealVerdict.provider, an
informational field callers are free to ignore).
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from datetime import date, datetime

import database
from config import MIN_DISCOUNT_FOR_ANALYSIS, SCORE_ENGINE_ENABLED, SCORE_ENGINE_LOG_VERBOSE
from listener import budget, family, learning, scoring
from listener.ai_providers import get_manager
from listener.parser import ParsedDeal

logger = logging.getLogger("fanzi.listener.analyzer")

_QUALITY_RANK = {"skip": 0, "average": 1, "good": 2, "great": 3}

# Keeps fire-and-forget shadow-log tasks alive until done (same pattern as
# listener/learning.py's _background_tasks) -- asyncio only holds weak refs.
_shadow_tasks: set[asyncio.Task] = set()


def _log_shadow_score(
    legacy_kwargs: dict, scored_kwargs: dict, discount_percent: int | None,
    score_result, stat_date: str,
) -> None:
    """Computes and logs the legacy vs scored priority side-by-side for one
    deal (structured, single line) and records the divergence counter --
    this is the shadow-mode signal reviewed before SCORE_ENGINE_ENABLED is
    ever flipped (docs/score_engine_rollout.md). Pure SQLite reads + one
    counter write; never raises into the pipeline.
    """
    try:
        if score_result is None:
            score_result = scoring.compute_value_score(
                discount_percent=discount_percent, **scored_kwargs
            )
        legacy_priority = budget.classify_priority_legacy(
            discount_percent=discount_percent, **legacy_kwargs
        )
        scored_priority = budget.classify_priority_scored(
            discount_percent=discount_percent, score_result=score_result, **scored_kwargs
        )
        family_pctl = (
            f"{score_result.family_percentile:.2f}" if score_result.family_percentile is not None else "none"
        )
        logger.info(
            "score_shadow asin=%s brand=%s category=%s legacy_priority=%d scored_priority=%d "
            "value_score=%.1f brand_reputation=%.2f price_percentile=%.2f family_percentile=%s "
            "category_deviation=%.2f rarity=%.2f",
            scored_kwargs["asin"], scored_kwargs["brand"], scored_kwargs["category"],
            legacy_priority, scored_priority, score_result.total,
            score_result.brand_reputation, score_result.price_percentile, family_pctl,
            score_result.category_deviation, score_result.rarity,
        )
        database.record_shadow_comparison(stat_date, diverged=legacy_priority != scored_priority)
    except Exception:
        logger.exception("shadow score logging failed -- pipeline unaffected")


def _spawn_shadow_score_log(
    legacy_kwargs: dict, scored_kwargs: dict, discount_percent: int | None,
    score_result, stat_date: str,
) -> None:
    """Fire-and-forget, same pattern as learning.spawn_learning_task -- the
    shadow computation runs after the current message's forward path yields,
    never inside it.
    """

    async def _run() -> None:
        _log_shadow_score(legacy_kwargs, scored_kwargs, discount_percent, score_result, stat_date)

    task = asyncio.create_task(_run())
    _shadow_tasks.add(task)
    task.add_done_callback(_shadow_tasks.discard)


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
    family_id: str | None = None, variant: dict | None = None,
    is_new_family: bool = False, is_new_family_low_price: bool = False,
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

    `family_id`/`variant`/`is_new_family`/`is_new_family_low_price`
    (listener/family.py, all optional/keyword-only, default to "no family
    context" for every existing caller/test) feed the daily AI budget
    manager (listener/budget.py): priority classification and the
    family-verdict cache. See that module's docstring for the full
    learned-rule -> family-cache -> Gemini -> Groq -> budget-skip order.
    """
    start = time.monotonic()
    try:
        return await _analyze_deal(
            deal, price_history, timing=timing, family_id=family_id, variant=variant or {},
            is_new_family=is_new_family, is_new_family_low_price=is_new_family_low_price,
        )
    finally:
        logger.debug("End-to-end analysis duration: %.0f ms", (time.monotonic() - start) * 1000)


async def _analyze_deal(
    deal: ParsedDeal, price_history: float | None, *, timing=None,
    family_id: str | None = None, variant: dict | None = None,
    is_new_family: bool = False, is_new_family_low_price: bool = False,
) -> DealVerdict | None:
    variant = variant or {}
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
    stat_date = date.today().isoformat()
    snapshot = budget.get_snapshot()
    decision = learning.decide(
        brand, guessed_category, deal.price, deal.discount_percent,
        validation_multiplier=snapshot.validation_multiplier,
    )
    if timing:
        timing.record("rule_lookup", (time.monotonic() - rule_lookup_start) * 1000)

    legacy_kwargs = dict(
        is_new_family=is_new_family,
        is_unknown_brand=brand is None,
        is_new_category=guessed_category is None,
        is_new_lowest_family_price=is_new_family_low_price,
        rule_confidence=decision.rule.confidence if decision.rule is not None else None,
    )
    scored_kwargs = dict(
        asin=deal.asin, brand=brand, category=guessed_category,
        price=deal.price, family_id=family_id,
    )
    # Computed synchronously only when the scored classifier is actually
    # deciding priority; in shadow-only mode (flag off, verbose on) the
    # score is computed inside the fire-and-forget shadow task instead, so
    # the live forward path pays nothing for it.
    score_result = (
        scoring.compute_value_score(discount_percent=deal.discount_percent, **scored_kwargs)
        if SCORE_ENGINE_ENABLED
        else None
    )

    priority = budget.classify_priority(
        discount_percent=deal.discount_percent, score_result=score_result,
        **legacy_kwargs, **scored_kwargs,
    )
    database.record_priority_classification(stat_date, priority)

    if SCORE_ENGINE_LOG_VERBOSE:
        _spawn_shadow_score_log(
            legacy_kwargs, scored_kwargs, deal.discount_percent, score_result, stat_date
        )

    if decision.kind == "rule" and priority == 1:
        # Priority 1 ("always analyze") overrides a matched rule -- a
        # high-discount / brand-new-family / newly-cheapest deal is always
        # worth a fresh, real look rather than trusting a rule silently.
        decision = learning.Decision("ai", None, had_candidate=True)

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

    # Tier 2 (listener/budget.py's provider order): a cached family verdict.
    # Not gated on priority -- priority only decides whether to *spend* a
    # real AI call below; reusing an already-valid cached verdict is free,
    # and family.get_cached_verdict() has its own strict invalidation rules
    # (price/discount moved, new family-wide low/high, unseen variant
    # attribute, smart sampling) that already reject the cache for exactly
    # the situations that would otherwise force Priority 1 here (e.g. "new
    # lowest family price" invalidates the cache on its own).
    if decision.kind == "ai" and family_id is not None:
        cached = family.get_cached_verdict(family_id, deal.price, deal.discount_percent, variant)
        if cached is not None:
            database.record_ai_call_saved(stat_date)
            logger.info("family verdict cache hit (family=%s) — AI call saved", family_id)
            return DealVerdict(
                deal_quality=cached.deal_quality, reason=cached.reason,
                suggested_target=cached.suggested_target, category=cached.category,
                provider="family_cache",
            )

    if decision.kind == "ai" and priority != 1 and not budget.should_spend_ai_call(priority, snapshot):
        logger.info(
            "budget gate: priority %d deal skipped (remaining=%d/%d) — synthetic verdict",
            priority, snapshot.remaining, snapshot.daily_budget,
        )
        return DealVerdict(
            deal_quality="average",
            reason=f"Priority {priority} deal skipped under today's AI budget ({snapshot.remaining}/{snapshot.daily_budget} left) — not sent to an AI provider.",
            suggested_target=int(deal.price * 0.97),
            category=guessed_category or "other",
            provider="budget_skip",
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

    if family_id is not None:
        family.record_verdict(family_id, verdict, deal.price, deal.discount_percent, variant)

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
