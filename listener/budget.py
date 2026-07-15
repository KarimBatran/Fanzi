"""Daily AI-call budget manager: paces AI usage across the whole day instead
of spending until the provider's real quota exhausts within a few hours.

Nothing here talks to Gemini/Groq directly. listener.analyzer consults
get_snapshot()/classify_priority()/should_spend_ai_call() *before* falling
through to listener.ai_providers.AIProviderManager.get_verdict(), and
config.DAILY_ANALYSIS_CAP -- previously declared but never actually
enforced anywhere in the codebase -- is the total daily budget being paced
against.

Provider order per request, cheapest first (see listener/analyzer.py):
1. Learned rule (listener/learning.py) -- free.
2. Cached family verdict (listener/family.py) -- free.
3. Gemini, 4. Groq (listener/ai_providers.py) -- real AI calls, gated by
   this module's priority/budget decision.
5. A synthetic "budget-skipped" placeholder verdict when neither of the
   above applies and budget doesn't allow spending a real call.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta

import database
from config import (
    BUDGET_MIN_VALIDATION_MULTIPLIER,
    BUDGET_RESERVE_PERCENT,
    DAILY_ANALYSIS_CAP,
    PRIORITY1_DISCOUNT_THRESHOLD,
    PRIORITY1_RULE_CONFIDENCE_FLOOR,
    PRIORITY2_DISCOUNT_THRESHOLD,
    PRIORITY3_HEALTHY_BUDGET_MULTIPLIER,
    PRIORITY3_RULE_CONFIDENCE_CEILING,
    SCORE_ENGINE_ENABLED,
)

# Value Score -> priority tier cut points for classify_priority_scored().
# A score at/above SCORE_PRIORITY1_THRESHOLD is Priority 1, at/above
# SCORE_PRIORITY2_THRESHOLD is Priority 2, else Priority 3. Chosen so a
# fully-neutral deal (every component 0.5 -> score 50) lands in Priority 2
# ("analyze if budget allows"), matching the legacy classifier's own
# middle-ground treatment of an unremarkable deal.
SCORE_PRIORITY1_THRESHOLD = 65.0
SCORE_PRIORITY2_THRESHOLD = 40.0


def _today() -> str:
    return date.today().isoformat()


def _hours_until_midnight(now: datetime | None = None) -> float:
    now = now or datetime.now()
    tomorrow = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return max((tomorrow - now).total_seconds() / 3600, 1 / 60)


@dataclass
class BudgetSnapshot:
    daily_budget: int
    used_today: int
    remaining: int
    hours_remaining: float
    deals_per_hour: float
    calls_per_hour: float
    calls_saved_today: int
    target_calls_per_hour: float
    reserve_floor: int
    validation_multiplier: float
    confidence_threshold_display: float  # for /status only -- see get_snapshot()
    projected_exhaustion: str | None
    priority_1_count: int
    priority_2_count: int
    priority_3_count: int
    # Score-engine shadow mode (listener/scoring.py): how many deals today
    # had both classifiers evaluated side-by-side, and what fraction
    # disagreed -- the pre-flip validation signal shown in /status.
    shadow_total: int = 0
    shadow_divergences: int = 0
    shadow_divergence_rate: float | None = None


def _calls_used_today(stat_date: str) -> int:
    return database.get_gemini_quota_count(stat_date) + database.get_groq_quota_count(stat_date)


def get_snapshot(now: datetime | None = None) -> BudgetSnapshot:
    now = now or datetime.now()
    stat_date = _today()
    used = _calls_used_today(stat_date)
    remaining = max(DAILY_ANALYSIS_CAP - used, 0)
    hours_remaining = _hours_until_midnight(now)
    hours_elapsed = max(24 - hours_remaining, 1 / 60)

    calls_saved_today = database.get_learning_stats(stat_date)["ai_calls_saved"]
    priority_counts = database.get_priority_counts(stat_date)
    shadow_stats = database.get_shadow_stats(stat_date)
    deals_seen_today = sum(priority_counts.values())

    deals_per_hour = deals_seen_today / hours_elapsed
    calls_per_hour = used / hours_elapsed
    target_calls_per_hour = remaining / hours_remaining
    reserve_floor = int(DAILY_ANALYSIS_CAP * BUDGET_RESERVE_PERCENT / 100)

    projected_exhaustion = None
    if calls_per_hour > 0:
        hours_to_exhaustion = remaining / calls_per_hour
        if hours_to_exhaustion < hours_remaining:
            projected_exhaustion = (now + timedelta(hours=hours_to_exhaustion)).strftime("%H:%M")

    budget_health = _budget_health(remaining, reserve_floor)
    validation_multiplier = max(budget_health, BUDGET_MIN_VALIDATION_MULTIPLIER)
    # Display-only figure for /status: the learned-rule confidence at/above
    # which a rule is trusted without AI validation today, scaled between
    # learning.py's own 0.70 (always-AI) and 0.95 (rarely-validated) bands.
    # Deliberately RISES with budget health, not falls -- a low budget means
    # we trust rules more readily (skip AI more), which corresponds to a
    # LOWER bar for "trust the rule," i.e. a lower displayed threshold.
    confidence_threshold_display = 0.70 + (0.25 * budget_health)

    return BudgetSnapshot(
        daily_budget=DAILY_ANALYSIS_CAP, used_today=used, remaining=remaining,
        hours_remaining=hours_remaining, deals_per_hour=deals_per_hour, calls_per_hour=calls_per_hour,
        calls_saved_today=calls_saved_today, target_calls_per_hour=target_calls_per_hour,
        reserve_floor=reserve_floor, validation_multiplier=validation_multiplier,
        confidence_threshold_display=confidence_threshold_display,
        projected_exhaustion=projected_exhaustion,
        priority_1_count=priority_counts[1], priority_2_count=priority_counts[2],
        priority_3_count=priority_counts[3],
        shadow_total=shadow_stats["shadow_total"],
        shadow_divergences=shadow_stats["shadow_divergences"],
        shadow_divergence_rate=(
            shadow_stats["shadow_divergences"] / shadow_stats["shadow_total"]
            if shadow_stats["shadow_total"] > 0
            else None
        ),
    )


def _budget_health(remaining: int, reserve_floor: int) -> float:
    """0.0 at/below the reserve floor, 1.0 once remaining is at least
    PRIORITY3_HEALTHY_BUDGET_MULTIPLIER times the reserve floor.
    """
    healthy_at = max(reserve_floor * PRIORITY3_HEALTHY_BUDGET_MULTIPLIER, 1)
    if remaining <= reserve_floor:
        return 0.0
    return min((remaining - reserve_floor) / (healthy_at - reserve_floor), 1.0) if healthy_at > reserve_floor else 1.0


def classify_priority_legacy(
    *,
    discount_percent: int | None,
    is_new_family: bool,
    is_unknown_brand: bool,
    is_new_category: bool,
    is_new_lowest_family_price: bool,
    rule_confidence: float | None,
) -> int:
    """The original heuristic classifier, preserved verbatim (only renamed
    from classify_priority) -- the SCORE_ENGINE_ENABLED=false path must be
    byte-for-byte identical to pre-score-engine production behavior.

    Priority 1 ("always analyze"): high discount, a genuinely new family/
    brand/category, a variant that's now the cheapest in its family, or a
    learned rule too unconfident to trust at all. Priority 2 ("analyze if
    budget allows"): moderate discount or moderate rule confidence.
    Priority 3 ("usually skip"): everything else -- low discount and either
    no rule signal or a well-established, high-confidence rule.
    """
    if (
        (discount_percent is not None and discount_percent >= PRIORITY1_DISCOUNT_THRESHOLD)
        or is_new_family
        or is_unknown_brand
        or is_new_category
        or is_new_lowest_family_price
        or rule_confidence is None
        or rule_confidence < PRIORITY1_RULE_CONFIDENCE_FLOOR
    ):
        return 1

    if (discount_percent is not None and discount_percent >= PRIORITY2_DISCOUNT_THRESHOLD) or (
        rule_confidence < PRIORITY3_RULE_CONFIDENCE_CEILING
    ):
        return 2

    return 3


def classify_priority_scored(
    *,
    discount_percent: int | None,
    asin: str = "",
    brand: str | None = None,
    category: str | None = None,
    price: float = 0.0,
    family_id: str | None = None,
    score_result=None,
) -> int:
    """Value-Score-driven classifier (listener/scoring.py). The outlier
    safety check in listener/learning.py takes precedence over any score:
    an outlier deal is always Priority 1, exactly as learning.decide()
    itself would force a real AI call for it regardless of any rule.

    `score_result` (a scoring.ValueScoreResult) can be passed by callers
    that already computed the score (listener/analyzer.py computes it once
    for shadow logging) so it's never computed twice per deal.
    """
    from listener import learning, scoring

    if learning.is_outlier(brand, category, price, discount_percent):
        return 1

    if score_result is None:
        score_result = scoring.compute_value_score(
            asin=asin, brand=brand, category=category, price=price,
            discount_percent=discount_percent, family_id=family_id,
        )

    if score_result.total >= SCORE_PRIORITY1_THRESHOLD:
        return 1
    if score_result.total >= SCORE_PRIORITY2_THRESHOLD:
        return 2
    return 3


def classify_priority(
    *,
    discount_percent: int | None,
    is_new_family: bool,
    is_unknown_brand: bool,
    is_new_category: bool,
    is_new_lowest_family_price: bool,
    rule_confidence: float | None,
    asin: str = "",
    brand: str | None = None,
    category: str | None = None,
    price: float = 0.0,
    family_id: str | None = None,
    score_result=None,
) -> int:
    """The single dispatch point between the legacy heuristic and the Value
    Score engine -- the only branch on SCORE_ENGINE_ENABLED in the entire
    codebase. The extra keyword args (asin/brand/category/price/family_id/
    score_result) are consumed only by the scored path; the legacy path
    ignores them entirely, so with the flag off, behavior is identical to
    calling classify_priority_legacy directly.
    """
    if SCORE_ENGINE_ENABLED:
        return classify_priority_scored(
            discount_percent=discount_percent, asin=asin, brand=brand, category=category,
            price=price, family_id=family_id, score_result=score_result,
        )
    return classify_priority_legacy(
        discount_percent=discount_percent, is_new_family=is_new_family,
        is_unknown_brand=is_unknown_brand, is_new_category=is_new_category,
        is_new_lowest_family_price=is_new_lowest_family_price, rule_confidence=rule_confidence,
    )


def should_spend_ai_call(priority: int, snapshot: BudgetSnapshot) -> bool:
    """Gate applied only when nothing cheaper (learned rule / family cache)
    already produced a verdict -- i.e. only real Gemini/Groq spend is ever
    gated here.
    """
    if priority == 1:
        # Never gated by budget -- the reserve exists precisely so Priority
        # 1 deals can always be analyzed, all the way to a zero remaining
        # budget (at which point the real provider quota itself, not this
        # manager, is what stops further calls).
        return True
    if snapshot.remaining <= 0:
        return False
    if priority == 2:
        return snapshot.remaining > snapshot.reserve_floor
    return snapshot.remaining > snapshot.reserve_floor * PRIORITY3_HEALTHY_BUDGET_MULTIPLIER


@dataclass
class KnowledgeCoverage:
    brands_learned: int
    categories_learned: int
    families_learned: int
    estimated_ai_reduction_percent: float


def get_knowledge_coverage() -> KnowledgeCoverage:
    brands, categories = database.get_learned_brands_and_categories()
    families = database.count_product_families()
    saved = database.get_alltime_calls_saved()
    used = database.get_alltime_ai_calls_used()
    total = saved + used
    reduction = (saved / total * 100) if total > 0 else 0.0
    return KnowledgeCoverage(
        brands_learned=len(brands), categories_learned=len(categories),
        families_learned=families, estimated_ai_reduction_percent=reduction,
    )
