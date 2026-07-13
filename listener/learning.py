"""Self-improving knowledge engine: learns from every successful Gemini/Groq
verdict and, once a pattern is confident enough, predicts the same verdict
for similar future deals without spending an AI request — while still
periodically re-validating itself against a real AI call so the knowledge
base can't go stale or silently drift wrong.

This module never replaces Gemini/Groq with hardcoded rules: every rule is
derived purely from real AI verdicts, decays in influence over time, and is
re-checked against fresh AI answers at a rate that scales with confidence
(see `decide()` / CONFIDENCE_BANDS).

Rule types, checked in priority order (first sufficiently confident one
wins):
1. brand_category  — key "brand|category"
2. brand           — key "brand"
3. category_price  — key "category|price_bucket"
4. category_discount — key "category|discount_bucket"

Category is not knowable before an AI call (Gemini/Groq decide it), so a
lightweight local keyword heuristic (`guess_category`) is used purely to
look up category-based rules ahead of time. It is never treated as
authoritative — the category actually stored in `verdicts`/used for
learning always comes from the real AI verdict.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
from dataclasses import dataclass
from datetime import date, datetime

import database
from config import (
    BRANDS_FILE,
    RULE_BRAND_CATEGORY_CONFIDENCE,
    RULE_BRAND_CONFIDENCE,
    RULE_DISCOUNT_CONFIDENCE,
    RULE_MIN_SAMPLES,
    RULE_MONTHLY_DECAY,
    RULE_OUTLIER_DISCOUNT,
    RULE_OUTLIER_MIN_PRICE,
    RULE_PRICE_CONFIDENCE,
    RULE_VALIDATION_RATE_HIGH,
    RULE_VALIDATION_RATE_MEDIUM,
)

logger = logging.getLogger("fanzi.listener.learning")

_SECONDS_PER_MONTH = 30.44 * 86400

_PRICE_BUCKETS: list[tuple[float, float | None]] = [
    (0, 100), (100, 200), (200, 500), (500, 1000), (1000, None),
]
_DISCOUNT_BUCKETS: list[tuple[float, float | None]] = [
    (0, 10), (10, 20), (20, 30), (30, 50), (50, None),
]

# Lightweight local heuristic for rule lookup only — never authoritative.
# Real category always comes from the AI verdict.
_CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "phone": ["موبايل", "هاتف", "iphone", "phone", "smartphone", "جالاكسي"],
    "headphones": ["سماعة", "سماعات", "headphone", "earbuds", "airpods", "earphone"],
    "laptop": ["لابتوب", "laptop", "notebook"],
    "cable": ["كابل", "شاحن", "شواحن", "cable", "charger", "adapter"],
    "appliance": [
        "غسالة", "ثلاجة", "مكيف", "خلاط", "مكنسة", "microwave", "fridge",
        "washing machine", "blender", "vacuum", "air fryer", "قهوة",
    ],
    "accessory": ["اكسسوار", "accessory", "case", "جراب", "حافظة"],
}

_RULE_TYPE_THRESHOLD = {
    "brand_category": RULE_BRAND_CATEGORY_CONFIDENCE,
    "brand": RULE_BRAND_CONFIDENCE,
    "category_price": RULE_PRICE_CONFIDENCE,
    "category_discount": RULE_DISCOUNT_CONFIDENCE,
}

_RULE_TYPE_PRIORITY = ("brand_category", "brand", "category_price", "category_discount")

_background_tasks: set[asyncio.Task] = set()

_brands_cache: list[str] | None = None


def _today() -> str:
    return date.today().isoformat()


def _load_brands() -> list[str]:
    global _brands_cache
    if _brands_cache is None:
        with open(BRANDS_FILE, encoding="utf-8") as f:
            _brands_cache = json.load(f)
    return _brands_cache


def extract_brand(title: str | None) -> str | None:
    """Case-insensitive substring match against the configured brand list.
    Never guesses — returns None (stored as NULL) if no known brand appears.
    """
    if not title:
        return None
    lowered = title.lower()
    for brand in _load_brands():
        if brand.lower() in lowered:
            return brand
    return None


def guess_category(title: str | None) -> str | None:
    if not title:
        return None
    lowered = title.lower()
    for category, keywords in _CATEGORY_KEYWORDS.items():
        for kw in keywords:
            if kw.lower() in lowered:
                return category
    return None


def price_bucket(price: float) -> str:
    for low, high in _PRICE_BUCKETS:
        if high is None:
            if price >= low:
                return f"{int(low)}+"
        elif low <= price < high:
            return f"{int(low)}-{int(high)}"
    return "unknown"


def discount_bucket(discount_percent: int | None) -> str:
    if discount_percent is None:
        return "unknown"
    for low, high in _DISCOUNT_BUCKETS:
        if high is None:
            if discount_percent >= low:
                return f"{int(low)}%+"
        elif low <= discount_percent < high:
            return f"{int(low)}-{int(high)}%"
    return "unknown"


@dataclass
class RuleMatch:
    rule_type: str
    key: str
    predicted_quality: str
    confidence: float
    sample_count: int


def _row_to_match(rule_type: str, row) -> RuleMatch:
    return RuleMatch(
        rule_type=rule_type,
        key=row["key"],
        predicted_quality=row["predicted_quality"],
        confidence=row["confidence"],
        sample_count=row["sample_count"],
    )


def _candidate_rules(brand: str | None, category: str | None, price: float, discount_percent: int | None) -> list[RuleMatch]:
    """Returns matching rules (if any row exists at all, regardless of
    whether it's confident enough yet), in priority order.
    """
    candidates: list[RuleMatch] = []

    if brand and category:
        row = database.get_learned_rule("brand_category", f"{brand}|{category}")
        if row is not None:
            candidates.append(_row_to_match("brand_category", row))

    if brand:
        row = database.get_learned_rule("brand", brand)
        if row is not None:
            candidates.append(_row_to_match("brand", row))

    if category:
        row = database.get_learned_rule("category_price", f"{category}|{price_bucket(price)}")
        if row is not None:
            candidates.append(_row_to_match("category_price", row))

        row = database.get_learned_rule("category_discount", f"{category}|{discount_bucket(discount_percent)}")
        if row is not None:
            candidates.append(_row_to_match("category_discount", row))

    return candidates


def is_outlier(brand: str | None, category: str | None, price: float, discount_percent: int | None) -> bool:
    """Conditions under which learned rules are always bypassed and AI is
    always called — these are treated as valuable learning opportunities,
    not just edge cases to avoid.
    """
    if discount_percent is not None and discount_percent > RULE_OUTLIER_DISCOUNT:
        return True
    if price is not None and price < RULE_OUTLIER_MIN_PRICE:
        return True
    if brand is None:
        return True
    if category is None:
        return True
    if not database.category_seen_before(category):
        return True
    return False


def _confidence_band_ai_probability(confidence: float) -> float | None:
    """Returns None if the rule shouldn't be used at all (confidence too
    low — always call AI instead), else the probability that this specific
    request is routed to AI anyway as a validation check.
    """
    if confidence < 0.70:
        return None
    if confidence < 0.85:
        return 0.50
    if confidence < 0.95:
        return RULE_VALIDATION_RATE_MEDIUM
    return RULE_VALIDATION_RATE_HIGH


@dataclass
class Decision:
    kind: str  # "rule" | "validate" | "ai"
    rule: RuleMatch | None
    had_candidate: bool  # whether any rule row existed at all (for rule_miss stats)


def decide(brand: str | None, category: str | None, price: float, discount_percent: int | None) -> Decision:
    if is_outlier(brand, category, price, discount_percent):
        return Decision("ai", None, had_candidate=False)

    candidates = _candidate_rules(brand, category, price, discount_percent)
    had_candidate = len(candidates) > 0

    for match in candidates:
        threshold = _RULE_TYPE_THRESHOLD[match.rule_type]
        if match.sample_count < RULE_MIN_SAMPLES or match.confidence < threshold:
            continue
        ai_probability = _confidence_band_ai_probability(match.confidence)
        if ai_probability is None:
            continue
        if random.random() < ai_probability:
            return Decision("validate", match, had_candidate=True)
        return Decision("rule", match, had_candidate=True)

    return Decision("ai", None, had_candidate=had_candidate)


def format_explanation(rule: RuleMatch, brand: str | None, category: str | None) -> str:
    label = {
        "brand_category": f"Brand: {brand}\nCategory: {category}",
        "brand": f"Brand: {brand}",
        "category_price": f"Category: {category}\nPrice range: {rule.key.split('|', 1)[1]} EGP",
        "category_discount": f"Category: {category}\nDiscount range: {rule.key.split('|', 1)[1]}",
    }[rule.rule_type]
    return (
        "📚 Learned Pattern\n\n"
        f"{label}\n\n"
        f"Based on {rule.sample_count} historical AI analyses\n"
        f"Confidence: {rule.confidence:.0%}\n\n"
        f"Predicted quality: {rule.predicted_quality.capitalize()}"
    )


def mark_validated(rule_type: str, key: str, when: datetime) -> None:
    database.set_rule_last_validated(rule_type, key, when.isoformat())


def _update_rule(rule_type: str, key: str, quality: str, now: datetime) -> None:
    """Incrementally updates one rule from a single new (real) verdict:
    decays existing votes by elapsed time since the rule's last update,
    adds the new vote, and recomputes predicted_quality/confidence —
    never rescans verdict history.
    """
    existing = database.get_learned_rule(rule_type, key)
    votes = database.get_rule_votes(rule_type, key)

    if existing is not None and existing["last_updated"]:
        last_updated = datetime.fromisoformat(existing["last_updated"])
        elapsed_months = max(0.0, (now - last_updated).total_seconds() / _SECONDS_PER_MONTH)
        decay = RULE_MONTHLY_DECAY ** elapsed_months
        votes = {q: w * decay for q, w in votes.items()}

    votes[quality] = votes.get(quality, 0.0) + 1.0
    database.set_rule_votes(rule_type, key, votes)

    total_weight = sum(votes.values())
    dominant_quality = max(votes, key=votes.get)
    confidence = votes[dominant_quality] / total_weight if total_weight > 0 else 0.0
    sample_count = (existing["sample_count"] if existing is not None else 0) + 1
    threshold = _RULE_TYPE_THRESHOLD[rule_type]
    enabled = sample_count >= RULE_MIN_SAMPLES and confidence >= threshold
    rule_version = (existing["rule_version"] if existing is not None else 0) + 1

    database.upsert_learned_rule(
        rule_type=rule_type,
        key=key,
        predicted_quality=dominant_quality,
        confidence=confidence,
        sample_count=sample_count,
        last_updated=now.isoformat(),
        last_validated=existing["last_validated"] if existing is not None else None,
        enabled=enabled,
        rule_version=rule_version,
    )


async def record_and_learn(
    asin: str, provider: str, brand: str | None, category: str, title: str | None,
    price: float, discount_percent: int | None, deal_quality: str, reason: str,
    suggested_target: int, channel: str | None,
) -> None:
    """Stores the verdict and updates every applicable rule type
    incrementally. Only ever called for successful Gemini/Groq verdicts —
    never for unavailable/skipped/fallback/parser-failure outcomes.
    Intended to run as a fire-and-forget background task (see
    `spawn_learning_task`) so it never delays forwarding the deal.
    """
    now = datetime.now()
    try:
        database.insert_verdict(
            asin=asin, provider=provider, brand=brand, category=category, title=title,
            current_price=price, discount_percent=discount_percent, deal_quality=deal_quality,
            reason=reason, suggested_target=suggested_target, channel=channel,
            timestamp=now.isoformat(),
        )
        if brand:
            _update_rule("brand", brand, deal_quality, now)
        if brand and category:
            _update_rule("brand_category", f"{brand}|{category}", deal_quality, now)
        if category:
            _update_rule("category_price", f"{category}|{price_bucket(price)}", deal_quality, now)
            _update_rule("category_discount", f"{category}|{discount_bucket(discount_percent)}", deal_quality, now)
    except Exception:
        logger.exception("learning task failed for ASIN %s", asin)


def spawn_learning_task(**kwargs) -> asyncio.Task:
    task = asyncio.create_task(record_and_learn(**kwargs))
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    return task


def status_snapshot() -> dict:
    """Feeds the "📚 Learned rules" / "🎯 AI calls saved" block in /status."""
    counts = database.count_rules_by_type()
    enabled_rules = database.list_learned_rules(enabled_only=True)
    avg_confidence = (
        sum(r["confidence"] for r in enabled_rules) / len(enabled_rules) if enabled_rules else 0.0
    )
    stats = database.get_learning_stats(_today())
    return {
        "brand_rules": counts.get("brand", 0),
        "brand_category_rules": counts.get("brand_category", 0),
        "category_price_rules": counts.get("category_price", 0),
        "category_discount_rules": counts.get("category_discount", 0),
        "total_rules": len(enabled_rules),
        "ai_calls_saved_today": stats["ai_calls_saved"],
        "avg_confidence": avg_confidence,
        "kb_version": database.get_kb_version(),
    }


def rebuild_rules_from_history() -> tuple[int, int]:
    """Deletes all learned rules and replays every stored verdict in
    chronological order to rebuild them from scratch. Returns
    (verdict_count, rules_created). Synchronous/CPU-light (pure Python +
    SQLite) — callers needing progress updates should run this in a thread
    or chunk it themselves; see bot.py's /rebuildrules for how it's wrapped
    to run without blocking the event loop.
    """
    database.clear_learned_rules()
    rows = database.get_all_verdicts_chronological()
    for row in rows:
        now = datetime.fromisoformat(row["timestamp"])
        brand, category = row["brand"], row["category"]
        quality = row["deal_quality"]
        if brand:
            _update_rule("brand", brand, quality, now)
        if brand and category:
            _update_rule("brand_category", f"{brand}|{category}", quality, now)
        if category:
            _update_rule("category_price", f"{category}|{price_bucket(row['current_price'])}", quality, now)
            _update_rule(
                "category_discount", f"{category}|{discount_bucket(row['discount_percent'])}", quality, now
            )
    database.bump_kb_version()
    rule_count = len(database.list_learned_rules(enabled_only=True))
    return len(rows), rule_count
