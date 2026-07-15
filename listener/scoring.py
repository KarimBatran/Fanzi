"""Deterministic, zero-AI-cost Value Score engine. Pure functions over
existing SQLite state -- no AI calls, no network calls, no writes. Combines
brand reputation, historical price percentile, family price percentile,
category deviation, and rarity into a single 0-100 score that
listener/budget.py's classify_priority_scored() maps to a priority tier,
replacing (behind SCORE_ENGINE_ENABLED) the legacy "raw discount % +
unknown-brand/new-family always wins Priority 1" heuristic.

Every component is normalized to 0.0-1.0 where higher = more worth spending
a real AI call on, and every component degrades to a neutral 0.5 (never an
extreme) when its underlying data doesn't exist yet -- a brand-new install
with empty tables scores everything at exactly the neutral midpoint rather
than systematically starving or flooding the AI budget.

All read helpers accept an optional sqlite3 connection so a caller scoring
many deals (or the benchmark test) can reuse one connection instead of
paying a connection-open per component lookup.
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass

import database
from config import (
    SCORE_WEIGHT_BRAND,
    SCORE_WEIGHT_CATEGORY_DEV,
    SCORE_WEIGHT_FAMILY_PCTL,
    SCORE_WEIGHT_PRICE_PCTL,
    SCORE_WEIGHT_RARITY,
)
from listener import learning

_NEUTRAL = 0.5

# skip=0.0 .. great=1.0, same scale as database._QUALITY_VALUE.
_QUALITY_VALUE = {"skip": 0.0, "average": 1 / 3, "good": 2 / 3, "great": 1.0}

# Observation count at (or beyond) which a product no longer counts as
# "rare" at all -- rarity decays linearly from 1.0 (never seen) to 0.0.
_RARITY_SATURATION_COUNT = 20

_WEIGHTS_SUM = (
    SCORE_WEIGHT_BRAND
    + SCORE_WEIGHT_PRICE_PCTL
    + SCORE_WEIGHT_FAMILY_PCTL
    + SCORE_WEIGHT_CATEGORY_DEV
    + SCORE_WEIGHT_RARITY
)
if abs(_WEIGHTS_SUM - 1.0) > 1e-6:
    raise ValueError(
        f"SCORE_WEIGHT_* values must sum to 1.0 (got {_WEIGHTS_SUM}) -- check .env"
    )


@dataclass
class ValueScoreResult:
    total: float  # 0-100
    brand_reputation: float
    price_percentile: float
    family_percentile: float | None  # None = deal has no family yet
    category_deviation: float
    rarity: float


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


# brand_reputation only ever changes via database.backfill_brand_reputation
# (a full rewrite), which bumps database.brand_reputation_generation -- so
# the whole (small) table is cached here and invalidated exactly when the
# data changes. learned category rules DO mutate at runtime (every new
# verdict), so their cache is a short TTL instead: category deviation is a
# soft signal built from slowly-accumulated history, and being up to
# _CATEGORY_RULES_TTL_SECONDS stale is immaterial -- while dropping a
# per-score SQL round-trip is what keeps scoring off the latency path.
_brand_cache: dict[str, float] | None = None
_brand_cache_generation: int = -1
_category_rules_cache: dict[tuple[str, str], tuple[str, float]] | None = None
_category_rules_cache_expires: float = 0.0
_CATEGORY_RULES_TTL_SECONDS = 30.0


def clear_caches() -> None:
    """Test isolation hook (tests/conftest.py) -- production code never
    needs it (generation/TTL invalidation covers real usage)."""
    global _brand_cache, _brand_cache_generation, _category_rules_cache, _category_rules_cache_expires
    _brand_cache = None
    _brand_cache_generation = -1
    _category_rules_cache = None
    _category_rules_cache_expires = 0.0


def _get_brand_cache(conn: sqlite3.Connection | None) -> dict[str, float]:
    global _brand_cache, _brand_cache_generation
    if _brand_cache is None or _brand_cache_generation != database.brand_reputation_generation:
        query = "SELECT brand, decayed_quality_mean FROM brand_reputation"
        if conn is not None:
            rows = conn.execute(query).fetchall()
        else:
            with database.get_connection() as owned:
                rows = owned.execute(query).fetchall()
        _brand_cache = {row["brand"]: row["decayed_quality_mean"] for row in rows}
        _brand_cache_generation = database.brand_reputation_generation
    return _brand_cache


def compute_brand_reputation(brand: str | None, conn: sqlite3.Connection | None = None) -> float:
    """Decay-weighted mean historical verdict quality for this brand (see
    database.backfill_brand_reputation). Neutral for an unseen/missing
    brand -- deliberately NOT a bonus: under the legacy heuristic an
    unknown brand always forced Priority 1; here it simply carries no
    signal either way.
    """
    if not brand:
        return _NEUTRAL
    mean = _get_brand_cache(conn).get(brand)
    if mean is None:
        return _NEUTRAL
    return _clamp01(mean)


def _price_percentile_from_stats(stats, price: float) -> float:
    n = stats["n"] or 0
    min_price, max_price = stats["min_price"], stats["max_price"]
    if n == 0 or min_price is None:
        return _NEUTRAL
    if price <= min_price:
        return 1.0
    if n < 2 or max_price is None or max_price <= min_price:
        return _NEUTRAL
    return _clamp01((max_price - price) / (max_price - min_price))


def compute_price_percentile(
    asin: str, family_id: str | None, price: float, conn: sqlite3.Connection | None = None,
) -> float:
    """How close this price sits to the observed historical floor for this
    ASIN (and, when in a family, its sibling ASINs): 1.0 = at/below the
    lowest price ever observed, 0.0 = at/above the highest. Neutral when
    fewer than two observations exist (a single point has no spread) or
    the observed spread is degenerate -- except that a price at/below the
    known floor still scores 1.0 even with one observation, since "matches
    or beats everything we've ever seen" is real signal on its own.
    """
    stats = database.get_price_observation_stats(asin, family_id, conn=conn)
    return _price_percentile_from_stats(stats, price)


def compute_family_percentile(
    family_id: str | None, price: float, conn: sqlite3.Connection | None = None,
) -> float | None:
    """Where this price sits among the family's known member prices:
    1.0 = at/below the family's best-known price, 0.0 = at/above the most
    expensive member. None when the deal has no family yet (the caller
    redistributes this component's weight, see compute_value_score).
    """
    if family_id is None:
        return None

    # Two plain statements (a point lookup + one single-pass aggregate) --
    # deliberately not folded into one scalar-subquery statement, which
    # measures several times slower per call (see
    # database.get_price_observation_stats for the same finding).
    family_query = "SELECT lowest_price FROM product_families WHERE family_id = ?"
    spread_query = (
        "SELECT MIN(last_price) AS min_p, MAX(last_price) AS max_p FROM family_members "
        "WHERE family_id = ? AND last_price IS NOT NULL"
    )
    if conn is not None:
        family = conn.execute(family_query, (family_id,)).fetchone()
        spread = conn.execute(spread_query, (family_id,)).fetchone() if family is not None else None
    else:
        with database.get_connection() as owned:
            family = owned.execute(family_query, (family_id,)).fetchone()
            spread = owned.execute(spread_query, (family_id,)).fetchone() if family is not None else None

    if family is None:
        return None
    lowest = family["lowest_price"]
    if lowest is not None and price <= lowest:
        return 1.0
    min_p, max_p = (spread["min_p"], spread["max_p"]) if spread is not None else (None, None)
    if min_p is None or max_p is None or max_p <= min_p:
        return _NEUTRAL
    return _clamp01((max_p - price) / (max_p - min_p))


def compute_category_deviation(
    category: str | None, price: float, discount_percent: float | None,
    conn: sqlite3.Connection | None = None,
) -> float:
    """What the learned category_price/category_discount rules say about
    this price/discount landing in this category: the confidence-weighted
    mean predicted quality of whichever bucket rules exist for this exact
    (category, bucket) pair. A category whose learned history says deals in
    this price/discount bucket tend to be great scores high; one whose
    history says they tend to be skips scores low. Neutral when no such
    rules exist yet (including category=None).
    """
    if not category:
        return _NEUTRAL

    rules = _get_category_rules_cache(conn)
    matches = [rules.get(("category_price", f"{category}|{learning.price_bucket(price)}"))]
    if discount_percent is not None:
        matches.append(
            rules.get(("category_discount", f"{category}|{learning.discount_bucket(int(discount_percent))}"))
        )
    matches = [m for m in matches if m is not None]

    if not matches:
        return _NEUTRAL

    weighted_sum = 0.0
    weight_sum = 0.0
    for predicted_quality, confidence in matches:
        quality_value = _QUALITY_VALUE.get(predicted_quality, _NEUTRAL)
        weighted_sum += quality_value * confidence
        weight_sum += confidence
    return _clamp01(weighted_sum / weight_sum) if weight_sum > 0 else _NEUTRAL


def _get_category_rules_cache(conn: sqlite3.Connection | None) -> dict[tuple[str, str], tuple[str, float]]:
    global _category_rules_cache, _category_rules_cache_expires
    if _category_rules_cache is None or time.monotonic() >= _category_rules_cache_expires:
        query = (
            "SELECT rule_type, key, predicted_quality, confidence FROM learned_rules "
            "WHERE enabled = 1 AND rule_type IN ('category_price', 'category_discount')"
        )
        if conn is not None:
            rows = conn.execute(query).fetchall()
        else:
            with database.get_connection() as owned:
                rows = owned.execute(query).fetchall()
        _category_rules_cache = {
            (row["rule_type"], row["key"]): (row["predicted_quality"], row["confidence"]) for row in rows
        }
        _category_rules_cache_expires = time.monotonic() + _CATEGORY_RULES_TTL_SECONDS
    return _category_rules_cache


def _rarity_from_stats(stats) -> float:
    """1.0 = never observed before, decaying linearly to 0.0 once this
    ASIN/family has _RARITY_SATURATION_COUNT or more price observations --
    a rarely-seen product is worth a fresh AI look more than one whose
    price history is already dense.
    """
    n = stats["n"] or 0
    return _clamp01(1.0 - n / _RARITY_SATURATION_COUNT)


def compute_value_score(
    *,
    asin: str,
    brand: str | None,
    category: str | None,
    price: float,
    discount_percent: float | None,
    family_id: str | None,
    conn: sqlite3.Connection | None = None,
) -> ValueScoreResult:
    """Weighted 0-100 combination of every component. When the deal has no
    family yet (family_percentile is None), that component's weight is
    redistributed proportionally across the others rather than silently
    scoring it neutral -- a family-less deal shouldn't be pulled toward the
    midpoint by a signal that structurally cannot exist for it.
    """
    if conn is not None:
        # Caller-supplied connection: transaction management is the
        # caller's -- batch callers (the benchmark, any future bulk scorer)
        # hold one read transaction across many scores.
        return _compute_value_score(asin, brand, category, price, discount_percent, family_id, conn)
    with database.get_connection() as owned:
        # One read transaction for all component lookups. Without this,
        # each SELECT in autocommit mode acquires/releases the database
        # file lock individually -- on Windows that's ~50us of syscalls
        # per query, dominating the actual work several times over.
        owned.execute("BEGIN")
        return _compute_value_score(asin, brand, category, price, discount_percent, family_id, owned)


def _compute_value_score(
    asin: str, brand: str | None, category: str | None, price: float,
    discount_percent: float | None, family_id: str | None, conn: sqlite3.Connection,
) -> ValueScoreResult:
    brand_score = compute_brand_reputation(brand, conn=conn)
    # Price percentile and rarity share the same MIN/MAX/COUNT aggregate --
    # fetched once, not twice (this function sits upstream of the AI soft
    # timeout, so every avoided round-trip matters at scale).
    observation_stats = database.get_price_observation_stats(asin, family_id, conn=conn)
    price_pctl = _price_percentile_from_stats(observation_stats, price)
    rarity = _rarity_from_stats(observation_stats)
    family_pctl = compute_family_percentile(family_id, price, conn=conn)
    category_dev = compute_category_deviation(category, price, discount_percent, conn=conn)

    components = [
        (brand_score, SCORE_WEIGHT_BRAND),
        (price_pctl, SCORE_WEIGHT_PRICE_PCTL),
        (category_dev, SCORE_WEIGHT_CATEGORY_DEV),
        (rarity, SCORE_WEIGHT_RARITY),
    ]
    if family_pctl is not None:
        components.append((family_pctl, SCORE_WEIGHT_FAMILY_PCTL))

    weight_sum = sum(w for _, w in components)
    total = sum(v * w for v, w in components) / weight_sum * 100 if weight_sum > 0 else 50.0

    return ValueScoreResult(
        total=total,
        brand_reputation=brand_score,
        price_percentile=price_pctl,
        family_percentile=family_pctl,
        category_deviation=category_dev,
        rarity=rarity,
    )
