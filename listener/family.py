"""Product Family detection: recognizes that multiple Amazon ASINs (colors,
sizes, capacities, pack sizes, ...) can be variants of the same underlying
product, so the bot avoids duplicate notifications for identical variants
while still notifying when a different variant is genuinely a better deal
(cheaper, higher discount, or a better AI verdict).

Multi-signal matching, cheapest signal first -- never a single field:
1. Exact: this ASIN was already assigned to a family before (family_members).
2. Deterministic: brand + normalized title (variant words stripped) +
   category all match a known family's fingerprint exactly.
3. Fuzzy: normalized-title similarity (difflib) against known families
   sharing the same brand/category, when the ratio is at or above
   FAMILY_FUZZY_MATCH_THRESHOLD.
4. AI similarity: consulted only when the fuzzy ratio falls in the
   ambiguous band [FAMILY_AI_MATCH_FLOOR, FAMILY_FUZZY_MATCH_THRESHOLD) --
   and even then, only once per (asin, anchor_asin) pair ever: the decision
   (true or false) is cached permanently in family_ai_decisions and never
   re-asked for that exact pair.

Known limitation, stated rather than faked: this pipeline only ever sees
parsed post text (listener/parser.py), never Amazon's own product-variation
API or a product image, so "parent ASIN" and "image hash" from the original
spec have no real data source in this app and are not used as matching
signals -- only brand, normalized title, category, and extracted variant
attributes are.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from difflib import SequenceMatcher

import database
from config import (
    DUPLICATE_WINDOW_HOURS,
    FAMILY_AI_MATCH_FLOOR,
    FAMILY_DUPLICATE_DISCOUNT_TOLERANCE_PERCENT,
    FAMILY_DUPLICATE_PRICE_TOLERANCE_EGP,
    FAMILY_FUZZY_MATCH_THRESHOLD,
    FAMILY_VERDICT_CACHE_WINDOW_HOURS,
    FAMILY_VERDICT_DISCOUNT_CHANGE_THRESHOLD_PERCENT,
    FAMILY_VERDICT_PRICE_CHANGE_THRESHOLD_PERCENT,
    SMART_SAMPLING_EVERY_N_VARIANTS,
    SMART_SAMPLING_INTERVAL_HOURS,
    SMART_SAMPLING_VARIANT_THRESHOLD,
)
from listener.ai_providers import AIVerdict, _extract_json_object, get_manager

logger = logging.getLogger("fanzi.listener.family")

_QUALITY_RANK = {"skip": 0, "average": 1, "good": 2, "great": 3}

# ---- Variant extraction (additive metadata -- never removed from the
# stored title itself, only from the *normalized* copy used for matching) --

_COLOR_WORDS = [
    "rose gold", "red", "black", "blue", "white", "green", "yellow", "pink",
    "grey", "gray", "silver", "gold", "purple", "orange", "brown", "beige",
    "navy", "أحمر", "أسود", "أزرق", "أبيض", "أخضر", "أصفر", "بيج", "رمادي",
    "فضي", "ذهبي",
]
_COLOR_RE = re.compile(
    r"\b(" + "|".join(re.escape(c) for c in sorted(_COLOR_WORDS, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)

_SIZE_WORD_RE = re.compile(r"\b(XXXL|XXL|XL|L|M|S)\b")
_SIZE_NUMERIC_RE = re.compile(r"(?:size|مقاس)\s*[:\-]?\s*(\d{1,3})\b", re.IGNORECASE)
# Standalone shoe/clothing size, no anchor word -- deliberately narrow to
# minimize false positives against model numbers/other digits in a title.
_SIZE_STANDALONE_RE = re.compile(r"\b(3[5-9]|4[0-6])\b")

_CAPACITY_RE = re.compile(r"\b(\d+)\s?(GB|TB|MB|ML|L|KG)\b", re.IGNORECASE)

_PACK_RE = re.compile(r"\bpack of\s*(\d+)\b|\b(\d+)\s*[- ]?pack\b|\bعبوة\s*(\d+)\b", re.IGNORECASE)

_WHITESPACE_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)


def extract_variant_attributes(title: str) -> dict[str, str]:
    """Extracts structured variant attributes (color/size/capacity/pack)
    from a title. Does not modify or strip anything from the caller's copy
    of the title -- this is purely additive metadata alongside it.
    """
    if not title:
        return {}
    variant: dict[str, str] = {}

    color_match = _COLOR_RE.search(title)
    if color_match:
        variant["color"] = color_match.group(1).title()

    size_match = _SIZE_NUMERIC_RE.search(title) or _SIZE_WORD_RE.search(title) or _SIZE_STANDALONE_RE.search(title)
    if size_match:
        variant["size"] = size_match.group(1)

    capacity_match = _CAPACITY_RE.search(title)
    if capacity_match:
        variant["capacity"] = f"{capacity_match.group(1)}{capacity_match.group(2).upper()}"

    pack_match = _PACK_RE.search(title)
    if pack_match:
        count = next(g for g in pack_match.groups() if g)
        variant["pack"] = f"{count} Pack"

    return variant


def variant_label(variant: dict[str, str]) -> str:
    """Short human-readable label for a variant dict, preferring the most
    visually distinctive attribute -- used in notification text.
    """
    for key in ("color", "size", "capacity", "pack"):
        if key in variant:
            return variant[key]
    return "Standard"


def normalize_title_for_family(title: str) -> str:
    """Lowercased, punctuation-stripped, whitespace-collapsed title with all
    recognized variant tokens removed -- this is the text actually compared
    across ASINs for family matching, so two colors of the same product
    normalize to the same string.
    """
    if not title:
        return ""
    text = title
    for pattern in (_COLOR_RE, _SIZE_WORD_RE, _SIZE_NUMERIC_RE, _CAPACITY_RE, _PACK_RE):
        text = pattern.sub(" ", text)
    text = _PUNCT_RE.sub(" ", text.lower())
    return _WHITESPACE_RE.sub(" ", text).strip()


def _deterministic_key(brand: str | None, normalized_title: str, category: str | None) -> str:
    raw = f"{(brand or '').lower()}|{normalized_title}|{(category or '').lower()}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:20]


def _new_family_id() -> str:
    return f"fam_{uuid.uuid4().hex[:16]}"


@dataclass
class FamilyDecision:
    family_id: str
    variant: dict[str, str]
    notify_kind: str  # "new_family" | "duplicate" | "pending" | "better_variant" | "new_variant"
    matched_via: str  # "new" | "exact" | "fuzzy" | "ai"
    confidence: float
    previous_best_price: float | None = None
    previous_best_label: str | None = None
    savings: float | None = None


# ---- AI similarity (only for the ambiguous confidence band) --------------

_AI_SIMILARITY_SYSTEM_PROMPT = (
    "You determine whether two Amazon product listings are variants of the "
    "same underlying product (different color/size/capacity/pack, not "
    "genuinely different products). You receive ONLY title, brand, "
    "category, and variant attributes for each -- never a URL. Return ONLY "
    'a JSON object with keys: same_family (true/false), confidence '
    "(0.0-1.0), reason (one short sentence)."
)


def _build_similarity_prompt(candidate: dict, family: dict) -> str:
    return (
        "Product A:\n"
        f"title={candidate['title']!r}\nbrand={candidate['brand'] or 'unknown'}\n"
        f"category={candidate['category'] or 'unknown'}\n\n"
        "Product B:\n"
        f"title={family['title']!r}\nbrand={family['brand'] or 'unknown'}\n"
        f"category={family['category'] or 'unknown'}\n"
    )


def _parse_similarity_response(text: str) -> tuple[bool, float, str] | None:
    data, *_ = _extract_json_object(text)
    if data is None or "same_family" not in data:
        return None
    try:
        return bool(data["same_family"]), float(data.get("confidence", 0.5)), str(data.get("reason", ""))
    except (TypeError, ValueError):
        return None


async def _ask_ai_same_family(candidate: dict, family: dict) -> tuple[bool, float, str]:
    """Tries Gemini then Groq (whichever is configured), never raises --
    AI unavailable defaults to "different family" (the safer failure mode:
    it costs an extra notification rather than silently merging two
    genuinely different products).
    """
    prompt = _build_similarity_prompt(candidate, family)
    manager = get_manager()
    for provider in (manager.gemini, manager.groq):
        if not provider.is_configured():
            continue
        try:
            text = await provider.generate(prompt, system_prompt=_AI_SIMILARITY_SYSTEM_PROMPT)
        except Exception:
            logger.warning("family AI similarity check failed via %s", provider.name)
            continue
        parsed = _parse_similarity_response(text)
        if parsed is not None:
            return parsed
    return False, 0.0, "AI unavailable -- defaulted to different families"


async def _fuzzy_or_ai_match(
    asin: str, title: str, normalized_title: str, brand: str | None, category: str | None,
):
    candidates = database.get_families_by_brand_category(brand, category)
    best_family = None
    best_ratio = 0.0
    for cand in candidates:
        ratio = SequenceMatcher(None, normalized_title, cand["normalized_title"]).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_family = cand

    if best_family is None:
        return None, "new", best_ratio

    if best_ratio >= FAMILY_FUZZY_MATCH_THRESHOLD:
        return best_family, "fuzzy", best_ratio

    if best_ratio < FAMILY_AI_MATCH_FLOOR:
        return None, "new", best_ratio

    # Ambiguous zone -- consult AI, permanently cached per (asin, anchor_asin).
    anchor_asin = best_family["anchor_asin"]
    cached = database.get_family_ai_decision(asin, anchor_asin)
    if cached is not None:
        same = bool(cached["same_family"])
        logger.info(
            "family AI decision for (%s, %s) served from cache: same_family=%s", asin, anchor_asin, same
        )
        return (best_family if same else None), ("ai" if same else "new"), cached["confidence"]

    same, confidence, reason = await _ask_ai_same_family(
        {"title": title, "brand": brand, "category": category},
        {"title": best_family["normalized_title"], "brand": best_family["brand"], "category": best_family["category"]},
    )
    database.store_family_ai_decision(asin, anchor_asin, same, confidence, reason, datetime.now().isoformat())
    logger.info(
        "family AI decision for (%s, %s): same_family=%s (confidence=%.2f) -- %s",
        asin, anchor_asin, same, confidence, reason,
    )
    return (best_family if same else None), ("ai" if same else "new"), confidence


async def pre_check(
    asin: str, title: str, price: float, discount_percent: int | None,
    brand: str | None, category: str | None,
) -> FamilyDecision:
    """Resolves this ASIN's family and decides whether it's a true duplicate
    *before* any deal-quality AI call is made, so a true-duplicate variant
    never costs an AI request. For every non-duplicate outcome, callers must
    call finalize() afterwards once a verdict (or None) is known -- mirrors
    dedup.check()/mark_seen()'s two-step shape.
    """
    variant = extract_variant_attributes(title)
    normalized_title = normalize_title_for_family(title)

    existing_member = database.get_family_member(asin)
    if existing_member is not None:
        family = database.get_product_family(existing_member["family_id"])
        matched_via, confidence = "exact", 1.0
    else:
        deterministic_key = _deterministic_key(brand, normalized_title, category)
        family = database.get_family_by_deterministic_key(deterministic_key)
        if family is not None:
            matched_via, confidence = "exact", 1.0
        else:
            family, matched_via, confidence = await _fuzzy_or_ai_match(
                asin, title, normalized_title, brand, category
            )

    now = datetime.now().isoformat()

    if family is None:
        family_id = _new_family_id()
        deterministic_key = _deterministic_key(brand, normalized_title, category)
        database.create_product_family(
            family_id=family_id, brand=brand, normalized_title=normalized_title,
            category=category, deterministic_key=deterministic_key, anchor_asin=asin,
            lowest_price=price, highest_discount_percent=discount_percent,
            first_seen=now, last_seen=now, best_variant_label=variant_label(variant),
        )
        database.upsert_family_member(
            asin=asin, family_id=family_id, variant_json=json.dumps(variant),
            price=price, discount_percent=discount_percent, seen_at=now,
        )
        return FamilyDecision(family_id=family_id, variant=variant, notify_kind="new_family", matched_via="new", confidence=1.0)

    family_id = family["family_id"]
    variant_json = json.dumps(variant)

    twin = database.get_family_member_by_variant(family_id, variant_json)
    if twin is not None:
        last_seen_at = datetime.fromisoformat(twin["last_seen_at"])
        within_window = datetime.now() - last_seen_at < timedelta(hours=DUPLICATE_WINDOW_HOURS)
        price_close = (
            twin["last_price"] is not None and abs(price - twin["last_price"]) <= FAMILY_DUPLICATE_PRICE_TOLERANCE_EGP
        )
        discount_close = (discount_percent is None and twin["last_discount_percent"] is None) or (
            discount_percent is not None
            and twin["last_discount_percent"] is not None
            and abs(discount_percent - twin["last_discount_percent"]) <= FAMILY_DUPLICATE_DISCOUNT_TOLERANCE_PERCENT
        )
        if within_window and price_close and discount_close:
            database.upsert_family_member(
                asin=asin, family_id=family_id, variant_json=variant_json,
                price=price, discount_percent=discount_percent, seen_at=now,
            )
            database.touch_product_family(family_id, now)
            return FamilyDecision(
                family_id=family_id, variant=variant, notify_kind="duplicate",
                matched_via=matched_via, confidence=confidence,
            )

    database.upsert_family_member(
        asin=asin, family_id=family_id, variant_json=variant_json,
        price=price, discount_percent=discount_percent, seen_at=now,
    )
    return FamilyDecision(
        family_id=family_id, variant=variant, notify_kind="pending",
        matched_via=matched_via, confidence=confidence,
        previous_best_price=family["lowest_price"], previous_best_label=family["best_variant_label"],
    )


def finalize(
    family_id: str, asin: str, variant: dict, price: float, discount_percent: int | None,
    verdict_quality: str | None,
) -> FamilyDecision:
    """Called after a deal-quality verdict (or None) is known, for every
    "pending" outcome from pre_check(). Updates the family's best-known
    price/discount/verdict-quality/label and decides "better_variant"
    (this variant beats the family's prior best on price, discount, or AI
    quality) vs "new_variant" (a genuine new choice with no clear edge).
    """
    family = database.get_product_family(family_id)
    now = datetime.now().isoformat()

    previous_lowest_price = family["lowest_price"]
    previous_highest_discount = family["highest_discount_percent"]
    previous_best_quality = family["best_verdict_quality"]
    previous_best_label = family["best_variant_label"]

    is_lower_price = previous_lowest_price is None or price < previous_lowest_price
    is_higher_discount = discount_percent is not None and (
        previous_highest_discount is None or discount_percent > previous_highest_discount
    )
    # previous_best_quality is None until some variant's verdict has actually
    # been recorded here (a brand-new family's anchor member never goes
    # through finalize() at all, since "new_family" isn't a pending outcome)
    # -- treated as "no baseline to compare against" rather than as a rank
    # of 0, so a variant's mere presence of *some* verdict never trivially
    # counts as "better" against a family that simply hasn't recorded one yet.
    is_better_verdict = (
        verdict_quality is not None
        and previous_best_quality is not None
        and _QUALITY_RANK.get(verdict_quality, 0) > _QUALITY_RANK.get(previous_best_quality, 0)
    )

    is_better = is_lower_price or is_higher_discount or is_better_verdict
    label = variant_label(variant)

    new_lowest_price = min(price, previous_lowest_price) if previous_lowest_price is not None else price
    new_highest_discount = discount_percent if is_higher_discount else previous_highest_discount
    new_best_quality = verdict_quality if is_better_verdict else previous_best_quality
    new_best_label = label if is_lower_price else previous_best_label
    new_best_asin = asin if is_lower_price else family["best_variant_asin"]

    database.update_product_family_aggregates(
        family_id=family_id, lowest_price=new_lowest_price,
        highest_discount_percent=new_highest_discount, best_verdict_quality=new_best_quality,
        best_variant_label=new_best_label, best_variant_asin=new_best_asin, last_seen=now,
    )

    savings = (previous_lowest_price - price) if (is_lower_price and previous_lowest_price is not None) else None

    return FamilyDecision(
        family_id=family_id, variant=variant,
        notify_kind="better_variant" if is_better else "new_variant",
        matched_via="", confidence=1.0,
        previous_best_price=previous_lowest_price, previous_best_label=previous_best_label,
        savings=savings,
    )


# ---- AI-verdict cache (listener/budget.py's "cached family verdict" tier) -

def _smart_sampling_forces_fresh_call(family, last_verdict_at: datetime) -> bool:
    """A well-learned family (enough known variants) still gets a real AI
    call periodically -- either every SMART_SAMPLING_EVERY_N_VARIANTS-th
    distinct variant, or after SMART_SAMPLING_INTERVAL_HOURS since the last
    real verdict, whichever comes first -- to detect market drift instead
    of trusting a cached verdict forever.
    """
    variant_count = family["variant_count"]
    if variant_count < SMART_SAMPLING_VARIANT_THRESHOLD:
        return False
    if variant_count % SMART_SAMPLING_EVERY_N_VARIANTS == 0:
        return True
    return datetime.now() - last_verdict_at >= timedelta(hours=SMART_SAMPLING_INTERVAL_HOURS)


def get_cached_verdict(family_id: str, price: float, discount_percent: int | None, variant: dict) -> AIVerdict | None:
    """Reuses the family's last REAL AI verdict for a new variant instead of
    spending another AI call -- unless price/discount moved significantly,
    this is a new family-wide lowest price/highest discount (always worth a
    fresh look), a genuinely new kind of variant attribute appeared, or
    smart sampling says this occurrence must be a real call. Returns None
    (never guesses) whenever any of these apply, or when no real verdict has
    ever been cached for this family yet.
    """
    family = database.get_product_family(family_id)
    if family is None or family["last_verdict_quality"] is None or family["last_verdict_at"] is None:
        return None

    last_verdict_at = datetime.fromisoformat(family["last_verdict_at"])
    if datetime.now() - last_verdict_at >= timedelta(hours=FAMILY_VERDICT_CACHE_WINDOW_HOURS):
        return None

    last_price = family["last_verdict_price"]
    if last_price:
        price_change_pct = abs(price - last_price) / last_price * 100
        if price_change_pct > FAMILY_VERDICT_PRICE_CHANGE_THRESHOLD_PERCENT:
            return None

    last_discount = family["last_verdict_discount_percent"]
    if (discount_percent is None) != (last_discount is None):
        return None
    if (
        discount_percent is not None
        and last_discount is not None
        and abs(discount_percent - last_discount) > FAMILY_VERDICT_DISCOUNT_CHANGE_THRESHOLD_PERCENT
    ):
        return None

    if family["lowest_price"] is not None and price < family["lowest_price"]:
        return None
    if discount_percent is not None and (
        family["highest_discount_percent"] is None or discount_percent > family["highest_discount_percent"]
    ):
        return None

    cached_keys = set(json.loads(family["last_verdict_variant_keys"] or "[]"))
    if not set(variant.keys()) <= cached_keys:
        return None

    if _smart_sampling_forces_fresh_call(family, last_verdict_at):
        return None

    return AIVerdict(
        provider="family_cache",
        deal_quality=family["last_verdict_quality"],
        reason=family["last_verdict_reason"],
        suggested_target=family["last_verdict_suggested_target"],
        category=family["last_verdict_category"],
    )


def record_verdict(family_id: str, verdict: AIVerdict, price: float, discount_percent: int | None, variant: dict) -> None:
    """Caches a REAL (Gemini/Groq) verdict for future get_cached_verdict()
    calls. Callers must never pass a verdict whose own provider is already
    "family_cache"/"learned"/"budget_skip" -- that would let a non-AI
    verdict silently perpetuate itself as if it were a fresh AI opinion.
    """
    database.record_family_verdict(
        family_id=family_id, quality=verdict.deal_quality, reason=verdict.reason,
        suggested_target=verdict.suggested_target, category=verdict.category, provider=verdict.provider,
        price=price, discount_percent=discount_percent,
        variant_keys_json=json.dumps(sorted(variant.keys())), decided_at=datetime.now().isoformat(),
    )
