"""Groq API call to assess deal quality and produce a verdict."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from groq import AsyncGroq, APIStatusError

from config import GROQ_API_KEY
from listener.parser import ParsedDeal

logger = logging.getLogger("fanzi.listener.analyzer")

_MODEL = "llama-3.1-8b-instant"
_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"

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


@dataclass
class DealVerdict:
    deal_quality: str
    reason: str
    suggested_target: int
    category: str


def meets_min_quality(deal_quality: str, min_quality: str) -> bool:
    return _QUALITY_RANK.get(deal_quality, 0) >= _QUALITY_RANK.get(min_quality, 2)


async def analyze_deal(deal: ParsedDeal, price_history: float | None) -> DealVerdict | None:
    """Returns None (rather than raising) on any API failure, so callers can
    fall back to forwarding the raw deal instead of dropping it silently.
    """
    client = AsyncGroq(api_key=GROQ_API_KEY)
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
        response = await client.chat.completions.create(
            model=_MODEL,
            max_tokens=256,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
        )
    except APIStatusError as exc:
        # Surface enough detail (status, body, Cloudflare/request ID, endpoint)
        # to diagnose network/WAF-level failures without needing an ad-hoc
        # script — these show up as generic SDK exceptions otherwise.
        ray_id = exc.response.headers.get("cf-ray") if exc.response is not None else None
        request_id = exc.response.headers.get("x-request-id") if exc.response is not None else None
        logger.error(
            "deal analysis failed for ASIN %s: endpoint=%s status=%s ray_id=%s "
            "request_id=%s body=%s",
            deal.asin,
            _ENDPOINT,
            exc.status_code,
            ray_id,
            request_id,
            exc.response.text if exc.response is not None else None,
        )
        return None
    except Exception:
        logger.exception("deal analysis failed for ASIN %s (endpoint=%s)", deal.asin, _ENDPOINT)
        return None
    finally:
        await client.close()

    text = response.choices[0].message.content if response.choices else None
    if text is None:
        logger.warning("no content in analysis response for ASIN %s", deal.asin)
        return None

    try:
        data = json.loads(text)
        return DealVerdict(
            deal_quality=data["deal_quality"],
            reason=data["reason"],
            suggested_target=int(data["suggested_target"]),
            category=data["category"],
        )
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        logger.exception("malformed analysis response for ASIN %s: %s", deal.asin, text)
        return None
