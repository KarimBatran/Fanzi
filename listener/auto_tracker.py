"""Adds qualifying deals to tracked_products for the admin user."""

from __future__ import annotations

import database
from amazon.parser import normalize_product_url
from config import ADMIN_TELEGRAM_ID
from listener.analyzer import DealVerdict
from listener.parser import ParsedDeal


def auto_add_if_qualifying(deal: ParsedDeal, verdict: DealVerdict) -> str:
    """Returns one of: "added", "already_tracked", "skipped"."""
    if verdict.deal_quality not in ("great", "good"):
        return "skipped"

    user = database.get_or_create_user(ADMIN_TELEGRAM_ID, None)

    if database.get_tracked_product_by_asin(user.id, deal.asin) is not None:
        return "already_tracked"

    database.add_tracked_product(
        user_id=user.id,
        asin=deal.asin,
        title=deal.title,
        url=normalize_product_url(deal.asin),
        current_price=deal.price,
        target_price=verdict.suggested_target,
    )
    return "added"
