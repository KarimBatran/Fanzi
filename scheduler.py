"""APScheduler background jobs: periodic tracked-product price checks +
channel watchdog + daily heartbeat.

Tracked-product monitoring and deal forwarding are two fully independent
systems. Audited explicitly for this: run_check_cycle and its helpers below
never call listener.dedup.check()/mark_seen() or touch
global_duplicate_deals in any way -- the only dedup-related call anywhere in
this module is dedup.cleanup_expired() at the very end of the cycle, which
is periodic maintenance for the *deal-forwarding* dedup table and is never
consulted (nor does it need to be) when deciding whether to notify about a
tracked product's price. A tracked product's own state
(current_price/available/last_notified_price on its row) is the only input
to that decision.
"""

from __future__ import annotations

import logging

from zoneinfo import ZoneInfo

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from telegram import Bot

import database
import health
from amazon.tracker import PriceNotFoundError, ProductFetchError, fetch_product, format_price
from config import ADMIN_TELEGRAM_ID, CHANNEL_WATCHDOG_INTERVAL_MINUTES, CHECK_INTERVAL_MINUTES
from listener import channels_store, dedup, watchdog
from models.tracked_product import TrackedProduct

_CAIRO_TZ = ZoneInfo("Africa/Cairo")

logger = logging.getLogger("fanzi.scheduler")

TARGET_ALERT_TEMPLATE = (
    "🔥 Target Reached\n\n"
    "Product:\n{title}\n\n"
    "Target:\n{target_price} EGP\n\n"
    "Current:\n{current_price} EGP\n\n"
    "https://www.amazon.eg/dp/{asin}"
)

PRICE_UPDATE_TEMPLATE = (
    "📈 Price Update\n\n"
    "Product:\n{title}\n\n"
    "Previous:\n{previous_price} EGP\n\n"
    "Current:\n{current_price} EGP\n\n"
    "Change:\n{change} EGP"
)

UNAVAILABLE_TEMPLATE = "⚠️ Product Unavailable\n\nProduct:\n{title}\n\nThis item is no longer available or in stock."

BACK_IN_STOCK_TEMPLATE = "✅ Back In Stock\n\nProduct:\n{title}\n\nCurrent:\n{current_price} EGP"


async def run_check_cycle(bot: Bot) -> None:
    """Fetch the current price for every active tracked product and run two
    fully independent notification checks per product:

    1. General price-change notification (_maybe_send_price_change_alert) --
       fires on ANY price movement (even 1 EGP) or an available/unavailable
       transition, regardless of the target price. Depends only on the
       product's own last-seen state, never on deal dedup.
    2. Target-reached alert (_send_target_alert) -- unchanged existing
       behavior: fires once when the price is at or below the user's target
       and a notification hasn't already gone out for that exact price.
       Both checks can fire in the same cycle for the same product.

    One product's fetch failure never stops the rest of the cycle, and
    never changes that product's stored state (so it's retried next cycle).
    """
    products = database.get_all_active_products_with_owner()
    logger.info("check cycle starting: %d active product(s)", len(products))

    for product, owner_telegram_id in products:
        # Snapshot pre-cycle state — the `product` dataclass is never
        # re-fetched mid-loop, so these stay stable across both checks
        # below regardless of what gets written to the DB in between.
        previous_price = product.current_price
        previous_available = product.available
        previous_notified_price = product.last_notified_price

        try:
            _, current_price = await fetch_product(product.url)
            is_available = True
        except PriceNotFoundError:
            # Page loaded (title present) but no price element found --
            # treated as the product being unavailable/out of stock, not a
            # transient fetch failure.
            current_price = None
            is_available = False
        except ProductFetchError as exc:
            # Genuinely could not retrieve the page (timeout, CAPTCHA, ...).
            # Per spec this is ignored entirely: no state change, no
            # notification -- just retried next cycle.
            logger.warning(
                "check cycle: fetch failed for product #%d (%s): %s", product.id, product.asin, exc
            )
            continue

        database.update_price_check(product.id, current_price, available=is_available)

        await _maybe_send_price_change_alert(
            bot, owner_telegram_id, product, previous_price, previous_available, current_price, is_available
        )

        if is_available and current_price is not None and current_price <= product.target_price:
            already_notified = current_price == previous_notified_price
            if not already_notified:
                await _send_target_alert(bot, owner_telegram_id, product, current_price)
                database.mark_notified(product.id, current_price)

    logger.info("check cycle complete")
    health.record_check_cycle_complete()
    health.write_health_file()

    expired = dedup.cleanup_expired()
    if expired:
        logger.info("duplicate-deals cache: purged %d expired record(s)", expired)


async def _maybe_send_price_change_alert(
    bot: Bot, telegram_id: int, product: TrackedProduct,
    previous_price: float | None, previous_available: bool,
    current_price: float | None, is_available: bool,
) -> None:
    """Independent of target alerts and of deal dedup — fires on any
    genuine state change: a price movement of any size (while available
    both before and after), or an available<->unavailable transition.
    Silent when there's no real baseline yet (a product that has never
    been price-checked before), so tracking a new product doesn't
    immediately double up with its "Tracking started!" confirmation —
    this is the only case ignored beyond "price identical" /
    "price could not be retrieved" (the latter never reaches this
    function at all, since a ProductFetchError `continue`s before it's
    called).
    """
    if previous_price is None and previous_available:
        return  # no baseline yet — this cycle establishes it silently

    if is_available and not previous_available:
        text = BACK_IN_STOCK_TEMPLATE.format(
            title=product.title or product.asin, current_price=format_price(current_price)
        )
    elif not is_available and previous_available:
        text = UNAVAILABLE_TEMPLATE.format(title=product.title or product.asin)
    elif is_available and previous_available and previous_price is not None and current_price != previous_price:
        change = current_price - previous_price
        text = PRICE_UPDATE_TEMPLATE.format(
            title=product.title or product.asin,
            previous_price=format_price(previous_price),
            current_price=format_price(current_price),
            change=f"{'+' if change > 0 else ''}{format_price(change)}",
        )
    else:
        return  # identical price, or still unavailable both times — no notification

    await _send_alert(bot, telegram_id, text, product)
    database.mark_notified(product.id, current_price if current_price is not None else previous_price)


async def _send_target_alert(bot: Bot, telegram_id: int, product: TrackedProduct, current_price: float) -> None:
    text = TARGET_ALERT_TEMPLATE.format(
        title=product.title or product.asin,
        target_price=format_price(product.target_price),
        current_price=format_price(current_price),
        asin=product.asin,
    )
    await _send_alert(bot, telegram_id, text, product)


async def _send_alert(bot: Bot, telegram_id: int, text: str, product: TrackedProduct) -> None:
    try:
        await bot.send_message(chat_id=telegram_id, text=text)
        health.record_alert_sent()
    except Exception:
        logger.exception(
            "failed to send alert to telegram_id=%s for product #%d", telegram_id, product.id
        )


async def run_channel_watchdog() -> None:
    """Proactively checks every monitored channel's posting activity against
    its own historical average and logs a WARNING for any anomaly — so a
    silently-dead channel subscription surfaces without anyone having to
    ask for /status first.
    """
    channels = channels_store.get_effective_channels()
    if channels:
        watchdog.check_all_channels(channels)


async def send_daily_heartbeat(bot: Bot) -> None:
    """Sends the same snapshot /status shows, once a day, so the admin knows
    the bot is alive without having to ask.
    """
    if ADMIN_TELEGRAM_ID == 0:
        return
    try:
        await bot.send_message(chat_id=ADMIN_TELEGRAM_ID, text=health.format_status_message())
    except Exception:
        logger.exception("failed to send daily heartbeat to telegram_id=%s", ADMIN_TELEGRAM_ID)


def build_scheduler(bot: Bot) -> AsyncIOScheduler:
    scheduler = AsyncIOScheduler()
    scheduler.add_job(
        run_check_cycle,
        "interval",
        minutes=CHECK_INTERVAL_MINUTES,
        args=[bot],
        id="price_check_cycle",
    )
    scheduler.add_job(
        send_daily_heartbeat,
        CronTrigger(hour=9, minute=0, timezone=_CAIRO_TZ),
        args=[bot],
        id="daily_heartbeat",
    )
    scheduler.add_job(
        run_channel_watchdog,
        "interval",
        minutes=CHANNEL_WATCHDOG_INTERVAL_MINUTES,
        id="channel_watchdog",
    )
    return scheduler
