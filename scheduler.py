"""APScheduler background jobs: periodic price checks + price-drop alerts."""

from __future__ import annotations

import logging

from zoneinfo import ZoneInfo

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from telegram import Bot

import database
import health
from amazon.tracker import ProductFetchError, fetch_product, format_price
from config import ADMIN_TELEGRAM_ID, CHANNEL_WATCHDOG_INTERVAL_MINUTES, CHECK_INTERVAL_MINUTES
from listener import channels_store, dedup, watchdog
from models.tracked_product import TrackedProduct

_CAIRO_TZ = ZoneInfo("Africa/Cairo")

logger = logging.getLogger("fanzi.scheduler")

ALERT_TEMPLATE = (
    "🔥 Price Drop!\n"
    "{title}\n\n"
    "Current: {current_price} EGP\n"
    "Target: {target_price} EGP ✅\n\n"
    "https://www.amazon.eg/dp/{asin}"
)


async def run_check_cycle(bot: Bot) -> None:
    """Fetch the current price for every active tracked product, update the
    DB, and send a dedup'd alert if the target has been hit. One product's
    fetch failure is logged and skipped — it never stops the rest of the
    cycle.
    """
    products = database.get_all_active_products_with_owner()
    logger.info("check cycle starting: %d active product(s)", len(products))

    for product, owner_telegram_id in products:
        try:
            _, current_price = await fetch_product(product.url)
        except ProductFetchError as exc:
            logger.warning(
                "check cycle: fetch failed for product #%d (%s): %s", product.id, product.asin, exc
            )
            continue

        database.update_price_check(product.id, current_price)

        hit_target = current_price <= product.target_price
        already_notified = current_price == product.last_notified_price
        if hit_target and not already_notified:
            await _send_alert(bot, owner_telegram_id, product, current_price)
            database.mark_notified(product.id, current_price)

    logger.info("check cycle complete")
    health.record_check_cycle_complete()
    health.write_health_file()

    expired = dedup.cleanup_expired()
    if expired:
        logger.info("duplicate-deals cache: purged %d expired record(s)", expired)


async def _send_alert(
    bot: Bot, telegram_id: int, product: TrackedProduct, current_price: float
) -> None:
    message = ALERT_TEMPLATE.format(
        title=product.title or product.asin,
        current_price=format_price(current_price),
        target_price=format_price(product.target_price),
        asin=product.asin,
    )
    try:
        await bot.send_message(chat_id=telegram_id, text=message)
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
