"""Telegram commands and handlers."""

from __future__ import annotations

import logging
import os
import sys

# This machine's Python is a portable/embeddable install whose python312._pth
# controls sys.path and doesn't add the script's own directory automatically
# (unlike a normal CPython install) — so local imports (database, amazon.*,
# models.*) fail without this. Harmless no-op on a standard Python install.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    ConversationHandler,
    MessageHandler,
    filters,
)

import database
from amazon.parser import extract_asin, normalize_product_url
from amazon.tracker import ProductFetchError, fetch_product, format_price
from config import TELEGRAM_BOT_TOKEN

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
# httpx logs full request URLs at INFO, which embeds the bot token
# (https://api.telegram.org/bot<TOKEN>/...) — keep it at WARNING so the
# token never lands in a log file.
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger("fanzi.bot")

WAITING_URL, WAITING_TARGET = range(2)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    database.get_or_create_user(update.effective_user.id, update.effective_user.username)
    await update.message.reply_text(
        "Welcome to Fanzi \U0001f9de\n\n"
        "I track Amazon.eg product prices and let you know when they drop "
        "below a target you set.\n\n"
        "/track — start tracking a product\n"
        "/mytracks — see what you're tracking\n"
        "/remove [id] — stop tracking a product"
    )


async def track_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text(
        "Send me the Amazon.eg product link (or just the ASIN). /cancel to stop."
    )
    return WAITING_URL


async def track_receive_url(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    asin = await extract_asin(update.message.text)
    if asin is None:
        await update.message.reply_text(
            "I couldn't find a valid ASIN in that. Send an amazon.eg product "
            "link (/dp/... or /gp/product/...) or a bare 10-character ASIN."
        )
        return WAITING_URL

    url = normalize_product_url(asin)
    await update.message.reply_text("Got it — fetching current price...")

    try:
        title, price = await fetch_product(url)
    except ProductFetchError as exc:
        logger.warning("product fetch failed for %s: %s", asin, exc)
        await update.message.reply_text(
            "Couldn't fetch that product right now — try again in a bit."
        )
        context.user_data.clear()
        return ConversationHandler.END

    context.user_data["asin"] = asin
    context.user_data["url"] = url
    context.user_data["title"] = title
    context.user_data["current_price"] = price

    await update.message.reply_text(
        f"Current price: {format_price(price)} EGP — notify below what price?"
    )
    return WAITING_TARGET


async def track_receive_target(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    try:
        target_price = float(update.message.text.strip().replace(",", ""))
        if target_price <= 0:
            raise ValueError
    except ValueError:
        await update.message.reply_text("That doesn't look like a valid price. Send a number, e.g. 1500")
        return WAITING_TARGET

    user = database.get_or_create_user(update.effective_user.id, update.effective_user.username)
    database.add_tracked_product(
        user_id=user.id,
        asin=context.user_data["asin"],
        title=context.user_data["title"],
        url=context.user_data["url"],
        current_price=context.user_data["current_price"],
        target_price=target_price,
    )

    await update.message.reply_text(
        f"✅ Tracking started!\nYou'll be notified when the price drops below {target_price:g} EGP."
    )
    context.user_data.clear()
    return ConversationHandler.END


async def track_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    context.user_data.clear()
    await update.message.reply_text("Cancelled.")
    return ConversationHandler.END


async def mytracks(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = database.get_or_create_user(update.effective_user.id, update.effective_user.username)
    products = database.get_active_products(user.id)

    if not products:
        await update.message.reply_text("You're not tracking anything yet. Use /track to start.")
        return

    lines = []
    for p in products:
        title = p.title or "(title not fetched yet)"
        price = f"{format_price(p.current_price)} {p.currency}" if p.current_price is not None else "not fetched yet"
        lines.append(
            f"#{p.id} — {p.asin}\n{title}\n"
            f"Current: {price} | Target: {format_price(p.target_price)} {p.currency}"
        )

    await update.message.reply_text("\n\n".join(lines))


async def remove(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args:
        await update.message.reply_text("Usage: /remove [id] (see the id from /mytracks)")
        return
    try:
        product_id = int(context.args[0])
    except ValueError:
        await update.message.reply_text("Usage: /remove [id] — id must be a number.")
        return

    user = database.get_or_create_user(update.effective_user.id, update.effective_user.username)
    removed = database.remove_product(product_id, user.id)
    if removed:
        await update.message.reply_text(f"Removed #{product_id}.")
    else:
        await update.message.reply_text(f"No tracked product #{product_id} found for you.")


def build_application() -> Application:
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    track_conversation = ConversationHandler(
        entry_points=[CommandHandler("track", track_start)],
        states={
            WAITING_URL: [MessageHandler(filters.TEXT & ~filters.COMMAND, track_receive_url)],
            WAITING_TARGET: [MessageHandler(filters.TEXT & ~filters.COMMAND, track_receive_target)],
        },
        fallbacks=[CommandHandler("cancel", track_cancel)],
    )

    application.add_handler(CommandHandler("start", start))
    application.add_handler(track_conversation)
    application.add_handler(CommandHandler("mytracks", mytracks))
    application.add_handler(CommandHandler("remove", remove))

    return application


def main() -> None:
    database.init_db()
    application = build_application()
    logger.info("Fanzi bot starting (polling)...")
    application.run_polling()


if __name__ == "__main__":
    main()
