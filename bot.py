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

from telegram import BotCommand, BotCommandScopeChat, InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    ConversationHandler,
    MessageHandler,
    filters,
)

import database
import health
import scheduler as scheduler_module
from amazon.parser import extract_asin, normalize_product_url
from amazon.tracker import ProductFetchError, fetch_product, format_price
from config import ADMIN_TELEGRAM_ID, TELEGRAM_BOT_TOKEN
from listener import pending_deals
from listener import watcher as listener_watcher
from listener.watcher import start_background_listener

PUBLIC_COMMANDS = [
    BotCommand("start", "Welcome & quick start guide"),
    BotCommand("track", "Track a new Amazon.eg product"),
    BotCommand("mytracks", "View your tracked products"),
    BotCommand("remove", "Remove a tracked product"),
    BotCommand("pause", "Pause tracking a product"),
    BotCommand("resume", "Resume tracking a product"),
    BotCommand("help", "How to use Fanzi"),
]

ADMIN_ONLY_COMMANDS = [
    BotCommand("status", "Bot health & stats"),
    BotCommand("checkall", "Trigger a manual price check"),
    BotCommand("channels", "List watched deal channels"),
    BotCommand("addchannel", "Add a new deal channel"),
    BotCommand("removechannel", "Remove a deal channel"),
]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
# httpx logs full request URLs at INFO, which embeds the bot token
# (https://api.telegram.org/bot<TOKEN>/...) — keep it at WARNING so the
# token never lands in a log file.
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger("fanzi.bot")

WAITING_URL, WAITING_TARGET = range(2)

TARGET_PERCENTAGE_OPTIONS = (10, 20, 25)


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


def _target_choice_keyboard(asin: str) -> InlineKeyboardMarkup:
    rows = [
        [InlineKeyboardButton(f"{pct}% below current price", callback_data=f"target:{asin}:{pct}")]
        for pct in TARGET_PERCENTAGE_OPTIONS
    ]
    rows.append([InlineKeyboardButton("Custom price", callback_data=f"target:{asin}:custom")])
    rows.append([InlineKeyboardButton("Cancel", callback_data=f"target:{asin}:cancel")])
    return InlineKeyboardMarkup(rows)


async def track_button_pressed(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handler for the 📉 Track Price button on a forwarded deal. Does not
    create a tracker yet — first checks for an existing one, then asks the
    user to choose a target price.

    Deliberately NOT a ConversationHandler: deals are forwarded continuously
    and the admin may click Track Price on a new deal before finishing the
    target-price flow for an earlier one. A ConversationHandler only tracks
    one active state per (chat, user) — a second "track:" button press
    while still "in conversation" for the first deal would silently match
    no state handler and do nothing. Using plain handlers plus pending_deals
    (keyed by ASIN, not by conversation) keeps every button independently
    responsive regardless of what other deals are mid-flow.
    """
    query = update.callback_query
    await query.answer()
    asin = query.data.split(":", 1)[1]

    user = database.get_or_create_user(update.effective_user.id, update.effective_user.username)
    if database.get_tracked_product_by_asin(user.id, asin) is not None:
        await query.message.reply_text("✅ You're already tracking this product.")
        return

    deal = pending_deals.get(asin)
    if deal is None:
        await query.message.reply_text(
            "This deal has expired — use /track to add this product manually."
        )
        return

    await query.message.reply_text(
        f"Current price: {format_price(deal['price'])} EGP\n\nChoose a target price:",
        reply_markup=_target_choice_keyboard(asin),
    )


async def _create_tracker_and_confirm(
    update: Update, context: ContextTypes.DEFAULT_TYPE, asin: str, deal: dict, target_price: float
) -> None:
    user = database.get_or_create_user(update.effective_user.id, update.effective_user.username)
    database.add_tracked_product(
        user_id=user.id,
        asin=asin,
        title=deal["title"],
        url=deal["url"],
        current_price=deal["price"],
        target_price=target_price,
    )
    pending_deals.pop(asin)
    if context.user_data.get("awaiting_custom_target_asin") == asin:
        context.user_data.pop("awaiting_custom_target_asin", None)

    message = update.callback_query.message if update.callback_query else update.message
    await message.reply_text(
        f"✅ Tracking started!\n\n{deal['title']}\n"
        f"Current: {format_price(deal['price'])} EGP\n"
        f"Target: {format_price(target_price)} EGP"
    )


async def target_percentage_chosen(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    _, asin, pct_str = query.data.split(":")

    deal = pending_deals.get(asin)
    if deal is None:
        await query.message.reply_text(
            "This deal has expired — use /track to add this product manually."
        )
        return

    target_price = deal["price"] * (1 - int(pct_str) / 100)
    await _create_tracker_and_confirm(update, context, asin, deal, target_price)


async def target_custom_prompt(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    asin = query.data.split(":")[1]
    context.user_data["awaiting_custom_target_asin"] = asin
    await query.message.reply_text("Send the target price as a number, e.g. 1500")


async def maybe_custom_target_received(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Only acts when the user has an outstanding "Custom price" prompt
    (set by target_custom_prompt above) — otherwise silently ignores the
    text message so it doesn't interfere with /track's own conversation or
    any other plain text.
    """
    asin = context.user_data.get("awaiting_custom_target_asin")
    if asin is None:
        return

    try:
        target_price = float(update.message.text.strip().replace(",", ""))
        if target_price <= 0:
            raise ValueError
    except ValueError:
        await update.message.reply_text("That doesn't look like a valid price. Send a number, e.g. 1500")
        return

    deal = pending_deals.get(asin)
    if deal is None:
        context.user_data.pop("awaiting_custom_target_asin", None)
        await update.message.reply_text(
            "This deal has expired — use /track to add this product manually."
        )
        return

    await _create_tracker_and_confirm(update, context, asin, deal, target_price)


async def target_cancelled(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    asin = query.data.split(":")[1]
    if context.user_data.get("awaiting_custom_target_asin") == asin:
        context.user_data.pop("awaiting_custom_target_asin", None)
    await query.message.reply_text("Cancelled — nothing was tracked.")


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


async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Fanzi tracks Amazon.eg product prices and alerts you when they drop.\n\n"
        "/track — start tracking a product\n"
        "/mytracks — see what you're tracking\n"
        "/pause [id] — pause a tracked product (stops price checks/alerts)\n"
        "/resume [id] — resume a paused product\n"
        "/remove [id] — stop tracking a product for good\n\n"
        "Use the id shown next to each product in /mytracks."
    )


async def pause(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args:
        await update.message.reply_text("Usage: /pause [id] (see the id from /mytracks)")
        return
    try:
        product_id = int(context.args[0])
    except ValueError:
        await update.message.reply_text("Usage: /pause [id] — id must be a number.")
        return

    user = database.get_or_create_user(update.effective_user.id, update.effective_user.username)
    if database.set_product_active(product_id, user.id, active=False):
        await update.message.reply_text(f"⏸ Paused #{product_id}.")
    else:
        await update.message.reply_text(f"No tracked product #{product_id} found for you.")


async def resume(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args:
        await update.message.reply_text("Usage: /resume [id] (see the id from /mytracks)")
        return
    try:
        product_id = int(context.args[0])
    except ValueError:
        await update.message.reply_text("Usage: /resume [id] — id must be a number.")
        return

    user = database.get_or_create_user(update.effective_user.id, update.effective_user.username)
    if database.set_product_active(product_id, user.id, active=True):
        await update.message.reply_text(f"▶️ Resumed #{product_id}.")
    else:
        await update.message.reply_text(f"No tracked product #{product_id} found for you.")


def _is_admin(update: Update) -> bool:
    return ADMIN_TELEGRAM_ID != 0 and update.effective_user.id == ADMIN_TELEGRAM_ID


async def channels(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_admin(update):
        await update.message.reply_text("This command is restricted.")
        return

    statuses = await listener_watcher.get_channel_statuses()
    if not statuses:
        await update.message.reply_text("The deal listener isn't running — no channels to show.")
        return

    active_count = sum(1 for _, ok in statuses if ok)
    lines = [f"📡 Watched Channels ({active_count}/{len(statuses)} active)", ""]
    for channel_name, ok in statuses:
        lines.append(f"✅ {channel_name}" if ok else f"❌ {channel_name} (failed to join)")
    await update.message.reply_text("\n".join(lines))


async def addchannel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_admin(update):
        await update.message.reply_text("This command is restricted.")
        return
    if not context.args:
        await update.message.reply_text("Usage: /addchannel @username")
        return

    _, reply = await listener_watcher.add_channel_runtime(context.args[0])
    await update.message.reply_text(reply)


async def removechannel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_admin(update):
        await update.message.reply_text("This command is restricted.")
        return
    if not context.args:
        await update.message.reply_text("Usage: /removechannel @username")
        return

    _, reply = await listener_watcher.remove_channel_runtime(context.args[0])
    await update.message.reply_text(reply)


async def checkall(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Manually triggers one scheduler check cycle. Restricted to
    ADMIN_TELEGRAM_ID so it can't be abused if the bot is ever shared.
    """
    if ADMIN_TELEGRAM_ID == 0 or update.effective_user.id != ADMIN_TELEGRAM_ID:
        await update.message.reply_text("This command is restricted.")
        return

    await update.message.reply_text("Running a check cycle now...")
    await scheduler_module.run_check_cycle(context.bot)
    await update.message.reply_text("Check cycle complete.")


async def status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Live health snapshot. Restricted to ADMIN_TELEGRAM_ID, same as /checkall."""
    if ADMIN_TELEGRAM_ID == 0 or update.effective_user.id != ADMIN_TELEGRAM_ID:
        await update.message.reply_text("This command is restricted.")
        return

    await update.message.reply_text(health.format_status_message())


async def _post_init(application: Application) -> None:
    await application.bot.set_my_commands(PUBLIC_COMMANDS)
    if ADMIN_TELEGRAM_ID != 0:
        await application.bot.set_my_commands(
            PUBLIC_COMMANDS + ADMIN_ONLY_COMMANDS,
            scope=BotCommandScopeChat(chat_id=ADMIN_TELEGRAM_ID),
        )
    logger.info(
        "command menu registered (admin scope: %s)", "enabled" if ADMIN_TELEGRAM_ID else "disabled"
    )

    sched = scheduler_module.build_scheduler(application.bot)
    sched.start()
    application.bot_data["scheduler"] = sched
    logger.info("scheduler started (interval-based background job)")

    # Deal listener runs invisibly alongside the bot — any failure here must
    # never prevent the bot itself from starting.
    try:
        telethon_client = await start_background_listener(application.bot)
    except Exception:
        logger.exception("deal listener failed to start — continuing without it")
        telethon_client = None
    application.bot_data["telethon_client"] = telethon_client

    # So health.json exists immediately (with real channel counts), not just
    # after the first price-check cycle.
    health.write_health_file()


async def _post_shutdown(application: Application) -> None:
    sched = application.bot_data.get("scheduler")
    if sched is not None:
        sched.shutdown(wait=False)
        logger.info("scheduler stopped")

    telethon_client = application.bot_data.get("telethon_client")
    if telethon_client is not None:
        await telethon_client.disconnect()
        logger.info("deal listener stopped")


def build_application() -> Application:
    application = (
        Application.builder()
        .token(TELEGRAM_BOT_TOKEN)
        .post_init(_post_init)
        .post_shutdown(_post_shutdown)
        .build()
    )

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

    # Plain handlers, not a ConversationHandler — see the docstring on
    # track_button_pressed for why (deals forward continuously, so more than
    # one Track Price flow can be mid-flight for the same admin at once).
    application.add_handler(CallbackQueryHandler(track_button_pressed, pattern=r"^track:"))
    application.add_handler(CallbackQueryHandler(target_percentage_chosen, pattern=r"^target:[^:]+:(10|20|25)$"))
    application.add_handler(CallbackQueryHandler(target_custom_prompt, pattern=r"^target:[^:]+:custom$"))
    application.add_handler(CallbackQueryHandler(target_cancelled, pattern=r"^target:[^:]+:cancel$"))
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, maybe_custom_target_received), group=1
    )

    application.add_handler(CommandHandler("mytracks", mytracks))
    application.add_handler(CommandHandler("remove", remove))
    application.add_handler(CommandHandler("pause", pause))
    application.add_handler(CommandHandler("resume", resume))
    application.add_handler(CommandHandler("help", help_cmd))
    application.add_handler(CommandHandler("checkall", checkall))
    application.add_handler(CommandHandler("status", status))
    application.add_handler(CommandHandler("channels", channels))
    application.add_handler(CommandHandler("addchannel", addchannel))
    application.add_handler(CommandHandler("removechannel", removechannel))

    return application


def main() -> None:
    database.init_db()
    application = build_application()
    logger.info("Fanzi bot starting (polling)...")
    application.run_polling()


if __name__ == "__main__":
    main()
