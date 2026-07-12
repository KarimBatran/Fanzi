"""Env var loading. Every secret/setting comes from .env — nothing hardcoded."""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
DATABASE_PATH = os.getenv("DATABASE_PATH", "fanzi.db")
CHECK_INTERVAL_MINUTES = int(os.getenv("CHECK_INTERVAL_MINUTES", "60"))

# Your own Telegram user ID (message @userinfobot to get it). Restricts
# /checkall so it can't be abused if the bot is ever shared. 0 = disabled.
ADMIN_TELEGRAM_ID = int(os.getenv("ADMIN_TELEGRAM_ID", "0"))
