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

# Deal listener (listener/watcher.py) — watches public Telegram deal channels
# via a separate Telethon user session (not the bot API).
TELETHON_API_ID = int(os.getenv("TELETHON_API_ID", "0"))
TELETHON_API_HASH = os.getenv("TELETHON_API_HASH", "")
TELETHON_SESSION_NAME = os.getenv("TELETHON_SESSION_NAME", "fanzi_listener")
DEAL_CHANNELS = [c.strip() for c in os.getenv("DEAL_CHANNELS", "").split(",") if c.strip()]
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
MIN_DEAL_QUALITY = os.getenv("MIN_DEAL_QUALITY", "good")
