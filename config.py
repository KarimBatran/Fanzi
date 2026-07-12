"""Env var loading. Every secret/setting comes from .env — nothing hardcoded."""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
DATABASE_PATH = os.getenv("DATABASE_PATH", "fanzi.db")
CHECK_INTERVAL_MINUTES = int(os.getenv("CHECK_INTERVAL_MINUTES", "45"))
