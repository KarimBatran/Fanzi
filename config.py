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
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
# From https://console.groq.com/keys (free tier) — automatic fallback when
# Gemini is unavailable/quota-exhausted. Left empty, Groq requests fail fast
# with a FatalProviderError (no key configured) rather than crashing.
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
MIN_DEAL_QUALITY = os.getenv("MIN_DEAL_QUALITY", "good")

# Gemini-specific quota management (listener/ai_providers.py) — keeps the app
# comfortably within the Gemini free tier.
RATE_LIMIT_PER_MIN = int(os.getenv("RATE_LIMIT_PER_MIN", "12"))
DAILY_ANALYSIS_CAP = int(os.getenv("DAILY_ANALYSIS_CAP", "1400"))
MIN_DISCOUNT_FOR_ANALYSIS = int(os.getenv("MIN_DISCOUNT_FOR_ANALYSIS", "10"))
DUPLICATE_WINDOW_HOURS = int(os.getenv("DUPLICATE_WINDOW_HOURS", "24"))

# Retry/circuit-breaker policy shared by both AI providers
# (listener/ai_providers.py) — mechanics only, never provider quota values.
AI_RETRY_COUNT = int(os.getenv("AI_RETRY_COUNT", "3"))
AI_RETRY_INITIAL_BACKOFF_SECONDS = float(os.getenv("AI_RETRY_INITIAL_BACKOFF_SECONDS", "1"))
AI_RETRY_MAX_BACKOFF_SECONDS = float(os.getenv("AI_RETRY_MAX_BACKOFF_SECONDS", "4"))
AI_CIRCUIT_BREAKER_FAILURE_THRESHOLD = int(os.getenv("AI_CIRCUIT_BREAKER_FAILURE_THRESHOLD", "5"))
AI_CIRCUIT_BREAKER_COOLDOWN_SECONDS = int(os.getenv("AI_CIRCUIT_BREAKER_COOLDOWN_MINUTES", "15")) * 60
# How often the background recovery job checks for an expired provider
# cooldown and sends a probe proactively, instead of waiting for the next
# real deal to trigger it.
AI_BACKGROUND_RECOVERY_INTERVAL_SECONDS = int(os.getenv("AI_BACKGROUND_RECOVERY_INTERVAL_SECONDS", "60"))

# Self-improving knowledge engine (listener/learning.py) — thresholds only,
# never a substitute for calling Gemini/Groq; see that module's docstring.
BRANDS_FILE = os.getenv("BRANDS_FILE", "data/brands.json")
RULE_MIN_SAMPLES = int(os.getenv("RULE_MIN_SAMPLES", "5"))
RULE_BRAND_CONFIDENCE = float(os.getenv("RULE_BRAND_CONFIDENCE", "0.80"))
RULE_BRAND_CATEGORY_CONFIDENCE = float(os.getenv("RULE_BRAND_CATEGORY_CONFIDENCE", "0.85"))
RULE_PRICE_CONFIDENCE = float(os.getenv("RULE_PRICE_CONFIDENCE", "0.80"))
RULE_DISCOUNT_CONFIDENCE = float(os.getenv("RULE_DISCOUNT_CONFIDENCE", "0.75"))
RULE_MONTHLY_DECAY = float(os.getenv("RULE_MONTHLY_DECAY", "0.98"))
RULE_VALIDATION_RATE_HIGH = float(os.getenv("RULE_VALIDATION_RATE_HIGH", "0.02"))
RULE_VALIDATION_RATE_MEDIUM = float(os.getenv("RULE_VALIDATION_RATE_MEDIUM", "0.10"))
RULE_OUTLIER_DISCOUNT = int(os.getenv("RULE_OUTLIER_DISCOUNT", "50"))
# Not explicitly named in the original spec's config list, but required by
# its own "extremely low price" outlier condition — same env-driven pattern.
RULE_OUTLIER_MIN_PRICE = float(os.getenv("RULE_OUTLIER_MIN_PRICE", "30"))

# Forwarding-pipeline performance instrumentation (listener/timing.py,
# listener/watcher.py). PERFORMANCE_LOGGING gates the per-deal timing
# summary (negligible cost either way — perf_counter() calls are cheap —
# but keeps normal logs quiet if turned off); the slow-request WARNING
# always fires regardless, so outliers stay visible either way.
PERFORMANCE_LOGGING = os.getenv("PERFORMANCE_LOGGING", "true").lower() == "true"
SLOW_REQUEST_THRESHOLD_SECONDS = float(os.getenv("SLOW_REQUEST_THRESHOLD_SECONDS", "3.0"))

# AI soft timeout (listener/watcher.py): if analysis hasn't completed within
# this long, the deal is forwarded immediately with a placeholder verdict and
# the AI call continues in the background, editing the message in place once
# it finishes. Set AI_SOFT_TIMEOUT_ENABLED=false to always wait for AI
# (the old behavior).
AI_SOFT_TIMEOUT_ENABLED = os.getenv("AI_SOFT_TIMEOUT_ENABLED", "true").lower() == "true"
AI_SOFT_TIMEOUT_SECONDS = float(os.getenv("AI_SOFT_TIMEOUT_SECONDS", "2.5"))

# Short-TTL cache for resolved shortener/link.amazon redirects
# (listener/parser.py) — the same deal link is often reposted/crossposted
# across channels within minutes; caching avoids a redundant network
# round-trip for an identical URL seen again inside the TTL window.
REDIRECT_CACHE_TTL_SECONDS = float(os.getenv("REDIRECT_CACHE_TTL_SECONDS", "300"))

# Channel health watchdog (listener/watchdog.py) — how often the scheduler
# proactively checks every monitored channel's posting activity.
CHANNEL_WATCHDOG_INTERVAL_MINUTES = int(os.getenv("CHANNEL_WATCHDOG_INTERVAL_MINUTES", "15"))

# Automatic message replay (listener/replay.py) — recovers deals missed
# during downtime/disconnects by fetching recent channel history and
# replaying anything newer than the last successfully processed message ID.
REPLAY_FETCH_LIMIT = int(os.getenv("REPLAY_FETCH_LIMIT", "50"))
# How often the reconnect watcher polls the Telethon client's connection
# state to detect a disconnect -> reconnect transition.
REPLAY_RECONNECT_POLL_SECONDS = float(os.getenv("REPLAY_RECONNECT_POLL_SECONDS", "10"))
