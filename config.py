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

# Appended to every generated canonical product URL (amazon.parser.
# normalize_product_url) when non-empty. Never taken from the source
# channel's own link — only ever this configured tag.
AMAZON_AFFILIATE_TAG = os.getenv("AMAZON_AFFILIATE_TAG", "")

# Product Family detection (listener/family.py) — matches related Amazon
# ASINs (color/size/capacity/pack variants) so the bot can tell "same family,
# same variant" (suppress) apart from "same family, genuinely better/other
# variant" (still notify). FUZZY_MATCH_THRESHOLD: normalized-title similarity
# at or above this is treated as the same family without ever asking AI.
# AI_MATCH_FLOOR: below this, treated as a different family without asking
# AI either — AI is only consulted in the ambiguous band between the two.
FAMILY_FUZZY_MATCH_THRESHOLD = float(os.getenv("FAMILY_FUZZY_MATCH_THRESHOLD", "0.82"))
FAMILY_AI_MATCH_FLOOR = float(os.getenv("FAMILY_AI_MATCH_FLOOR", "0.55"))
# True-duplicate suppression tolerance: a repost of the same family + same
# variant attributes is only ever suppressed if price/discount are also
# within these tolerances (and within DUPLICATE_WINDOW_HOURS) — otherwise
# it's a genuine price/discount change worth notifying about.
FAMILY_DUPLICATE_PRICE_TOLERANCE_EGP = float(os.getenv("FAMILY_DUPLICATE_PRICE_TOLERANCE_EGP", "1.0"))
FAMILY_DUPLICATE_DISCOUNT_TOLERANCE_PERCENT = int(os.getenv("FAMILY_DUPLICATE_DISCOUNT_TOLERANCE_PERCENT", "2"))

# Daily AI budget manager (listener/budget.py) -- paces AI usage across the
# whole day instead of spending until quota exhausts within a few hours.
# DAILY_ANALYSIS_CAP (above) is the total daily budget being paced against;
# previously it was declared but never actually enforced anywhere.
# BUDGET_RESERVE_PERCENT of the daily budget is never spent except by
# Priority 1 deals, so a late-day surge of genuinely important deals is
# never starved by earlier spending.
BUDGET_RESERVE_PERCENT = float(os.getenv("BUDGET_RESERVE_PERCENT", "10"))
# Priority 3 ("usually skip") is only allowed to spend AI when the remaining
# budget is comfortably above the reserve -- this multiple of reserve_floor.
PRIORITY3_HEALTHY_BUDGET_MULTIPLIER = float(os.getenv("PRIORITY3_HEALTHY_BUDGET_MULTIPLIER", "3"))
# Priority-tier discount thresholds (percent).
PRIORITY1_DISCOUNT_THRESHOLD = int(os.getenv("PRIORITY1_DISCOUNT_THRESHOLD", "40"))
PRIORITY2_DISCOUNT_THRESHOLD = int(os.getenv("PRIORITY2_DISCOUNT_THRESHOLD", "20"))
# A learned rule below this confidence is always treated as Priority 1
# ("learned confidence below threshold") -- mirrors learning.py's own
# always-call-AI floor (see learning._confidence_band_ai_probability).
PRIORITY1_RULE_CONFIDENCE_FLOOR = float(os.getenv("PRIORITY1_RULE_CONFIDENCE_FLOOR", "0.70"))
# A rule at/above this confidence is "well understood" -> Priority 3;
# between the floor above and this value -> Priority 2 ("moderate confidence").
PRIORITY3_RULE_CONFIDENCE_CEILING = float(os.getenv("PRIORITY3_RULE_CONFIDENCE_CEILING", "0.90"))
# Floor for the budget-adaptive validation-probability multiplier (see
# learning.decide()'s validation_multiplier param) -- even at a fully
# exhausted budget, a tiny chance of AI validation is kept so the knowledge
# base never goes completely stale.
BUDGET_MIN_VALIDATION_MULTIPLIER = float(os.getenv("BUDGET_MIN_VALIDATION_MULTIPLIER", "0.05"))

# Product Family AI-verdict cache (listener/family.py) -- reuses a family's
# most recent real AI verdict for a new variant instead of spending another
# AI call, unless price/discount moved significantly or a genuinely new kind
# of variant attribute appeared.
FAMILY_VERDICT_CACHE_WINDOW_HOURS = float(os.getenv("FAMILY_VERDICT_CACHE_WINDOW_HOURS", "12"))
FAMILY_VERDICT_PRICE_CHANGE_THRESHOLD_PERCENT = float(os.getenv("FAMILY_VERDICT_PRICE_CHANGE_THRESHOLD_PERCENT", "10"))
FAMILY_VERDICT_DISCOUNT_CHANGE_THRESHOLD_PERCENT = float(
    os.getenv("FAMILY_VERDICT_DISCOUNT_CHANGE_THRESHOLD_PERCENT", "10")
)
# Once a family has this many known variants, only every Nth new variant (or
# one per SMART_SAMPLING_INTERVAL_HOURS, whichever comes first) forces a
# real AI call even if the verdict cache would otherwise apply -- keeps a
# well-learned family from drifting silently out of date.
SMART_SAMPLING_VARIANT_THRESHOLD = int(os.getenv("SMART_SAMPLING_VARIANT_THRESHOLD", "10"))
SMART_SAMPLING_EVERY_N_VARIANTS = int(os.getenv("SMART_SAMPLING_EVERY_N_VARIANTS", "10"))
SMART_SAMPLING_INTERVAL_HOURS = float(os.getenv("SMART_SAMPLING_INTERVAL_HOURS", "6"))

# Value Score engine (listener/scoring.py + listener/budget.py). When
# SCORE_ENGINE_ENABLED is false (the shipping default), priority
# classification behaves byte-for-byte identically to before this engine
# existed (classify_priority_legacy); the score engine still runs in shadow
# mode -- computing and logging what it *would* have decided -- whenever
# SCORE_ENGINE_LOG_VERBOSE is true, so it can be validated against real
# traffic before ever being flipped on. See docs/score_engine_rollout.md.
SCORE_ENGINE_ENABLED = os.getenv("SCORE_ENGINE_ENABLED", "false").lower() == "true"
SCORE_ENGINE_LOG_VERBOSE = os.getenv("SCORE_ENGINE_LOG_VERBOSE", "true").lower() == "true"
# Component weights for the combined 0-100 Value Score -- must sum to 1.0
# (validated at import in listener/scoring.py). rarity = how little price
# history exists for this ASIN/family (a rarely-seen product is worth a
# fresh look more than one observed dozens of times).
SCORE_WEIGHT_BRAND = float(os.getenv("SCORE_WEIGHT_BRAND", "0.25"))
SCORE_WEIGHT_PRICE_PCTL = float(os.getenv("SCORE_WEIGHT_PRICE_PCTL", "0.25"))
SCORE_WEIGHT_FAMILY_PCTL = float(os.getenv("SCORE_WEIGHT_FAMILY_PCTL", "0.15"))
SCORE_WEIGHT_CATEGORY_DEV = float(os.getenv("SCORE_WEIGHT_CATEGORY_DEV", "0.15"))
SCORE_WEIGHT_RARITY = float(os.getenv("SCORE_WEIGHT_RARITY", "0.20"))
