# Fanzi

Personal Amazon.eg price tracker over Telegram. One bot, one SQLite database,
one Playwright worker, one scheduler — no multi-tenancy, no payments.

Playwright-based scraping here is for personal use only. Keep request rates
modest and use a persistent browser profile (Milestone 2+). Amazon's
Conditions of Use prohibit scraping for a commercial product — don't turn
this into a paid product without switching to a compliant data source
(PA-API or a licensed provider) first.

## Setup

```
python -m pip install -r requirements.txt
python -m playwright install chromium
copy .env.example .env   # then fill in TELEGRAM_BOT_TOKEN from @BotFather
python bot.py
```

`python bot.py` is the only command needed for normal operation — it starts the bot, the
price-check scheduler, and the deal listener together in one process.

**One-time exception:** if you're enabling the deal listener (`TELETHON_API_ID`/`DEAL_CHANNELS`
below) for the first time, there's no Telegram session yet, so `bot.py` will log a warning and
skip starting the listener. Run `python listener\watcher.py` once, complete the interactive phone
number + OTP login it prompts for, then stop it (Ctrl+C) and go back to running `python bot.py` —
the saved session file lets it start silently from then on.

## Configuration

All settings come from `.env` (copy `.env.example`) — see that file for the full list with
where-to-get-it comments. Summary:

| Variable | Required | Purpose |
|---|---|---|
| `TELEGRAM_BOT_TOKEN` | yes | The bot's Telegram API token, from @BotFather. |
| `DATABASE_PATH` | no (default `fanzi.db`) | SQLite file path. |
| `CHECK_INTERVAL_MINUTES` | no (default `60`) | How often the scheduler re-checks tracked prices. |
| `ADMIN_TELEGRAM_ID` | no | Your Telegram user ID; restricts `/checkall`. |
| `TELETHON_API_ID` / `TELETHON_API_HASH` | no | From my.telegram.org — enables the deal listener. Leave unset to disable it entirely. |
| `TELETHON_SESSION_NAME` | no (default `fanzi_listener`) | Telethon session file name. |
| `DEAL_CHANNELS` | no | Comma-separated public channel usernames the listener watches. |
| `GEMINI_API_KEY` | no | Google Gemini API key (free tier) used to grade forwarded deals. |
| `MIN_DEAL_QUALITY` | no (default `good`) | Minimum verdict (`good` or `great`) to forward/auto-track a deal. |
| `RATE_LIMIT_PER_MIN` | no (default `12`) | Max Gemini calls per minute; extra requests wait for a slot. |
| `DAILY_ANALYSIS_CAP` | no (default `1400`) | Max Gemini calls per day; resets at local midnight. |
| `MIN_DISCOUNT_FOR_ANALYSIS` | no (default `10`) | Posts with a lower detected discount skip Gemini entirely. |
| `DUPLICATE_WINDOW_HOURS` | no (default `24`) | How long the same product from the same channel is suppressed as a duplicate. |

## Status

- Milestone 1: scaffold, database, `/start`, `/track` (price fetch mocked), `/mytracks`, `/remove`.
- Milestone 2: real Playwright price/title fetching (persistent browser profile in
  `playwright_profile/`, retry+backoff, specific fetch-failure exceptions), wired into `/track`
  and `/mytracks`.
- Milestone 3: APScheduler background job (`CHECK_INTERVAL_MINUTES`, default 60) that checks all
  active products, updates prices, and sends dedup'd price-drop alerts. `/checkall` (restricted to
  `ADMIN_TELEGRAM_ID`) triggers a cycle manually for testing.
- Deal aggregator: a Telethon-based listener (`listener/`) watches public Amazon-deal Telegram
  channels, parses each post (Arabic/English, several link formats), gets a Gemini-generated quality
  verdict, auto-tracks qualifying deals, and forwards a verdict summary to `ADMIN_TELEGRAM_ID`.
  Runs invisibly as a background task inside `bot.py` — see Setup above for first-time login.
  Gemini calls are quota-managed (`listener/analyzer.py`'s `QuotaGuard`): rate-limited per minute,
  capped per day, and skipped for low-discount or duplicate posts to stay within the free tier.
