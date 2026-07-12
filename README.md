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
copy .env.example .env   # then fill in TELEGRAM_BOT_TOKEN from @BotFather
python bot.py
```

## Status

- Milestone 1: scaffold, database, `/start`, `/track` (price fetch mocked), `/mytracks`, `/remove`.
