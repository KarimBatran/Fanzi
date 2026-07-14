"""Shared in-memory health/heartbeat state — powers /status and the
health.json heartbeat file. Updated from bot.py (uptime baseline),
scheduler.py (check-cycle completion, alerts), and listener/watcher.py
(deals analyzed, channel connectivity).
"""

from __future__ import annotations

import json
import os
import time
from datetime import date, datetime

import database
from config import CHECK_INTERVAL_MINUTES
from listener import channels_store, dedup, learning, replay, watchdog
from listener.ai_providers import get_manager

# A channel showing this many combined parse failures (no price/no ASIN/
# non-Amazon link) today is flagged even if it's still posting normally —
# distinct from the watchdog's silence-based anomaly.
_PARSER_FAILURE_WARNING_THRESHOLD = 5

HEALTH_FILE_PATH = "health.json"

_start_time = time.time()
_last_check: datetime | None = None
_channels_configured = 0
_channels_active = 0
_deals_analyzed_today = 0
_alerts_sent_today = 0
_duplicates_skipped_today = 0
_counters_date = date.today()


def _reset_counters_if_new_day() -> None:
    global _counters_date, _deals_analyzed_today, _alerts_sent_today, _duplicates_skipped_today
    today = date.today()
    if today != _counters_date:
        _counters_date = today
        _deals_analyzed_today = 0
        _alerts_sent_today = 0
        _duplicates_skipped_today = 0


def record_deal_analyzed() -> None:
    global _deals_analyzed_today
    _reset_counters_if_new_day()
    _deals_analyzed_today += 1


def record_alert_sent() -> None:
    global _alerts_sent_today
    _reset_counters_if_new_day()
    _alerts_sent_today += 1


def record_duplicate_skipped() -> None:
    global _duplicates_skipped_today
    _reset_counters_if_new_day()
    _duplicates_skipped_today += 1


def set_channels_status(active: int, configured: int) -> None:
    global _channels_active, _channels_configured
    _channels_active = active
    _channels_configured = configured


def record_check_cycle_complete() -> None:
    global _last_check
    _last_check = datetime.now()


def _uptime_seconds() -> int:
    return int(time.time() - _start_time)


def _format_uptime(seconds: int) -> str:
    hours, remainder = divmod(seconds, 3600)
    minutes = remainder // 60
    return f"{hours}h {minutes}m"


def _is_delayed() -> bool:
    if _last_check is None:
        return False
    threshold_seconds = 2 * CHECK_INTERVAL_MINUTES * 60
    return (datetime.now() - _last_check).total_seconds() > threshold_seconds


def get_channel_health() -> list[dict]:
    """Per-channel health for /status and the Phase 1 audit table — built
    entirely from real persisted counters (database.channel_stats /
    channel_last_post), never estimated.
    """
    stat_date = date.today().isoformat()
    results = []
    for channel in channels_store.get_effective_channels():
        stats = database.get_channel_stats(channel, stat_date)
        wd = watchdog.check_channel(channel)
        results.append(
            {
                "channel": channel,
                "posts": stats["posts_received"],
                "parsed": stats["parsed"],
                "forwarded": stats["forwarded"],
                "duplicates": stats["duplicates"],
                "ai_analyses": stats["ai_analyses"],
                "rule_hits": stats["rule_hits"],
                "no_price_failures": stats["no_price_failures"],
                "no_asin_failures": stats["no_asin_failures"],
                "non_amazon_links": stats["non_amazon_links"],
                "total_failures": stats["total_failures"],
                "last_post_at": wd.last_post_at,
                "minutes_since_last_post": wd.minutes_since_last_post,
                "is_silent_anomaly": wd.is_silent_anomaly,
                "warning": wd.warning,
            }
        )
    return results


def build_snapshot() -> dict:
    """Full snapshot used by both /status and health.json."""
    _reset_counters_if_new_day()
    tracked_items = len(database.get_all_active_products_with_owner())
    last_check_seconds_ago = (
        int((datetime.now() - _last_check).total_seconds()) if _last_check is not None else None
    )
    providers = get_manager().status_snapshot()
    learning_snapshot = learning.status_snapshot()
    return {
        "status": "delayed" if _is_delayed() else "ok",
        "uptime_seconds": _uptime_seconds(),
        "last_check": _last_check.isoformat(timespec="seconds") if _last_check is not None else None,
        "last_check_seconds_ago": last_check_seconds_ago,
        "tracked_items": tracked_items,
        "channels_active": _channels_active,
        "channels_configured": _channels_configured,
        "deals_today": _deals_analyzed_today,
        "alerts_today": _alerts_sent_today,
        "duplicates_skipped_today": _duplicates_skipped_today,
        "active_duplicate_entries": dedup.get_active_count(),
        "providers": providers,
        "learning": learning_snapshot,
        "channel_health": get_channel_health(),
        "replay": replay.get_status(),
        "pid": os.getpid(),
    }


def write_health_file() -> None:
    snapshot = build_snapshot()
    payload = {
        "status": snapshot["status"],
        "uptime_seconds": snapshot["uptime_seconds"],
        "last_check": snapshot["last_check"],
        "tracked_items": snapshot["tracked_items"],
        "channels_active": snapshot["channels_active"],
        "deals_today": snapshot["deals_today"],
        "alerts_today": snapshot["alerts_today"],
        "pid": snapshot["pid"],
    }
    with open(HEALTH_FILE_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _format_channel_health(channel_health: list[dict]) -> list[str]:
    if not channel_health:
        return ["(no channels configured)"]

    lines: list[str] = []
    for ch in channel_health:
        lines.append(ch["channel"])
        parser_failures_total = ch["no_price_failures"] + ch["no_asin_failures"] + ch["non_amazon_links"]
        if ch["is_silent_anomaly"]:
            lines.append(f"⚠ {ch['warning']}")
            continue
        if parser_failures_total >= _PARSER_FAILURE_WARNING_THRESHOLD:
            lines.append(f"⚠ Parser failures: {parser_failures_total} today")
        else:
            lines.append("✅ Healthy")
        lines.append(f"Posts: {ch['posts']}")
        lines.append(f"Forwarded: {ch['forwarded']}")
        if ch["minutes_since_last_post"] is not None:
            lines.append(f"Last post: {int(ch['minutes_since_last_post'])} min ago")
    return lines


def format_status_message() -> str:
    snapshot = build_snapshot()
    delayed = snapshot["status"] == "delayed"
    header = "🟡 Fanzi is running (delayed)" if delayed else "🟢 Fanzi is running"

    if snapshot["last_check_seconds_ago"] is not None:
        minutes_ago = max(0, snapshot["last_check_seconds_ago"] // 60)
        last_check_line = f"🔍 Last price check: {minutes_ago} min ago"
    else:
        last_check_line = "🔍 Last price check: not yet run"

    providers = snapshot["providers"]
    learning_snapshot = snapshot["learning"]

    def _timestamp(dt: datetime | None) -> str:
        return dt.strftime("%Y-%m-%d %H:%M:%S") if dt is not None else "Never"

    def _provider_lines(name: str, p: dict) -> list[str]:
        latency = f"{int(p['avg_latency_ms'])} ms" if p["avg_latency_ms"] is not None else "n/a"
        cooldown = (
            f"{int(p['cooldown_remaining_seconds'] // 60)}m {int(p['cooldown_remaining_seconds'] % 60)}s"
            if p["cooldown_remaining_seconds"] is not None
            else "None"
        )
        return [
            name,
            f"Status: {p['status']}",
            f"Calls today: {p['calls_today']}",
            f"Failures: {p['consecutive_failures']}",
            f"Latency: {latency}",
            f"Last success: {_timestamp(p['last_success'])}",
            f"Last failure: {_timestamp(p['last_failure'])}",
            f"Cooldown: {cooldown}",
            f"Quota: {'Available' if p['quota_available'] else 'EXHAUSTED'}",
            f"API Key: {'Yes' if p['api_key_configured'] else 'No'}",
        ]

    lines = [
        header,
        "",
        f"⏱ Uptime: {_format_uptime(snapshot['uptime_seconds'])}",
        f"📦 Tracked items: {snapshot['tracked_items']}",
        last_check_line,
        f"📡 Channels: {snapshot['channels_active']}/{snapshot['channels_configured']} listening",
        f"🤖 Deals analyzed today: {snapshot['deals_today']}",
        f"💸 Alerts sent today: {snapshot['alerts_today']}",
        f"   {snapshot['duplicates_skipped_today']} duplicates skipped today, "
        f"{snapshot['active_duplicate_entries']} active duplicate entries",
        "",
        "🧠 AI Providers",
        *_provider_lines("Gemini", providers["gemini"]),
        *_provider_lines("Groq", providers["groq"]),
        "Current primary provider:",
        providers["current_primary"],
        "Current fallback provider:",
        providers["current_fallback"],
        "Last provider used:",
        providers["last_provider_used"],
        f"Total successful failovers today: {providers['total_failovers_today']}",
        f"Total provider failures today: {providers['total_failures_today']}",
        "",
        f"🧠 Gemini calls today: {providers['gemini']['calls_today']}",
        f"⚡ Groq calls today: {providers['groq']['calls_today']}",
        "📚 Learned rules",
        f"   Brand rules: {learning_snapshot['brand_rules']}",
        f"   Brand+Category rules: {learning_snapshot['brand_category_rules']}",
        f"   Category+Price rules: {learning_snapshot['category_price_rules']}",
        f"   Category+Discount rules: {learning_snapshot['category_discount_rules']}",
        f"🎯 AI calls saved today: {learning_snapshot['ai_calls_saved_today']}",
        f"Learning confidence average: {learning_snapshot['avg_confidence']:.0%}",
        f"Knowledge base version: {learning_snapshot['kb_version']}",
        "",
        "📡 Channels",
        *_format_channel_health(snapshot["channel_health"]),
        "",
        "Replay",
        "Last replay:",
        _timestamp(snapshot["replay"]["last_replay_at"]),
        "Recovered messages today:",
        str(snapshot["replay"]["recovered_today"]),
        "Replay state:",
        snapshot["replay"]["state"].capitalize(),
    ]
    if delayed:
        lines.append("")
        lines.append("⚠️ Warning: price check cycle is overdue — scheduler may be stuck.")
    return "\n".join(lines)
