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
from listener import dedup
from listener.ai_providers import get_manager

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


def build_snapshot() -> dict:
    """Full snapshot used by both /status and health.json."""
    _reset_counters_if_new_day()
    tracked_items = len(database.get_all_active_products_with_owner())
    last_check_seconds_ago = (
        int((datetime.now() - _last_check).total_seconds()) if _last_check is not None else None
    )
    providers = get_manager().status_snapshot()
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

    def _latency_line(p: dict) -> str:
        return f"Last latency: {int(p['last_latency_ms'])} ms" if p["last_latency_ms"] is not None else "Last latency: n/a"

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
        "Gemini",
        f"Status: {providers['gemini']['status']}",
        f"Calls today: {providers['gemini']['calls_today']}",
        _latency_line(providers["gemini"]),
        "Groq",
        f"Status: {providers['groq']['status']}",
        f"Calls today: {providers['groq']['calls_today']}",
        _latency_line(providers["groq"]),
        f"Current primary: {providers['primary']}",
        f"Fallback: {providers['fallback']}",
    ]
    if delayed:
        lines.append("")
        lines.append("⚠️ Warning: price check cycle is overdue — scheduler may be stuck.")
    return "\n".join(lines)
