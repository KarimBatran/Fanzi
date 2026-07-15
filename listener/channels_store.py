"""Runtime-managed deal channel list. .env's DEAL_CHANNELS is always the
base/fallback set; additions and removals made via /addchannel and
/removechannel are persisted in data/channels.json as overrides on top of
it, so the base set is always recoverable by simply deleting that file.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from config import DEAL_CHANNELS

logger = logging.getLogger("fanzi.listener.channels_store")

STORE_PATH = Path("data") / "channels.json"

_TME_URL_RE = re.compile(r"^(?:https?://)?(?:www\.)?t\.me/(.+)$", re.IGNORECASE)


def normalize_channel_input(raw: str) -> str:
    """Extracts a bare channel username from whatever the admin pastes:
    "@ba3bou3_deals", "ba3bou3_deals", "https://t.me/ba3bou3_deals",
    "t.me/ba3bou3_deals", or "t.me/s/ba3bou3_deals" (the web-preview form)
    all normalize to "ba3bou3_deals". A private invite link
    (t.me/+<hash> or t.me/joinchat/<hash>) has no @username, so its path is
    returned as-is for the caller's get_entity to resolve. Trailing slashes
    and query/fragment junk are stripped.
    """
    s = raw.strip()
    match = _TME_URL_RE.match(s)
    if match:
        s = match.group(1)
        if s.lower().startswith("s/"):  # t.me/s/<name> web-preview link
            s = s[2:]
        # Private invite links carry no @username -- hand the original back
        # untouched so the caller's get_entity resolves it via the full hash.
        if s.startswith("+") or s.lower().startswith("joinchat/"):
            return raw.strip()
    # Drop anything after the username (a trailing "/", "?utm=...", "#...").
    s = re.split(r"[/?#]", s, maxsplit=1)[0]
    return s.lstrip("@").strip()


def _load_overrides() -> dict:
    if not STORE_PATH.is_file():
        return {"added": [], "removed": []}
    try:
        with open(STORE_PATH, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        logger.exception("failed to read %s — treating as empty", STORE_PATH)
        return {"added": [], "removed": []}
    return {
        "added": list(data.get("added", [])),
        "removed": list(data.get("removed", [])),
    }


def _save_overrides(overrides: dict) -> None:
    STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(STORE_PATH, "w", encoding="utf-8") as f:
        json.dump(overrides, f, indent=2)


def get_effective_channels() -> list[str]:
    """.env DEAL_CHANNELS plus channels.json additions, minus removals —
    de-duplicated, .env order preserved first.
    """
    overrides = _load_overrides()
    removed = set(overrides["removed"])
    channels = [c for c in DEAL_CHANNELS if c not in removed]
    for c in overrides["added"]:
        if c not in channels and c not in removed:
            channels.append(c)
    return channels


def add_channel(channel: str) -> None:
    overrides = _load_overrides()
    if channel in overrides["removed"]:
        overrides["removed"].remove(channel)
    if channel not in overrides["added"] and channel not in DEAL_CHANNELS:
        overrides["added"].append(channel)
    _save_overrides(overrides)


def remove_channel(channel: str) -> None:
    overrides = _load_overrides()
    if channel in overrides["added"]:
        overrides["added"].remove(channel)
    elif channel in DEAL_CHANNELS and channel not in overrides["removed"]:
        overrides["removed"].append(channel)
    _save_overrides(overrides)
