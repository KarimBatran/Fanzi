"""Runtime-managed deal channel list. .env's DEAL_CHANNELS is always the
base/fallback set; additions and removals made via /addchannel and
/removechannel are persisted in data/channels.json as overrides on top of
it, so the base set is always recoverable by simply deleting that file.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from config import DEAL_CHANNELS

logger = logging.getLogger("fanzi.listener.channels_store")

STORE_PATH = Path("data") / "channels.json"


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
