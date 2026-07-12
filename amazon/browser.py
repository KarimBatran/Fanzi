"""Playwright browser management: one persistent profile, reused across
every fetch, so the browser accumulates cookies/history like a real user
over time instead of presenting as a fresh, suspicious session each time.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from playwright.async_api import BrowserContext, Playwright, async_playwright

PROFILE_DIR = Path(__file__).resolve().parent.parent / "playwright_profile"

_playwright: Playwright | None = None
_context: BrowserContext | None = None
_lock = asyncio.Lock()


async def get_context() -> BrowserContext:
    """Returns the shared persistent browser context, launching it on first use."""
    global _playwright, _context
    async with _lock:
        if _context is None:
            PROFILE_DIR.mkdir(parents=True, exist_ok=True)
            _playwright = await async_playwright().start()
            _context = await _playwright.chromium.launch_persistent_context(
                str(PROFILE_DIR),
                headless=True,
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36"
                ),
                viewport={"width": 1366, "height": 768},
                locale="en-US",
            )
        return _context


async def close_browser() -> None:
    """Releases the persistent context/Playwright driver. Safe to call even
    if get_context() was never called. Wired into graceful shutdown in
    Milestone 4.
    """
    global _playwright, _context
    async with _lock:
        if _context is not None:
            await _context.close()
            _context = None
        if _playwright is not None:
            await _playwright.stop()
            _playwright = None
