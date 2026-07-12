"""Covers listener/dedup.py: the same product from the same channel is
suppressed as a duplicate within the configured window; the same product
from a different channel is unaffected; a missing ASIN falls back to a
normalized title fingerprint.
"""

from __future__ import annotations

from listener import dedup


def test_same_asin_same_channel_is_duplicate_after_mark_seen():
    dedup._seen.clear()
    assert dedup.is_duplicate("chan_a", "B0TEST12345", "Some Title") is False
    dedup.mark_seen("chan_a", "B0TEST12345", "Some Title")
    assert dedup.is_duplicate("chan_a", "B0TEST12345", "Some Title") is True


def test_same_asin_different_channel_is_not_duplicate():
    dedup._seen.clear()
    dedup.mark_seen("chan_a", "B0TEST99999", "Some Title")
    assert dedup.is_duplicate("chan_b", "B0TEST99999", "Some Title") is False


def test_no_asin_falls_back_to_title_fingerprint():
    dedup._seen.clear()
    dedup.mark_seen("chan_a", None, "  Some   Title  ")
    assert dedup.is_duplicate("chan_a", None, "some title") is True
    assert dedup.is_duplicate("chan_a", None, "a totally different title") is False
