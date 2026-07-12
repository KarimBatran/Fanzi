"""Covers listener/dedup.py: same ASIN+channel+price within the window is a
duplicate; a price drop or a bigger discount forces reprocessing; a
different channel is independent; an expired window forces reprocessing
too. Backed by the isolated temp DB from conftest.py.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import database
from listener import dedup


def test_new_product_is_not_a_duplicate():
    assert dedup.check("chan_a", "B0TEST00001", "Title", 100.0, 20) == dedup.NEW


def test_same_price_same_discount_is_duplicate():
    dedup.mark_seen("chan_a", "B0TEST00002", "Title", 100.0, 20)
    assert dedup.check("chan_a", "B0TEST00002", "Title", 100.0, 20) == dedup.DUPLICATE


def test_price_decrease_forces_reprocess():
    dedup.mark_seen("chan_a", "B0TEST00003", "Title", 100.0, 20)
    assert dedup.check("chan_a", "B0TEST00003", "Title", 90.0, 20) == dedup.PRICE_CHANGED


def test_discount_increase_forces_reprocess():
    dedup.mark_seen("chan_a", "B0TEST00004", "Title", 100.0, 20)
    assert dedup.check("chan_a", "B0TEST00004", "Title", 100.0, 30) == dedup.PRICE_CHANGED


def test_price_increase_alone_is_still_duplicate():
    dedup.mark_seen("chan_a", "B0TEST00005", "Title", 100.0, 20)
    assert dedup.check("chan_a", "B0TEST00005", "Title", 110.0, 20) == dedup.DUPLICATE


def test_different_channel_is_independent():
    dedup.mark_seen("chan_a", "B0TEST00006", "Title", 100.0, 20)
    assert dedup.check("chan_b", "B0TEST00006", "Title", 100.0, 20) == dedup.NEW


def test_no_asin_falls_back_to_title_fingerprint():
    dedup.mark_seen("chan_a", None, "  Some   Title  ", 50.0, 10)
    assert dedup.check("chan_a", None, "some title", 50.0, 10) == dedup.DUPLICATE
    assert dedup.check("chan_a", None, "a totally different title", 50.0, 10) == dedup.NEW


def test_window_expired_forces_reprocess():
    old_seen_at = (datetime.now() - timedelta(hours=100)).isoformat()
    database.upsert_duplicate_record("chan_a", "asin:B0TEST00007", 100.0, 20, old_seen_at)
    assert dedup.check("chan_a", "B0TEST00007", "Title", 100.0, 20) == dedup.WINDOW_EXPIRED


def test_active_count_reflects_non_expired_entries():
    dedup.mark_seen("chan_a", "B0TEST00008", "Title", 100.0, 20)
    assert dedup.get_active_count() >= 1


def test_cleanup_expired_removes_old_entries():
    old_seen_at = (datetime.now() - timedelta(hours=100)).isoformat()
    database.upsert_duplicate_record("chan_a", "asin:B0TEST00009", 100.0, 20, old_seen_at)
    deleted = dedup.cleanup_expired()
    assert deleted >= 1
    assert dedup.check("chan_a", "B0TEST00009", "Title", 100.0, 20) == dedup.NEW
