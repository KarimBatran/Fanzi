"""Covers listener/dedup.py: same ASIN+price within the window is a
duplicate REGARDLESS of which channel posted it (the same product from two
different channels must only ever forward once); a price drop or a bigger
discount forces reprocessing; an expired window forces reprocessing too.
Backed by the isolated temp DB from conftest.py.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import database
from listener import dedup


def test_new_product_is_not_a_duplicate():
    assert dedup.check("B0TEST00001", "Title", 100.0, 20, channel_name="chan_a") == dedup.NEW


def test_same_price_same_discount_is_duplicate():
    dedup.mark_seen("B0TEST00002", "Title", 100.0, 20, channel_name="chan_a")
    assert dedup.check("B0TEST00002", "Title", 100.0, 20, channel_name="chan_a") == dedup.DUPLICATE


def test_same_asin_from_a_different_channel_is_also_a_duplicate():
    """The exact production bug this fixes: the same ASIN posted by two
    different channels must be recognized as the same deal.
    """
    dedup.mark_seen("B0TEST00002B", "Tornado TOF-49Y fan", 1799.0, 15, channel_name="OffersEgyptofficial")
    assert (
        dedup.check("B0TEST00002B", "Tornado TOF-49Y fan (different title text)", 1799.0, 15, channel_name="Mego_Reviews")
        == dedup.DUPLICATE
    )


def test_price_decrease_forces_reprocess():
    dedup.mark_seen("B0TEST00003", "Title", 100.0, 20, channel_name="chan_a")
    assert dedup.check("B0TEST00003", "Title", 90.0, 20, channel_name="chan_a") == dedup.PRICE_CHANGED


def test_discount_increase_forces_reprocess():
    dedup.mark_seen("B0TEST00004", "Title", 100.0, 20, channel_name="chan_a")
    assert dedup.check("B0TEST00004", "Title", 100.0, 30, channel_name="chan_a") == dedup.PRICE_CHANGED


def test_price_increase_alone_is_still_duplicate():
    dedup.mark_seen("B0TEST00005", "Title", 100.0, 20, channel_name="chan_a")
    assert dedup.check("B0TEST00005", "Title", 110.0, 20, channel_name="chan_a") == dedup.DUPLICATE


def test_no_asin_falls_back_to_title_and_price_fingerprint():
    dedup.mark_seen(None, "  Some   Title  ", 50.0, 10, channel_name="chan_a")
    assert dedup.check(None, "some title", 50.0, 10, channel_name="chan_a") == dedup.DUPLICATE
    assert dedup.check(None, "a totally different title", 50.0, 10, channel_name="chan_a") == dedup.NEW
    # Same title but a different price is a different fingerprint entirely.
    assert dedup.check(None, "some title", 75.0, 10, channel_name="chan_a") == dedup.NEW


def test_window_expired_forces_reprocess():
    old_seen_at = (datetime.now() - timedelta(hours=100)).isoformat()
    database.upsert_global_duplicate_record("asin:B0TEST00007", "chan_a", 100.0, 20, old_seen_at)
    assert dedup.check("B0TEST00007", "Title", 100.0, 20, channel_name="chan_a") == dedup.WINDOW_EXPIRED


def test_active_count_reflects_non_expired_entries():
    dedup.mark_seen("B0TEST00008", "Title", 100.0, 20, channel_name="chan_a")
    assert dedup.get_active_count() >= 1


def test_cleanup_expired_removes_old_entries():
    old_seen_at = (datetime.now() - timedelta(hours=100)).isoformat()
    database.upsert_global_duplicate_record("asin:B0TEST00009", "chan_a", 100.0, 20, old_seen_at)
    deleted = dedup.cleanup_expired()
    assert deleted >= 1
    assert dedup.check("B0TEST00009", "Title", 100.0, 20, channel_name="chan_a") == dedup.NEW


def test_url_never_influences_the_dedup_key():
    """The fingerprint is derived purely from asin/title/price -- there is
    no url parameter accepted by check()/mark_seen() at all, so a channel's
    original link (however it's formatted) structurally cannot affect
    duplicate detection.
    """
    import inspect

    check_params = inspect.signature(dedup.check).parameters
    mark_seen_params = inspect.signature(dedup.mark_seen).parameters
    assert "url" not in check_params
    assert "url" not in mark_seen_params
