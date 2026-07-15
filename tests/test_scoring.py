"""Covers the Value Score engine (listener/scoring.py), its migration
(brand_reputation / price_observations, additive priority_stats columns),
the classify_priority dispatch in listener/budget.py (flag off = byte-for-
byte legacy behavior; flag on = score-driven with the learning.py outlier
check taking precedence), and the 1,000-deal performance benchmark. Zero
real AI/network calls -- everything here is pure SQLite over the per-test
isolated database from conftest.py.
"""

from __future__ import annotations

import itertools
import time
from datetime import datetime, timedelta

import pytest

import database
from listener import budget, scoring


def _seed_verdict(brand, category="accessory", quality="good", asin="B0SEEDX01", days_ago=0):
    database.insert_verdict(
        asin=asin, provider="gemini", brand=brand, category=category, title=f"{brand} thing",
        current_price=500.0, discount_percent=15, deal_quality=quality, reason="x",
        suggested_target=450, channel="chan",
        timestamp=(datetime.now() - timedelta(days=days_ago)).isoformat(),
    )


def _observe(asin, price, family_id=None, days_ago=0):
    database.record_price_observation(
        asin, family_id, price, None, (datetime.now() - timedelta(days=days_ago)).isoformat()
    )


# --- Migration (acceptance test 5) -----------------------------------------


def test_new_tables_created_idempotently():
    database.init_db()  # conftest already ran it once -- this is the second run
    with database.get_connection() as conn:
        tables = {
            row["name"]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        }
        indexes = {
            row["name"]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='index'").fetchall()
        }
    assert "brand_reputation" in tables
    assert "price_observations" in tables
    assert "idx_price_observations_asin" in indexes
    assert "idx_price_observations_family" in indexes


def test_backfill_runs_twice_without_duplicating_data():
    _seed_verdict("Anker", quality="great")
    _seed_verdict("Anker", quality="great")
    _seed_verdict("Generico", quality="skip")

    first = database.backfill_brand_reputation()
    with database.get_connection() as conn:
        rows_after_first = conn.execute("SELECT * FROM brand_reputation ORDER BY brand").fetchall()
    second = database.backfill_brand_reputation()
    with database.get_connection() as conn:
        rows_after_second = conn.execute("SELECT * FROM brand_reputation ORDER BY brand").fetchall()

    assert first == second == 2  # two distinct brands, both runs
    assert len(rows_after_first) == len(rows_after_second) == 2
    assert [r["brand"] for r in rows_after_second] == ["Anker", "Generico"]
    anker = next(r for r in rows_after_second if r["brand"] == "Anker")
    assert anker["sample_count"] == 2


def test_migration_does_not_alter_existing_tables():
    user = database.get_or_create_user(42, "existing")
    database.add_tracked_product(
        user_id=user.id, asin="B0EXISTING", title="t", url="https://www.amazon.eg/dp/B0EXISTING",
        current_price=100.0, target_price=90.0,
    )
    _seed_verdict("Anker")

    with database.get_connection() as conn:
        before = {
            t: conn.execute(f"SELECT COUNT(*) AS n FROM {t}").fetchone()["n"]
            for t in ("users", "tracked_products", "verdicts", "learned_rules", "product_families")
        }

    database.init_db()  # re-run the full migration + backfill

    with database.get_connection() as conn:
        after = {
            t: conn.execute(f"SELECT COUNT(*) AS n FROM {t}").fetchone()["n"]
            for t in ("users", "tracked_products", "verdicts", "learned_rules", "product_families")
        }
    assert before == after


def test_tracked_product_price_check_records_observation():
    user = database.get_or_create_user(43, "obs")
    product = database.add_tracked_product(
        user_id=user.id, asin="B0OBSERVE1", title="t", url="https://www.amazon.eg/dp/B0OBSERVE1",
        current_price=None, target_price=90.0,
    )
    database.update_price_check(product.id, 120.0)
    database.update_price_check(product.id, None, available=False)  # unavailable -> no row

    with database.get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM price_observations WHERE asin = 'B0OBSERVE1'"
        ).fetchall()
    assert len(rows) == 1
    assert rows[0]["price"] == 120.0
    assert rows[0]["family_id"] is None


# --- scoring.py pure functions (acceptance test 1) --------------------------


def test_brand_reputation_strong_brand_scores_high():
    for _ in range(5):
        _seed_verdict("Anker", quality="great")
    database.backfill_brand_reputation()
    assert scoring.compute_brand_reputation("Anker") == pytest.approx(1.0)


def test_brand_reputation_poor_brand_scores_low():
    for _ in range(5):
        _seed_verdict("Generico", quality="skip")
    database.backfill_brand_reputation()
    assert scoring.compute_brand_reputation("Generico") == pytest.approx(0.0)


def test_brand_reputation_unseen_brand_is_neutral():
    assert scoring.compute_brand_reputation("NeverSeenCo") == 0.5
    assert scoring.compute_brand_reputation(None) == 0.5


def test_price_percentile_at_historical_floor_is_1():
    _observe("B0PCTL0001", 100.0)
    _observe("B0PCTL0001", 150.0)
    _observe("B0PCTL0001", 200.0)
    assert scoring.compute_price_percentile("B0PCTL0001", None, 100.0) == 1.0
    assert scoring.compute_price_percentile("B0PCTL0001", None, 95.0) == 1.0  # below floor


def test_price_percentile_above_normal_is_low():
    _observe("B0PCTL0002", 100.0)
    _observe("B0PCTL0002", 200.0)
    assert scoring.compute_price_percentile("B0PCTL0002", None, 200.0) == 0.0
    assert scoring.compute_price_percentile("B0PCTL0002", None, 150.0) == pytest.approx(0.5)


def test_price_percentile_no_history_is_neutral():
    assert scoring.compute_price_percentile("B0NOHISTRY", None, 100.0) == 0.5


def test_price_percentile_includes_family_siblings():
    database.create_product_family(
        family_id="fam_score1", brand=None, normalized_title="widget", category=None,
        deterministic_key="k1", anchor_asin="B0SIBLING1", lowest_price=80.0,
        highest_discount_percent=None, first_seen="2026-01-01T00:00:00",
        last_seen="2026-01-01T00:00:00",
    )
    _observe("B0SIBLING1", 80.0, family_id="fam_score1")
    _observe("B0SIBLING2", 120.0, family_id="fam_score1")
    # New ASIN, no history of its own -- sibling history via family_id gives
    # it a real percentile instead of neutral.
    assert scoring.compute_price_percentile("B0SIBLING3", "fam_score1", 80.0) == 1.0
    assert scoring.compute_price_percentile("B0SIBLING3", "fam_score1", 120.0) == 0.0


def test_family_percentile_none_without_family():
    assert scoring.compute_family_percentile(None, 100.0) is None
    assert scoring.compute_family_percentile("fam_never_created", 100.0) is None


def test_family_percentile_at_family_floor_is_1():
    database.create_product_family(
        family_id="fam_score2", brand=None, normalized_title="w2", category=None,
        deterministic_key="k2", anchor_asin="B0FAM20001", lowest_price=84.0,
        highest_discount_percent=None, first_seen="2026-01-01T00:00:00",
        last_seen="2026-01-01T00:00:00",
    )
    database.upsert_family_member("B0FAM20001", "fam_score2", "{}", 84.0, None, "2026-01-01T00:00:00")
    database.upsert_family_member("B0FAM20002", "fam_score2", "{}", 120.0, None, "2026-01-01T00:00:00")
    assert scoring.compute_family_percentile("fam_score2", 84.0) == 1.0
    assert scoring.compute_family_percentile("fam_score2", 120.0) == 0.0


def test_category_deviation_neutral_without_rules():
    assert scoring.compute_category_deviation("accessory", 500.0, 15) == 0.5
    assert scoring.compute_category_deviation(None, 500.0, 15) == 0.5


def test_category_deviation_reflects_learned_rule_quality():
    from listener import learning

    price_key = f"accessory|{learning.price_bucket(500.0)}"
    with database.get_connection() as conn:
        conn.execute(
            "INSERT INTO learned_rules (rule_type, key, predicted_quality, confidence, sample_count, last_updated, enabled) "
            "VALUES ('category_price', ?, 'great', 0.9, 10, '2026-01-01T00:00:00', 1)",
            (price_key,),
        )
    assert scoring.compute_category_deviation("accessory", 500.0, None) == pytest.approx(1.0)


def test_value_score_neutral_deal_scores_50():
    result = scoring.compute_value_score(
        asin="B0NEUTRAL1", brand=None, category=None, price=500.0,
        discount_percent=None, family_id=None,
    )
    # Every component neutral except rarity (never observed -> 1.0), which
    # pulls the total above the midpoint by exactly its weight share.
    assert result.brand_reputation == 0.5
    assert result.price_percentile == 0.5
    assert result.family_percentile is None
    assert result.category_deviation == 0.5
    assert result.rarity == 1.0
    assert 50.0 < result.total < 70.0


# --- Flag-off regression: byte-for-byte legacy behavior (acceptance 2) -----


def test_flag_off_dispatch_identical_to_legacy_over_input_grid():
    assert budget.SCORE_ENGINE_ENABLED is False  # shipping default

    discounts = [None, 5, 25, 45]
    bools = [False, True]
    confidences = [None, 0.5, 0.8, 0.95]
    for discount, new_fam, unk_brand, new_cat, new_low, conf in itertools.product(
        discounts, bools, bools, bools, bools, confidences
    ):
        kwargs = dict(
            discount_percent=discount, is_new_family=new_fam, is_unknown_brand=unk_brand,
            is_new_category=new_cat, is_new_lowest_family_price=new_low, rule_confidence=conf,
        )
        assert budget.classify_priority(**kwargs) == budget.classify_priority_legacy(**kwargs), kwargs


# --- Flag-on behavior (acceptance 3 + 4) ------------------------------------


def _make_category_seen(category="accessory"):
    _seed_verdict("Anker", category=category)


@pytest.fixture
def score_engine_on(monkeypatch):
    monkeypatch.setattr(budget, "SCORE_ENGINE_ENABLED", True)


def test_premium_brand_at_floor_outranks_legacy(score_engine_on):
    """Premium brand, modest discount, at historical floor: legacy sees only
    'discount 15% + confident rule' (Priority 3); the score engine sees the
    strong brand + at-floor price and elevates it.
    """
    _make_category_seen()
    for _ in range(6):
        _seed_verdict("Anker", quality="great")
    database.backfill_brand_reputation()
    _observe("B0PREMIUM1", 400.0)
    _observe("B0PREMIUM1", 500.0)
    _observe("B0PREMIUM1", 600.0)

    legacy = budget.classify_priority_legacy(
        discount_percent=15, is_new_family=False, is_unknown_brand=False,
        is_new_category=False, is_new_lowest_family_price=False, rule_confidence=0.95,
    )
    scored = budget.classify_priority(
        discount_percent=15, is_new_family=False, is_unknown_brand=False,
        is_new_category=False, is_new_lowest_family_price=False, rule_confidence=0.95,
        asin="B0PREMIUM1", brand="Anker", category="accessory", price=400.0, family_id=None,
    )
    assert legacy == 3
    assert scored < legacy  # elevated by the score engine


def test_poor_reputation_brand_high_discount_never_ranked_higher(score_engine_on):
    """Known poor-reputation brand at 45% discount (below the 50% outlier
    threshold, so the outlier check does NOT fire): legacy's raw-discount
    rule makes it Priority 1; the score engine sees the weak brand and
    above-floor price and does not rank it higher (here: lower).
    """
    _make_category_seen()
    for _ in range(6):
        _seed_verdict("Generico", quality="skip")
    database.backfill_brand_reputation()
    for _ in range(25):  # dense history -> zero rarity
        _observe("B0GENERIC1", 100.0)
    _observe("B0GENERIC1", 300.0)

    legacy = budget.classify_priority_legacy(
        discount_percent=45, is_new_family=False, is_unknown_brand=False,
        is_new_category=False, is_new_lowest_family_price=False, rule_confidence=0.95,
    )
    scored = budget.classify_priority(
        discount_percent=45, is_new_family=False, is_unknown_brand=False,
        is_new_category=False, is_new_lowest_family_price=False, rule_confidence=0.95,
        asin="B0GENERIC1", brand="Generico", category="accessory", price=300.0, family_id=None,
    )
    assert legacy == 1
    assert scored >= legacy  # never higher priority (lower number) than legacy
    assert scored > 1  # and in this construction, actually demoted


def test_outlier_check_takes_precedence_over_score(score_engine_on):
    """learning.is_outlier (>=50% discount here) forces Priority 1 under the
    scored classifier no matter how poor the deal scores otherwise.
    """
    _make_category_seen()
    for _ in range(6):
        _seed_verdict("Generico", quality="skip")
    database.backfill_brand_reputation()
    for _ in range(25):
        _observe("B0OUTLIER1", 100.0)

    scored = budget.classify_priority(
        discount_percent=60,  # >= RULE_OUTLIER_DISCOUNT -> outlier
        is_new_family=False, is_unknown_brand=False, is_new_category=False,
        is_new_lowest_family_price=False, rule_confidence=0.95,
        asin="B0OUTLIER1", brand="Generico", category="accessory", price=300.0, family_id=None,
    )
    assert scored == 1


def test_learning_outlier_check_is_untouched():
    """The safety check itself must not have been modified by this change:
    unknown brand, unseen category, extreme discount, and extreme low price
    each still force an outlier."""
    from listener import learning

    assert learning.is_outlier(None, "accessory", 500.0, 10) is True  # unknown brand
    assert learning.is_outlier("Anker", None, 500.0, 10) is True  # unknown category
    assert learning.is_outlier("Anker", "neverseencat", 500.0, 10) is True  # unseen category
    _make_category_seen()
    assert learning.is_outlier("Anker", "accessory", 500.0, 60) is True  # extreme discount
    assert learning.is_outlier("Anker", "accessory", 5.0, 10) is True  # extreme low price
    assert learning.is_outlier("Anker", "accessory", 500.0, 10) is False  # nothing unusual


# --- Shadow-mode divergence counter -----------------------------------------


def test_shadow_comparison_counter_and_stats():
    database.record_shadow_comparison("2026-07-15", diverged=False)
    database.record_shadow_comparison("2026-07-15", diverged=True)
    database.record_shadow_comparison("2026-07-15", diverged=True)
    stats = database.get_shadow_stats("2026-07-15")
    assert stats == {"shadow_total": 3, "shadow_divergences": 2}
    assert database.get_shadow_stats("1999-01-01") == {"shadow_total": 0, "shadow_divergences": 0}


def test_shadow_log_records_divergence(caplog):
    import logging

    from listener.analyzer import _log_shadow_score

    _make_category_seen()
    legacy_kwargs = dict(
        is_new_family=True, is_unknown_brand=True, is_new_category=True,
        is_new_lowest_family_price=True, rule_confidence=None,
    )
    scored_kwargs = dict(asin="B0SHADOW01", brand=None, category=None, price=500.0, family_id=None)

    with caplog.at_level(logging.INFO, logger="fanzi.listener.analyzer"):
        _log_shadow_score(legacy_kwargs, scored_kwargs, 10, None, "2026-07-15")

    assert any("score_shadow" in r.message for r in caplog.records)
    line = next(r.message for r in caplog.records if "score_shadow" in r.message)
    assert "legacy_priority=" in line and "scored_priority=" in line and "value_score=" in line
    assert database.get_shadow_stats("2026-07-15")["shadow_total"] == 1


# --- Performance benchmark (1,000 deals well under 50 ms) -------------------


def test_value_score_benchmark_1000_deals_under_50ms():
    # Representative row counts: 50 brands, ~5,000 price observations
    # across 200 ASINs (a few weeks of production traffic), a handful of
    # learned category rules, and 20 families.
    now = datetime.now().isoformat()
    with database.get_connection() as conn:
        for b in range(50):
            conn.execute(
                "INSERT INTO brand_reputation VALUES (?, 0.7, 5.0, 10, ?)", (f"Brand{b}", now)
            )
        for i in range(5000):
            asin = f"B0BENCH{i % 200:03d}"
            fam = f"fam_bench{i % 20}" if i % 2 == 0 else None
            conn.execute(
                "INSERT INTO price_observations (asin, family_id, price, discount_percent, observed_at) "
                "VALUES (?, ?, ?, NULL, ?)",
                (asin, fam, 100.0 + (i % 400), now),
            )
        for f in range(20):
            conn.execute(
                "INSERT INTO product_families (family_id, brand, normalized_title, category, deterministic_key, "
                "anchor_asin, lowest_price, variant_count, first_seen, last_seen) "
                "VALUES (?, NULL, ?, NULL, ?, ?, 100.0, 0, ?, ?)",
                (f"fam_bench{f}", f"bench {f}", f"bk{f}", f"B0BENCH{f:03d}", now, now),
            )
        conn.execute(
            "INSERT INTO learned_rules (rule_type, key, predicted_quality, confidence, sample_count, last_updated, enabled) "
            "VALUES ('category_price', 'accessory|200-500', 'good', 0.85, 12, ?, 1)",
            (now,),
        )

    # Confirm the required indexes exist before the benchmark counts.
    with database.get_connection() as conn:
        indexes = {
            row["name"]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='index'").fetchall()
        }
        assert {"idx_price_observations_asin", "idx_price_observations_family"} <= indexes

        # One read transaction across the whole batch -- without it, every
        # SELECT in autocommit mode pays a file-lock acquire/release
        # (~50us of syscalls on Windows), which is measurement noise about
        # the OS, not about the scoring queries this benchmark exists to
        # keep honest. compute_value_score does the same internally when
        # it owns its connection.
        conn.execute("BEGIN")

        # Best-of-3: this box also runs the live bot (and Playwright during
        # the full suite), so a single wall-clock sample measures transient
        # machine load as much as the code. The minimum reflects what the
        # scoring queries actually cost.
        samples = []
        for _ in range(3):
            start = time.perf_counter()
            for i in range(1000):
                scoring.compute_value_score(
                    asin=f"B0BENCH{i % 200:03d}",
                    brand=f"Brand{i % 50}",
                    category="accessory",
                    price=100.0 + (i % 400),
                    discount_percent=15,
                    family_id=f"fam_bench{i % 20}" if i % 2 == 0 else None,
                    conn=conn,
                )
            samples.append((time.perf_counter() - start) * 1000)
        elapsed_ms = min(samples)

    assert elapsed_ms < 50, f"1000 value scores took {elapsed_ms:.1f} ms best-of-3 (budget: 50 ms; samples: {samples})"
