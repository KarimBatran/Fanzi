# Value Score Engine — Rollout Plan

The Value Score engine (`listener/scoring.py` + `listener/budget.py`'s
`classify_priority_scored`) replaces the legacy discount-percentage /
"unknown brand always wins Priority 1" heuristic with a deterministic,
zero-AI-cost 0–100 score built from brand reputation, historical price
percentile, family price percentile, category deviation, and rarity.

It ships **disabled**. The flag that controls it is the only behavioral
branch point in the entire change:

| Flag | Default | Meaning |
|---|---|---|
| `SCORE_ENGINE_ENABLED` | `false` | `false` = legacy classifier decides priority (byte-for-byte pre-change behavior); `true` = scored classifier decides |
| `SCORE_ENGINE_LOG_VERBOSE` | `true` | Logs both classifiers' decisions side-by-side per deal (shadow mode), regardless of which one is actually used |

## Phase 1 — Ship (shadow mode)

- Deploy with `SCORE_ENGINE_ENABLED=false`, `SCORE_ENGINE_LOG_VERBOSE=true`.
- **Zero behavior change**: priority classification is
  `classify_priority_legacy`, verbatim the pre-change heuristic.
- Every analyzed deal additionally logs one structured `score_shadow` line
  (fire-and-forget, off the live forward path):
  `asin`, `brand`, `category`, `legacy_priority`, `scored_priority`,
  `value_score`, and each component sub-score.
- Divergences are counted per-day in `priority_stats.shadow_total` /
  `shadow_divergences` and surfaced in `/status` as
  "Score engine shadow divergence: N% (X/Y deals)".
- The `brand_reputation` table is rebuilt from `verdicts` on every startup
  (idempotent); `price_observations` starts accumulating from both the
  deal-forwarding and tracked-product paths immediately.

## Phase 2 — Validate (minimum 5–7 days of shadow data)

- Watch the `/status` divergence rate daily.
- Grep the logs for `score_shadow` lines where
  `legacy_priority != scored_priority` and manually review a sample:
  - Known **premium brands** already in the data (e.g. Anker, Samsung,
    Sony) at modest discounts near their historical price floor — the
    scored classifier should rank these **higher** than legacy did.
  - Known **low-reputation / unknown brands** with very high discounts —
    the scored classifier should rank these **lower** than legacy's
    automatic Priority 1 (unless the `learning.py` outlier check fires,
    which always forces Priority 1 in both classifiers).
- Produce a short before/after summary from the logs: overall divergence
  rate, and for the sampled divergent cases, which classification looks
  more correct on inspection. Only proceed if the scored calls look right.

## Phase 3 — Flip

- Set `SCORE_ENGINE_ENABLED=true` and restart.
- Keep `SCORE_ENGINE_LOG_VERBOSE=true` initially — shadow lines keep
  logging both priorities, so post-flip drift is still observable.
- Monitor daily AI spend via the existing `/status` Budget section and
  `priority_stats` counters. Expected shift: fewer Priority-1 calls burned
  on low-reputation-brand/high-discount noise, more correctly-elevated
  calls on high-reputation or near-historical-floor deals.
- Once satisfied, `SCORE_ENGINE_LOG_VERBOSE=false` quiets the per-deal
  shadow lines (divergence counting stops with it).

## Rollback

- Set `SCORE_ENGINE_ENABLED=false` and restart — instantly reverts to
  `classify_priority_legacy`, identical to pre-change behavior.
- **No data migration needed**: `brand_reputation` and
  `price_observations` are additive and unused by any other code path;
  they simply keep accumulating (or sit idle) either way.
- A post-Phase-3 rollback needs no reprocessing: the score engine only
  ever affected the priority classification of new incoming deals, never
  stored `verdicts`, `learned_rules`, family state, or dedup state.

## Invariants (enforced by tests)

- `classify_priority()` with the flag off ≡ `classify_priority_legacy()`
  for identical inputs (regression-tested over a fixed input grid).
- `learning.is_outlier()` takes precedence over any score: an outlier deal
  is Priority 1 under both classifiers, always.
- Both new tables are created idempotently; the backfill is a full
  DELETE+rebuild, safe to run any number of times; no existing table's
  schema or rows are altered (additive `priority_stats` columns only).
- Scoring 1,000 deals over representative table sizes completes in well
  under 50 ms total on a shared connection (benchmark-tested) — the score
  sits upstream of the AI soft-timeout and must add no perceptible
  per-message latency.
