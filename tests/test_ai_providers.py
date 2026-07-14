"""Covers the dual-provider (Gemini primary, Groq fallback) architecture in
listener/ai_providers.py: provider selection order, retry policy, circuit
breaker activation/recovery, quota handling, and the unified verdict schema.
No real Gemini or Groq call is ever made — tests/conftest.py's
block_real_ai_providers fixture would fail loudly if one slipped through;
here we go further and patch each provider's `generate()` directly.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from listener.ai_providers import (
    AIProviderManager,
    FatalProviderError,
    GeminiProvider,
    GroqProvider,
    QuotaExhaustedError,
    TransientProviderError,
    _parse_verdict,
)

_VALID_JSON = (
    '{"deal_quality": "great", "reason": "43%% off a well-known brand.", '
    '"suggested_target": 4800, "category": "appliance"}'
).replace("%%", "%")


def _manager() -> AIProviderManager:
    return AIProviderManager(GeminiProvider(), GroqProvider())


async def _get_verdict(manager: AIProviderManager):
    return await manager.get_verdict(
        title="Test product",
        price=1000.0,
        discount_percent=30,
        channel_name="test_channel",
        price_history=None,
    )


@pytest.mark.asyncio
async def test_gemini_success_never_touches_groq():
    manager = _manager()
    with patch.object(manager.gemini, "generate", new=AsyncMock(return_value=_VALID_JSON)), patch.object(
        manager.groq, "generate", new=AsyncMock()
    ) as groq_generate:
        verdict = await _get_verdict(manager)

    assert verdict is not None
    assert verdict.provider == "gemini"
    assert verdict.deal_quality == "great"
    groq_generate.assert_not_called()


@pytest.mark.asyncio
async def test_gemini_unavailable_falls_back_to_groq(monkeypatch):
    manager = _manager()
    monkeypatch.setattr("listener.ai_providers.asyncio.sleep", AsyncMock())
    with patch.object(
        manager.gemini, "generate", new=AsyncMock(side_effect=TransientProviderError("UNAVAILABLE"))
    ), patch.object(manager.groq, "generate", new=AsyncMock(return_value=_VALID_JSON)):
        verdict = await _get_verdict(manager)

    assert verdict is not None
    assert verdict.provider == "groq"


@pytest.mark.asyncio
async def test_gemini_quota_exhausted_falls_back_to_groq():
    manager = _manager()
    assert manager.gemini.quota_exhausted() is False
    with patch.object(
        manager.gemini, "generate", new=AsyncMock(side_effect=QuotaExhaustedError("RESOURCE_EXHAUSTED"))
    ), patch.object(manager.groq, "generate", new=AsyncMock(return_value=_VALID_JSON)):
        verdict = await _get_verdict(manager)

    assert verdict is not None
    assert verdict.provider == "groq"
    assert manager.gemini.quota_exhausted() is True

    # A second deal must not even attempt Gemini now — it's gated by the
    # persisted quota flag, not the in-memory circuit breaker.
    with patch.object(manager.gemini, "generate", new=AsyncMock()) as gemini_generate, patch.object(
        manager.groq, "generate", new=AsyncMock(return_value=_VALID_JSON)
    ):
        verdict2 = await _get_verdict(manager)
    gemini_generate.assert_not_called()
    assert verdict2 is not None
    assert verdict2.provider == "groq"


@pytest.mark.asyncio
async def test_groq_unavailable_gemini_still_succeeds():
    """Groq's health is irrelevant when Gemini (tried first) is healthy —
    Groq must never even be called.
    """
    manager = _manager()
    with patch.object(manager.gemini, "generate", new=AsyncMock(return_value=_VALID_JSON)), patch.object(
        manager.groq, "generate", new=AsyncMock(side_effect=FatalProviderError("groq is down"))
    ) as groq_generate:
        verdict = await _get_verdict(manager)

    assert verdict is not None
    assert verdict.provider == "gemini"
    groq_generate.assert_not_called()


@pytest.mark.asyncio
async def test_both_providers_unavailable_returns_none_not_dropped(monkeypatch):
    manager = _manager()
    monkeypatch.setattr("listener.ai_providers.asyncio.sleep", AsyncMock())
    with patch.object(
        manager.gemini, "generate", new=AsyncMock(side_effect=TransientProviderError("UNAVAILABLE"))
    ), patch.object(manager.groq, "generate", new=AsyncMock(side_effect=FatalProviderError("auth failed"))):
        verdict = await _get_verdict(manager)

    assert verdict is None  # caller (analyze_deal/watcher) forwards the raw deal instead


@pytest.mark.asyncio
async def test_circuit_breaker_activation(monkeypatch):
    """5 consecutive transient failures must mark the provider unhealthy and
    stop sending it further requests (checked via a call that would raise if
    reached).
    """
    manager = _manager()
    monkeypatch.setattr("listener.ai_providers.asyncio.sleep", AsyncMock())

    # First get_verdict call: Gemini fails all 4 attempts (1 + 3 retries) —
    # 4 consecutive failures, not yet at the threshold of 5.
    with patch.object(
        manager.gemini, "generate", new=AsyncMock(side_effect=TransientProviderError("UNAVAILABLE"))
    ), patch.object(manager.groq, "generate", new=AsyncMock(return_value=_VALID_JSON)):
        await _get_verdict(manager)
    assert manager.health["gemini"].consecutive_failures == 4
    assert manager.health["gemini"].healthy is True

    # Second call: 1 more failure trips the breaker mid-attempt (5th
    # consecutive failure) and must stop retrying immediately rather than
    # burning the remaining 2 retries against an already-tripped breaker.
    with patch.object(
        manager.gemini, "generate", new=AsyncMock(side_effect=TransientProviderError("UNAVAILABLE"))
    ) as gemini_generate, patch.object(manager.groq, "generate", new=AsyncMock(return_value=_VALID_JSON)):
        verdict = await _get_verdict(manager)
    assert manager.health["gemini"].healthy is False
    assert gemini_generate.call_count == 1
    assert verdict is not None
    assert verdict.provider == "groq"

    # Third call: breaker is open and cooldown hasn't expired — Gemini must
    # not be attempted at all.
    with patch.object(manager.gemini, "generate", new=AsyncMock()) as gemini_generate_2, patch.object(
        manager.groq, "generate", new=AsyncMock(return_value=_VALID_JSON)
    ):
        await _get_verdict(manager)
    gemini_generate_2.assert_not_called()


@pytest.mark.asyncio
async def test_circuit_breaker_recovery_on_successful_probe():
    manager = _manager()
    health = manager.health["gemini"]
    health.healthy = False
    health.consecutive_failures = 5
    health.cooldown_until_monotonic = 0.0  # already expired

    with patch.object(manager.gemini, "generate", new=AsyncMock(return_value=_VALID_JSON)) as gemini_generate:
        verdict = await _get_verdict(manager)

    assert verdict is not None
    assert verdict.provider == "gemini"
    assert gemini_generate.call_count == 1  # exactly one probe request
    assert manager.health["gemini"].healthy is True
    assert manager.health["gemini"].consecutive_failures == 0


@pytest.mark.asyncio
async def test_circuit_breaker_reenters_cooldown_on_failed_probe():
    manager = _manager()
    health = manager.health["gemini"]
    health.healthy = False
    health.consecutive_failures = 5
    health.cooldown_until_monotonic = 0.0  # already expired -> probe eligible

    with patch.object(
        manager.gemini, "generate", new=AsyncMock(side_effect=TransientProviderError("still down"))
    ) as gemini_generate, patch.object(manager.groq, "generate", new=AsyncMock(return_value=_VALID_JSON)):
        await _get_verdict(manager)

    assert gemini_generate.call_count == 1  # probe was a single attempt, no retry
    assert manager.health["gemini"].healthy is False
    assert manager.health["gemini"].cooldown_until_monotonic > 0.0  # re-armed


@pytest.mark.asyncio
async def test_retry_behavior_backoff_then_succeeds(monkeypatch):
    manager = _manager()
    sleep_mock = AsyncMock()
    monkeypatch.setattr("listener.ai_providers.asyncio.sleep", sleep_mock)

    with patch.object(
        manager.gemini,
        "generate",
        new=AsyncMock(
            side_effect=[
                TransientProviderError("UNAVAILABLE"),
                TransientProviderError("UNAVAILABLE"),
                _VALID_JSON,
            ]
        ),
    ) as gemini_generate:
        verdict = await _get_verdict(manager)

    assert verdict is not None
    assert verdict.provider == "gemini"
    assert gemini_generate.call_count == 3
    assert sleep_mock.call_count == 2


@pytest.mark.asyncio
async def test_non_retryable_error_fails_immediately_without_retry(monkeypatch):
    manager = _manager()
    sleep_mock = AsyncMock()
    monkeypatch.setattr("listener.ai_providers.asyncio.sleep", sleep_mock)

    with patch.object(
        manager.gemini, "generate", new=AsyncMock(side_effect=FatalProviderError("invalid api key"))
    ) as gemini_generate, patch.object(manager.groq, "generate", new=AsyncMock(return_value=_VALID_JSON)):
        verdict = await _get_verdict(manager)

    gemini_generate.assert_called_once()
    sleep_mock.assert_not_called()
    assert verdict is not None
    assert verdict.provider == "groq"


@pytest.mark.asyncio
async def test_unified_response_schema_identical_across_providers():
    manager = _manager()

    with patch.object(manager.gemini, "generate", new=AsyncMock(return_value=_VALID_JSON)):
        gemini_verdict = await _get_verdict(manager)

    manager2 = _manager()
    with patch.object(
        manager2.gemini, "generate", new=AsyncMock(side_effect=TransientProviderError("down"))
    ), patch.object(manager2.groq, "generate", new=AsyncMock(return_value=_VALID_JSON)), patch(
        "listener.ai_providers.asyncio.sleep", new=AsyncMock()
    ):
        groq_verdict = await _get_verdict(manager2)

    for verdict, expected_provider in ((gemini_verdict, "gemini"), (groq_verdict, "groq")):
        assert verdict is not None
        assert verdict.provider == expected_provider
        assert verdict.deal_quality == "great"
        assert verdict.reason
        assert verdict.suggested_target == 4800
        assert verdict.category == "appliance"


def test_parse_verdict_tolerates_trailing_extra_brace():
    """Real production case: Gemini emitted one complete, valid JSON object
    followed by a stray extra "}" ("Extra data" JSONDecodeError under
    json.loads). raw_decode recovers the real verdict instead of discarding
    it as malformed.
    """
    text = (
        "{\n"
        '  "deal_quality": "good",\n'
        '  "reason": "Strong 41% discount on a reliable TP-Link PoE switch, offering great value.",\n'
        '  "suggested_target": 745,\n'
        '  "category": "accessory"\n'
        "}\n"
        "}"
    )
    verdict = _parse_verdict(text, "gemini")
    assert verdict is not None
    assert verdict.deal_quality == "good"
    assert verdict.suggested_target == 745
    assert verdict.category == "accessory"


_FENCED_JSON = "```json\n" + _VALID_JSON + "\n```"
_LEADING_PROSE_JSON = "Here is the analysis:\n" + _VALID_JSON
_TRAILING_COMMENTARY_JSON = _VALID_JSON + "\nHope this helps."
_MISSING_FIELD_JSON = '{"deal_quality": "good", "reason": "Decent deal.", "suggested_target": 900}'  # no "category"


@pytest.mark.asyncio
async def test_markdown_fenced_json_recovered_without_retry():
    manager = _manager()
    with patch.object(manager.gemini, "generate", new=AsyncMock(return_value=_FENCED_JSON)) as gemini_generate:
        verdict = await _get_verdict(manager)
    assert gemini_generate.call_count == 1  # no repair retry needed
    assert verdict is not None
    assert verdict.provider == "gemini"
    assert verdict.deal_quality == "great"


@pytest.mark.asyncio
async def test_leading_explanatory_text_recovered_without_retry():
    manager = _manager()
    with patch.object(manager.gemini, "generate", new=AsyncMock(return_value=_LEADING_PROSE_JSON)) as gemini_generate:
        verdict = await _get_verdict(manager)
    assert gemini_generate.call_count == 1
    assert verdict is not None
    assert verdict.deal_quality == "great"


@pytest.mark.asyncio
async def test_trailing_commentary_recovered_without_retry():
    manager = _manager()
    with patch.object(manager.gemini, "generate", new=AsyncMock(return_value=_TRAILING_COMMENTARY_JSON)) as gemini_generate:
        verdict = await _get_verdict(manager)
    assert gemini_generate.call_count == 1
    assert verdict is not None
    assert verdict.deal_quality == "great"


@pytest.mark.asyncio
async def test_missing_required_field_triggers_exactly_one_repair_retry():
    manager = _manager()
    with patch.object(
        manager.gemini, "generate", new=AsyncMock(side_effect=[_MISSING_FIELD_JSON, _VALID_JSON])
    ) as gemini_generate:
        verdict = await _get_verdict(manager)

    assert gemini_generate.call_count == 2
    # The repair retry must use the stricter prompt.
    assert gemini_generate.call_args_list[1].kwargs.get("strict") is True
    assert gemini_generate.call_args_list[0].kwargs.get("strict") in (False, None)
    assert verdict is not None
    assert verdict.provider == "gemini"
    assert verdict.deal_quality == "great"


@pytest.mark.asyncio
async def test_successful_repair_retry_does_not_increment_failure_counters():
    import database
    from listener.ai_providers import _today

    manager = _manager()
    with patch.object(manager.gemini, "generate", new=AsyncMock(side_effect=[_MISSING_FIELD_JSON, _VALID_JSON])):
        verdict = await _get_verdict(manager)

    assert verdict is not None
    assert manager.health["gemini"].consecutive_failures == 0
    assert manager.health["gemini"].healthy is True
    stats = database.get_provider_stats("gemini", _today())
    assert stats["failed_requests"] == 0
    assert stats["successful_requests"] == 1


@pytest.mark.asyncio
async def test_repair_retry_also_malformed_fails_over_normally():
    import database
    from listener.ai_providers import _today

    manager = _manager()
    with patch.object(
        manager.gemini, "generate", new=AsyncMock(side_effect=[_MISSING_FIELD_JSON, _MISSING_FIELD_JSON])
    ) as gemini_generate, patch.object(manager.groq, "generate", new=AsyncMock(return_value=_VALID_JSON)):
        verdict = await _get_verdict(manager)

    assert gemini_generate.call_count == 2  # original + exactly one repair retry, no more
    assert verdict is not None
    assert verdict.provider == "groq"  # normal failover executed

    # Exactly one failure counted for the whole malformed+failed-retry group.
    assert manager.health["gemini"].consecutive_failures == 1
    stats = database.get_provider_stats("gemini", _today())
    assert stats["failed_requests"] == 1


@pytest.mark.asyncio
async def test_raw_valid_json_still_parses_normally_no_retry():
    manager = _manager()
    with patch.object(manager.gemini, "generate", new=AsyncMock(return_value=_VALID_JSON)) as gemini_generate:
        verdict = await _get_verdict(manager)
    assert gemini_generate.call_count == 1
    assert verdict is not None
    assert verdict.deal_quality == "great"


def test_parse_verdict_still_rejects_genuine_mid_object_corruption():
    """Real production case: a stray quote landed *inside* the object,
    before it closed ("Expecting ',' delimiter"). This is genuine
    corruption, not trailing garbage — must still be rejected, not guessed at.
    """
    text = (
        "{\n"
        '  "deal_quality": "good",\n'
        '  "reason": "A solid 28% discount on a reliable Kenwood appliance with high utility.",\n'
        '  "suggested_target": 1850,\n'
        '  "category": "appliance"\n'
        '"\n'
        "}"
    )
    verdict = _parse_verdict(text, "gemini")
    assert verdict is None


@pytest.mark.asyncio
async def test_startup_with_missing_api_key_disables_provider(monkeypatch):
    monkeypatch.setattr("listener.ai_providers.GROQ_API_KEY", "")
    manager = _manager()

    assert manager.health["groq"].disabled is True
    assert manager.status_snapshot()["groq"]["status"] == "DISABLED"
    assert manager.status_snapshot()["groq"]["api_key_configured"] is False

    # A disabled provider must never be attempted, even as a fallback.
    with patch.object(
        manager.gemini, "generate", new=AsyncMock(side_effect=TransientProviderError("down"))
    ), patch.object(manager.groq, "generate", new=AsyncMock()) as groq_generate, patch(
        "listener.ai_providers.asyncio.sleep", new=AsyncMock()
    ):
        verdict = await _get_verdict(manager)

    groq_generate.assert_not_called()
    assert verdict is None


@pytest.mark.asyncio
async def test_startup_with_valid_api_keys_marks_both_healthy(monkeypatch):
    monkeypatch.setattr("listener.ai_providers.GEMINI_API_KEY", "fake-gemini-key")
    monkeypatch.setattr("listener.ai_providers.GROQ_API_KEY", "fake-groq-key")
    manager = _manager()

    assert manager.health["gemini"].disabled is False
    assert manager.health["groq"].disabled is False
    snapshot = manager.status_snapshot()
    assert snapshot["gemini"]["api_key_configured"] is True
    assert snapshot["groq"]["api_key_configured"] is True


@pytest.mark.asyncio
async def test_unhealthy_provider_skipped_without_any_retry_attempt():
    """Once a provider is in an open circuit breaker (not yet probe-ready),
    the manager must not call generate() at all — no wasted retries against
    a provider already known to be unavailable.
    """
    manager = _manager()
    health = manager.health["gemini"]
    health.healthy = False
    health.consecutive_failures = 5
    health.cooldown_until_monotonic = __import__("time").monotonic() + 900  # far in the future

    with patch.object(manager.gemini, "generate", new=AsyncMock()) as gemini_generate, patch.object(
        manager.groq, "generate", new=AsyncMock(return_value=_VALID_JSON)
    ):
        verdict = await _get_verdict(manager)

    gemini_generate.assert_not_called()
    assert verdict is not None
    assert verdict.provider == "groq"


@pytest.mark.asyncio
async def test_background_recovery_probes_only_after_cooldown_expires():
    """probe_if_ready() (what the background recovery loop calls on a timer)
    must be a no-op while the cooldown hasn't expired yet, and send exactly
    one probe the moment it has — without waiting for a real user request.
    """
    manager = _manager()
    health = manager.health["gemini"]
    health.healthy = False
    health.consecutive_failures = 5
    health.cooldown_until_monotonic = __import__("time").monotonic() + 900  # not expired yet

    with patch.object(manager.gemini, "generate", new=AsyncMock(return_value=_VALID_JSON)) as gemini_generate:
        await manager.probe_if_ready(manager.gemini)
    gemini_generate.assert_not_called()
    assert manager.health["gemini"].healthy is False

    # Now expire the cooldown and confirm exactly one probe is sent, and it
    # recovers the provider.
    health.cooldown_until_monotonic = 0.0
    with patch.object(manager.gemini, "generate", new=AsyncMock(return_value=_VALID_JSON)) as gemini_generate:
        await manager.probe_if_ready(manager.gemini)
    gemini_generate.assert_called_once()
    assert manager.health["gemini"].healthy is True


@pytest.mark.asyncio
async def test_background_recovery_loop_probes_expired_providers_without_a_request(monkeypatch):
    """run_background_recovery() must probe an expired-cooldown provider on
    its own timer — no analyze_deal()/get_verdict() call involved at all.
    """
    manager = _manager()
    health = manager.health["gemini"]
    health.healthy = False
    health.consecutive_failures = 5
    health.cooldown_until_monotonic = 0.0  # already expired

    sleep_calls = 0

    async def _fake_sleep(_seconds):
        nonlocal sleep_calls
        sleep_calls += 1
        if sleep_calls > 1:
            raise asyncio.CancelledError()

    monkeypatch.setattr("listener.ai_providers.asyncio.sleep", _fake_sleep)

    with patch.object(manager.gemini, "generate", new=AsyncMock(return_value=_VALID_JSON)) as gemini_generate:
        with pytest.raises(asyncio.CancelledError):
            await manager.run_background_recovery()

    gemini_generate.assert_called_once()
    assert manager.health["gemini"].healthy is True


def test_startup_summary_logged_once(caplog):
    import logging

    with caplog.at_level(logging.INFO, logger="fanzi.listener.ai_providers"):
        manager = _manager()
        manager.log_startup_summary()

    summary_logs = [r for r in caplog.records if "AI Providers" in r.message]
    assert len(summary_logs) == 1
    assert "Primary provider:" in summary_logs[0].message
    assert "Fallback:" in summary_logs[0].message


@pytest.mark.asyncio
async def test_provider_statistics_persist_across_a_fresh_manager_instance():
    """Simulates a restart: stats recorded via one AIProviderManager/provider
    instance must be visible from a brand-new instance reading the same
    (test) database — proving they're not just in-memory.
    """
    manager = _manager()
    with patch.object(manager.gemini, "generate", new=AsyncMock(return_value=_VALID_JSON)):
        await _get_verdict(manager)

    import database
    from listener.ai_providers import _today

    stats_before_restart = database.get_provider_stats("gemini", _today())
    assert stats_before_restart["successful_requests"] == 1

    # A brand-new manager/provider pair, as if the process had restarted.
    fresh_manager = _manager()
    stats_after_restart = database.get_provider_stats("gemini", _today())
    assert stats_after_restart["successful_requests"] == 1
    assert fresh_manager.gemini.calls_today() == 1  # gemini_quota table also survives


@pytest.mark.asyncio
async def test_provider_statistics_track_failures_retries_and_failovers(monkeypatch):
    import database
    from listener.ai_providers import _today

    manager = _manager()
    monkeypatch.setattr("listener.ai_providers.asyncio.sleep", AsyncMock())

    with patch.object(
        manager.gemini, "generate", new=AsyncMock(side_effect=TransientProviderError("down"))
    ), patch.object(manager.groq, "generate", new=AsyncMock(return_value=_VALID_JSON)):
        verdict = await _get_verdict(manager)

    assert verdict is not None
    assert verdict.provider == "groq"

    gemini_stats = database.get_provider_stats("gemini", _today())
    groq_stats = database.get_provider_stats("groq", _today())
    assert gemini_stats["failed_requests"] == 4  # 1 initial + 3 retries
    assert gemini_stats["total_retries"] == 3
    assert groq_stats["successful_requests"] == 1
    assert groq_stats["total_failovers"] == 1
