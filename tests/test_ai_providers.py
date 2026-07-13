"""Covers the dual-provider (Gemini primary, Groq fallback) architecture in
listener/ai_providers.py: provider selection order, retry policy, circuit
breaker activation/recovery, quota handling, and the unified verdict schema.
No real Gemini or Groq call is ever made — tests/conftest.py's
block_real_ai_providers fixture would fail loudly if one slipped through;
here we go further and patch each provider's `generate()` directly.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from listener.ai_providers import (
    AIProviderManager,
    FatalProviderError,
    GeminiProvider,
    GroqProvider,
    QuotaExhaustedError,
    TransientProviderError,
)

_VALID_JSON = (
    '{"deal_quality": "great", "reason": "43%% off a well-known brand.", '
    '"suggested_target": 4800, "category": "appliance"}'
).replace("%%", "%")


def _manager() -> AIProviderManager:
    return AIProviderManager(GeminiProvider(), GroqProvider())


async def _get_verdict(manager: AIProviderManager):
    return await manager.get_verdict(
        raw_text="test post",
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


@pytest.mark.asyncio
async def test_startup_with_missing_api_key_disables_provider(monkeypatch):
    monkeypatch.setattr("listener.ai_providers.GROQ_API_KEY", "")
    manager = _manager()

    assert manager.health["groq"].disabled is True
    assert manager.status_snapshot()["groq"]["status"] == "DISABLED (missing API key)"
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
