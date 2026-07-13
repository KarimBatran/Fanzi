"""Dual-provider AI verdict generation: Gemini and Groq, with automatic
failover between whichever is currently healthiest. Everything outside this
module (and listener.analyzer, which wraps it) is provider-agnostic —
callers only ever see analyze_deal()'s existing DealVerdict-shaped result,
plus a `provider` field saying who actually answered.

Architecture:
- GeminiProvider / GroqProvider: each knows how to make one raw API call,
  whether it has a configured API key, and how to translate its own SDK's
  exceptions into the shared ProviderError hierarchy (TransientProviderError
  = retry-worthy, QuotaExhaustedError = external quota gone, FatalProviderError
  = anything else, e.g. auth/invalid request).
- ProviderHealth: per-provider circuit breaker + latency/call bookkeeping.
  A provider with no configured API key is permanently `disabled` at
  startup — never instantiated against a real client, never attempted.
- AIProviderManager: retry policy, circuit breaker enforcement, and the
  Gemini-then-Groq selection order — the only thing listener.analyzer talks
  to. Selection is health-based, not "always try Gemini first no matter
  what": an unhealthy/quota-exhausted/disabled provider is skipped entirely
  (no wasted request), and a provider that's been in cooldown automatically
  gets exactly one probe once its cooldown expires, resuming normal use the
  moment that probe succeeds.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import re
import time
from dataclasses import dataclass, field
from datetime import date, datetime

from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types
from groq import (
    APIConnectionError as GroqAPIConnectionError,
    APIStatusError as GroqAPIStatusError,
    APITimeoutError as GroqAPITimeoutError,
    AsyncGroq,
    InternalServerError as GroqInternalServerError,
    RateLimitError as GroqRateLimitError,
)

import database
from config import (
    AI_CIRCUIT_BREAKER_COOLDOWN_SECONDS,
    AI_CIRCUIT_BREAKER_FAILURE_THRESHOLD,
    AI_RETRY_COUNT,
    AI_RETRY_INITIAL_BACKOFF_SECONDS,
    AI_RETRY_MAX_BACKOFF_SECONDS,
    GEMINI_API_KEY,
    GROQ_API_KEY,
)

logger = logging.getLogger("fanzi.listener.ai_providers")

_SYSTEM_PROMPT = (
    "You are a deal analyst for Amazon Egypt. You receive a product deal post "
    "and optional price history. Return ONLY a JSON object with these exact "
    "keys: deal_quality (one of: great/good/average/skip), reason (one "
    "sentence, English, max 15 words), suggested_target (integer EGP, 5-10% "
    "below current price for good deals, same as current for average), "
    "category (phone/headphones/laptop/accessory/cable/appliance/other). "
    "Judge based on: discount percentage mentioned, product category "
    "prestige, price vs history if available. Be conservative — most deals "
    "are average."
)

_JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)

_DISPLAY_NAME = {"gemini": "Gemini", "groq": "Groq"}

# How many recent latency samples feed the rolling average shown in /status.
_LATENCY_WINDOW = 20


def _strip_json_fence(text: str) -> str:
    return _JSON_FENCE_RE.sub("", text.strip()).strip()


class ProviderError(Exception):
    """Base class for all provider failures."""


class TransientProviderError(ProviderError):
    """Worth retrying: timeouts, connection errors, 5xx/UNAVAILABLE."""


class QuotaExhaustedError(ProviderError):
    """The provider's external quota is used up for the day — not retryable,
    and no further requests should be sent until the next reset window.
    """


class FatalProviderError(ProviderError):
    """Not retryable: auth failures, invalid requests, malformed responses."""


@dataclass
class AIVerdict:
    provider: str
    deal_quality: str
    reason: str
    suggested_target: int
    category: str


def _parse_verdict(text: str, provider: str) -> AIVerdict | None:
    try:
        data = json.loads(_strip_json_fence(text))
        return AIVerdict(
            provider=provider,
            deal_quality=str(data["deal_quality"]).strip().lower(),
            reason=str(data["reason"]).strip(),
            suggested_target=int(float(data["suggested_target"])),
            category=str(data["category"]).strip().lower(),
        )
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        logger.exception("malformed %s analysis response: %s", provider, text)
        return None


def _build_user_content(raw_text: str, title: str, price: float, discount_percent: int | None, channel_name: str, price_history: float | None) -> str:
    history_line = (
        f"Previously tracked price: {price_history:g} EGP." if price_history is not None else "No price history available."
    )
    return (
        f"{raw_text}\n\n"
        f"Parsed: title={title!r}, price={price:g} EGP, "
        f"discount={discount_percent}%, channel={channel_name}. "
        f"{history_line}"
    )


class GeminiProvider:
    name = "gemini"
    _MODEL = "gemini-flash-latest"

    def __init__(self) -> None:
        self._client: genai.Client | None = None

    def is_configured(self) -> bool:
        return bool(GEMINI_API_KEY)

    def _get_client(self) -> genai.Client:
        # Lazy singleton — constructing genai.Client() raises immediately if
        # GEMINI_API_KEY is empty, so this must not run at import time.
        if self._client is None:
            self._client = genai.Client(api_key=GEMINI_API_KEY)
        return self._client

    @staticmethod
    def _today() -> str:
        return date.today().isoformat()

    def calls_today(self) -> int:
        return database.get_gemini_quota_count(self._today())

    def quota_exhausted(self) -> bool:
        return database.get_gemini_external_quota_exhausted(self._today())

    def mark_quota_exhausted(self) -> None:
        database.mark_gemini_external_quota_exhausted(self._today())

    def record_call(self) -> None:
        database.increment_gemini_quota_count(self._today())

    async def generate(self, user_content: str) -> str:
        client = self._get_client()
        try:
            response = await client.aio.models.generate_content(
                model=self._MODEL,
                contents=user_content,
                config=genai_types.GenerateContentConfig(
                    system_instruction=_SYSTEM_PROMPT,
                    response_mime_type="application/json",
                    max_output_tokens=256,
                    # Extended thinking just burns quota/tokens for this
                    # simple structured-extraction task.
                    thinking_config=genai_types.ThinkingConfig(thinking_budget=0),
                ),
            )
        except genai_errors.APIError as exc:
            if exc.status == "RESOURCE_EXHAUSTED":
                raise QuotaExhaustedError(f"{exc.status}: {exc.message}") from exc
            if exc.status == "UNAVAILABLE":
                raise TransientProviderError(f"{exc.status}: {exc.message}") from exc
            raise FatalProviderError(f"{exc.status}: {exc.message}") from exc

        text = response.text
        if not text:
            raise FatalProviderError("empty response body")
        return text


class GroqProvider:
    name = "groq"
    _MODEL = "llama-3.1-8b-instant"

    def __init__(self) -> None:
        self._client: AsyncGroq | None = None

    def is_configured(self) -> bool:
        return bool(GROQ_API_KEY)

    def _get_client(self) -> AsyncGroq:
        if self._client is None:
            if not GROQ_API_KEY:
                raise FatalProviderError("GROQ_API_KEY not configured")
            self._client = AsyncGroq(api_key=GROQ_API_KEY)
        return self._client

    @staticmethod
    def _today() -> str:
        return date.today().isoformat()

    def calls_today(self) -> int:
        return database.get_groq_quota_count(self._today())

    def quota_exhausted(self) -> bool:
        return database.get_groq_external_quota_exhausted(self._today())

    def mark_quota_exhausted(self) -> None:
        database.mark_groq_external_quota_exhausted(self._today())

    def record_call(self) -> None:
        database.increment_groq_quota_count(self._today())

    async def generate(self, user_content: str) -> str:
        client = self._get_client()
        try:
            response = await client.chat.completions.create(
                model=self._MODEL,
                max_tokens=256,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
            )
        except GroqRateLimitError as exc:
            raise QuotaExhaustedError(str(exc)) from exc
        except (GroqInternalServerError, GroqAPITimeoutError, GroqAPIConnectionError) as exc:
            raise TransientProviderError(str(exc)) from exc
        except GroqAPIStatusError as exc:
            # Catch-all for any other status code this SDK version didn't
            # give a dedicated exception class for — 5xx is transient,
            # everything else (4xx auth/invalid-request) is fatal.
            if exc.status_code in (500, 502, 503, 504):
                raise TransientProviderError(str(exc)) from exc
            raise FatalProviderError(str(exc)) from exc

        text = response.choices[0].message.content if response.choices else None
        if not text:
            raise FatalProviderError("empty response body")
        return text


@dataclass
class ProviderHealth:
    name: str
    healthy: bool = True
    disabled: bool = False  # no API key configured at startup — permanent
    last_success: datetime | None = None
    consecutive_failures: int = 0
    last_error: str | None = None
    last_latency_ms: float | None = None
    avg_latency_ms: float | None = None
    cooldown_until_monotonic: float | None = None
    _minute_call_times: list[float] = field(default_factory=list)
    _latency_samples: list[float] = field(default_factory=list)

    def calls_this_minute(self) -> int:
        cutoff = time.monotonic() - 60
        self._minute_call_times = [t for t in self._minute_call_times if t > cutoff]
        return len(self._minute_call_times)

    def record_attempt(self) -> None:
        self._minute_call_times.append(time.monotonic())

    def record_success(self, latency_ms: float) -> None:
        self.healthy = True
        self.consecutive_failures = 0
        self.last_error = None
        self.last_success = datetime.now()
        self.last_latency_ms = latency_ms
        self.cooldown_until_monotonic = None
        self._latency_samples.append(latency_ms)
        if len(self._latency_samples) > _LATENCY_WINDOW:
            self._latency_samples.pop(0)
        self.avg_latency_ms = sum(self._latency_samples) / len(self._latency_samples)

    def record_failure(self, error: str) -> None:
        self.consecutive_failures += 1
        self.last_error = error
        if self.consecutive_failures >= AI_CIRCUIT_BREAKER_FAILURE_THRESHOLD:
            if self.healthy:
                logger.warning(
                    "%s entered circuit breaker after %d consecutive failures — "
                    "cooling down %ds",
                    _DISPLAY_NAME[self.name],
                    self.consecutive_failures,
                    AI_CIRCUIT_BREAKER_COOLDOWN_SECONDS,
                )
            self.healthy = False
            self.cooldown_until_monotonic = time.monotonic() + AI_CIRCUIT_BREAKER_COOLDOWN_SECONDS

    def cooldown_remaining_seconds(self) -> float:
        if self.healthy or self.cooldown_until_monotonic is None:
            return 0.0
        return max(0.0, self.cooldown_until_monotonic - time.monotonic())

    def is_probe_ready(self) -> bool:
        return (
            not self.healthy
            and self.cooldown_until_monotonic is not None
            and time.monotonic() >= self.cooldown_until_monotonic
        )

    def can_attempt(self) -> tuple[bool, bool]:
        """Returns (can_attempt, is_probe)."""
        if self.disabled:
            return False, False
        if self.healthy:
            return True, False
        if self.is_probe_ready():
            return True, True
        return False, False


class AIProviderManager:
    """Selects and fails over between GeminiProvider and GroqProvider. The
    only entry point the rest of the app needs is get_verdict().
    """

    def __init__(self, gemini: GeminiProvider, groq: GroqProvider) -> None:
        self.gemini = gemini
        self.groq = groq
        self.health: dict[str, ProviderHealth] = {
            gemini.name: ProviderHealth(gemini.name),
            groq.name: ProviderHealth(groq.name),
        }
        self.last_provider_used: str | None = None

        # Startup validation: a provider with no configured API key is
        # permanently disabled — never instantiated against a real client,
        # never attempted, regardless of circuit-breaker/cooldown state.
        for provider in (self.gemini, self.groq):
            health = self.health[provider.name]
            if not provider.is_configured():
                health.disabled = True
                logger.warning("%s DISABLED (missing API key)", _DISPLAY_NAME[provider.name])
            else:
                logger.info("%s HEALTHY (API key configured)", _DISPLAY_NAME[provider.name])

    async def _attempt(self, provider, user_content: str, max_attempts: int) -> AIVerdict | None:
        """Calls `provider` up to max_attempts times (1 = no retry, used for
        circuit-breaker probes), retrying only TransientProviderError with
        exponential backoff + jitter. Returns the parsed verdict on success,
        or None if every attempt failed / the breaker tripped mid-retry.
        Quota exhaustion and fatal errors stop immediately (never retried).
        """
        health = self.health[provider.name]
        attempt = 0
        while True:
            health.record_attempt()
            start = time.monotonic()
            try:
                text = await provider.generate(user_content)
            except QuotaExhaustedError as exc:
                # Quota exhaustion is tracked independently of the transient-
                # failure circuit breaker (it's already a hard, deterministic
                # gate via quota_exhausted() for the rest of the day) — don't
                # also count it toward consecutive_failures/health.healthy.
                provider.mark_quota_exhausted()
                health.last_error = str(exc)
                logger.warning("%s quota exhausted", _DISPLAY_NAME[provider.name])
                return None
            except TransientProviderError as exc:
                health.record_failure(str(exc))
                attempt += 1
                if attempt >= max_attempts or not health.healthy:
                    logger.error(
                        "%s failed after %d attempt(s): %s", _DISPLAY_NAME[provider.name], attempt, exc
                    )
                    return None
                backoff = min(
                    AI_RETRY_INITIAL_BACKOFF_SECONDS * (2 ** (attempt - 1)), AI_RETRY_MAX_BACKOFF_SECONDS
                )
                jitter = random.uniform(0, backoff * 0.1)
                logger.warning(
                    "%s transient failure (attempt %d/%d) — retrying in %.2fs",
                    _DISPLAY_NAME[provider.name],
                    attempt,
                    max_attempts,
                    backoff + jitter,
                )
                await asyncio.sleep(backoff + jitter)
                continue
            except FatalProviderError as exc:
                health.record_failure(str(exc))
                logger.error("%s failed (non-retryable): %s", _DISPLAY_NAME[provider.name], exc)
                return None

            latency_ms = (time.monotonic() - start) * 1000
            verdict = _parse_verdict(text, provider.name)
            if verdict is None:
                health.record_failure("malformed response")
                return None
            was_unhealthy = not health.healthy
            health.record_success(latency_ms)
            provider.record_call()
            if was_unhealthy:
                logger.info("%s recovered after cooldown", _DISPLAY_NAME[provider.name])
            return verdict

    async def get_verdict(
        self, raw_text: str, title: str, price: float, discount_percent: int | None,
        channel_name: str, price_history: float | None,
    ) -> AIVerdict | None:
        selection_start = time.monotonic()
        user_content = _build_user_content(raw_text, title, price, discount_percent, channel_name, price_history)

        gemini_verdict = await self._try_provider(self.gemini, user_content)
        if gemini_verdict is not None:
            self.last_provider_used = "gemini"
            logger.info("AIProviderManager selected Gemini")
            self._log_selection_time(selection_start)
            return gemini_verdict

        groq_verdict = await self._try_provider(self.groq, user_content)
        if groq_verdict is not None:
            self.last_provider_used = "groq"
            logger.info("AIProviderManager selected Groq (Gemini unavailable)")
            self._log_selection_time(selection_start)
            return groq_verdict

        logger.info("AIProviderManager selected none (both providers unavailable)")
        self._log_selection_time(selection_start)
        return None

    @staticmethod
    def _log_selection_time(selection_start: float) -> None:
        elapsed_ms = (time.monotonic() - selection_start) * 1000
        logger.info("Provider selection completed in %.0f ms", elapsed_ms)

    async def _try_provider(self, provider, user_content: str) -> AIVerdict | None:
        health = self.health[provider.name]
        if health.disabled:
            logger.info("%s disabled (no API key) — skipping", _DISPLAY_NAME[provider.name])
            return None

        if provider.quota_exhausted():
            logger.info("%s quota exhausted for today — skipping", _DISPLAY_NAME[provider.name])
            return None

        can_attempt, is_probe = health.can_attempt()
        if not can_attempt:
            logger.info("%s in cooldown — skipping", _DISPLAY_NAME[provider.name])
            return None

        if is_probe:
            logger.info("%s cooldown expired — sending single probe request", _DISPLAY_NAME[provider.name])
            return await self._attempt(provider, user_content, max_attempts=1)

        return await self._attempt(provider, user_content, max_attempts=AI_RETRY_COUNT + 1)

    def status_snapshot(self) -> dict:
        def _status_label(provider, health: ProviderHealth) -> str:
            if health.disabled:
                return "DISABLED (missing API key)"
            if provider.quota_exhausted():
                return "QUOTA EXHAUSTED"
            if not health.healthy and not health.is_probe_ready():
                return "UNHEALTHY"
            return "HEALTHY"

        def _provider_info(provider, health: ProviderHealth) -> dict:
            cooldown_seconds = health.cooldown_remaining_seconds()
            return {
                "status": _status_label(provider, health),
                "calls_today": provider.calls_today(),
                "consecutive_failures": health.consecutive_failures,
                "last_success": health.last_success,
                "avg_latency_ms": health.avg_latency_ms,
                "cooldown_remaining_seconds": cooldown_seconds if cooldown_seconds > 0 else None,
                "quota_available": not provider.quota_exhausted(),
                "api_key_configured": not health.disabled,
            }

        gemini_health = self.health["gemini"]
        groq_health = self.health["groq"]

        def _usable(provider, health: ProviderHealth) -> bool:
            can_attempt, _ = health.can_attempt()
            return can_attempt and not provider.quota_exhausted()

        if _usable(self.gemini, gemini_health):
            current_provider = "Gemini"
        elif _usable(self.groq, groq_health):
            current_provider = "Groq"
        else:
            current_provider = "None"

        return {
            "gemini": _provider_info(self.gemini, gemini_health),
            "groq": _provider_info(self.groq, groq_health),
            "current_provider": current_provider,
            "fallback": "Groq",
            "last_provider_used": _DISPLAY_NAME.get(self.last_provider_used, "None"),
        }

    def both_quota_exhausted(self) -> bool:
        return self.gemini.quota_exhausted() and self.groq.quota_exhausted()

    def both_unavailable(self) -> bool:
        gemini_can, _ = self.health["gemini"].can_attempt()
        groq_can, _ = self.health["groq"].can_attempt()
        return (not gemini_can or self.gemini.quota_exhausted()) and (not groq_can or self.groq.quota_exhausted())


_manager = AIProviderManager(GeminiProvider(), GroqProvider())


def get_manager() -> AIProviderManager:
    return _manager
