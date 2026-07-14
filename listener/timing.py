"""High-resolution per-deal timing instrumentation for the forwarding
pipeline. A single DealTiming instance is created once per incoming post
(listener/watcher.py) and threaded optionally through
listener.analyzer.analyze_deal() -> listener.ai_providers.AIProviderManager,
so every stage records into the same object without changing any of those
functions' required positional signatures (the `timing` parameter is
keyword-only and defaults to None everywhere).

Overhead is one time.perf_counter() call and a dict write per stage —
negligible even on a slow deal. PERFORMANCE_LOGGING only gates whether the
per-deal summary line is emitted at INFO; the slow-request WARNING (Phase 3)
always fires regardless, so a config mistake can't hide real outliers.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from config import PERFORMANCE_LOGGING, SLOW_REQUEST_THRESHOLD_SECONDS

logger = logging.getLogger("fanzi.listener.timing")

# Display order for the per-deal summary — anything recorded outside this
# list (there shouldn't be, but stages evolve) still prints, just after.
_STAGE_ORDER = (
    "receive",
    "parser",
    "redirect",
    "dedup",
    "rule_lookup",
    "ai_selection",
    "gemini",
    "groq",
    "ai_parse",
    "learning_enqueue",
    "telegram_send",
)

_STAGE_LABEL = {
    "receive": "Receive",
    "parser": "Parser",
    "redirect": "Redirect",
    "dedup": "Dedup",
    "rule_lookup": "Rule lookup",
    "ai_selection": "AI selection",
    "gemini": "Gemini",
    "groq": "Groq",
    "ai_parse": "AI parse",
    "learning_enqueue": "Learning enqueue",
    "telegram_send": "Telegram send",
}


@dataclass
class DealTiming:
    asin: str = ""
    title: str = ""
    channel: str = ""
    durations_ms: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)  # e.g. "Gemini failed after 930 ms"
    _start: float = field(default_factory=time.perf_counter)

    def record(self, stage: str, duration_ms: float) -> None:
        self.durations_ms[stage] = self.durations_ms.get(stage, 0.0) + duration_ms

    def note(self, text: str) -> None:
        self.notes.append(text)

    def total_ms(self) -> float:
        return (time.perf_counter() - self._start) * 1000

    def stage_group_total(self, stages: tuple[str, ...]) -> float:
        return sum(self.durations_ms.get(s, 0.0) for s in stages)

    def ai_total_ms(self) -> float:
        return self.stage_group_total(("ai_selection", "gemini", "groq", "ai_parse"))

    def network_total_ms(self) -> float:
        return self.stage_group_total(("redirect", "gemini", "groq", "telegram_send"))

    def database_total_ms(self) -> float:
        return self.stage_group_total(("dedup", "rule_lookup"))

    def parsing_total_ms(self) -> float:
        return self.stage_group_total(("parser",))

    def slowest_stage(self) -> tuple[str, float] | None:
        if not self.durations_ms:
            return None
        stage = max(self.durations_ms, key=self.durations_ms.get)
        return stage, self.durations_ms[stage]

    def format_summary(self) -> str:
        total = self.total_ms()
        lines = ["Deal timing"]
        for stage in _STAGE_ORDER:
            if stage in self.durations_ms:
                label = _STAGE_LABEL.get(stage, stage)
                lines.append(f"{label + ':':<14}{self.durations_ms[stage]:>7.0f} ms")
        for extra in self.notes:
            lines.append(extra)
        lines.append(f"{'TOTAL:':<14}{total:>7.0f} ms")
        return "\n".join(lines)

    def log_summary(self) -> None:
        if PERFORMANCE_LOGGING:
            logger.info(self.format_summary())
        self._maybe_warn_slow()

    def _maybe_warn_slow(self) -> None:
        total = self.total_ms()
        if total <= SLOW_REQUEST_THRESHOLD_SECONDS * 1000:
            return
        slowest = self.slowest_stage()
        slowest_desc = (
            f"{_STAGE_LABEL.get(slowest[0], slowest[0])} ({slowest[1]:.0f} ms)" if slowest else "unknown"
        )
        logger.warning(
            "SLOW DEAL: title=%r channel=%s asin=%s total=%.0fms slowest=%s\n%s",
            self.title,
            self.channel,
            self.asin,
            total,
            slowest_desc,
            self.format_summary(),
        )
