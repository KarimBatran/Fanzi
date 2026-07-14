"""Covers listener/replay.py: startup/reconnect replay recovers messages
missed while offline, replay is idempotent (dedup prevents duplicate
notifications), replay never advances its checkpoint past a message that
raised, and replay is independent per channel. Also covers the real
listener/watcher.py pipeline end-to-end (mocked AI) to prove replayed
messages behave exactly like live ones. Zero real Gemini/Groq calls.
"""

from __future__ import annotations

import asyncio
from datetime import date
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import database
from listener import replay
from listener.analyzer import DealVerdict

TODAY = date.today().isoformat()


class _FakePeerId:
    def __init__(self, channel_id):
        self.channel_id = channel_id


class _FakeMessage:
    def __init__(self, msg_id, text, channel_id=123):
        self.id = msg_id
        self.message = text
        self.peer_id = _FakePeerId(channel_id)


class _FakeClient:
    def __init__(self, messages):
        self._messages = messages
        self.get_messages_calls = 0

    async def get_messages(self, channel, limit=50):
        self.get_messages_calls += 1
        return list(self._messages)


def _deal_text(asin: str, price: int = 500) -> str:
    return f"Test deal price: {price} EGP\nhttps://www.amazon.eg/dp/{asin}"


def _checkpoint_advancing_stub():
    """A minimal handle_post stub that mimics the one real side effect of
    listener.watcher._handle_post that replay.py itself relies on: advancing
    the replay checkpoint on success. Real checkpoint semantics (success vs.
    exception) are covered separately using the real _handle_post; these
    tests exercise replay_channel's own ordering/counting/idempotency logic.
    """
    calls = []

    async def _stub(bot, text, channel, message_id=None, channel_id=None):
        calls.append(message_id)
        database.set_channel_replay_state(channel, channel_id, message_id, "2026-01-01T00:00:00")

    _stub.calls = calls
    return _stub


@pytest.mark.asyncio
async def test_replay_recovers_messages_missed_while_offline():
    """Simulates: bot processed nothing yet for this channel, three deals
    were posted while it was offline, restart runs replay and recovers all
    three exactly once, in chronological order.
    """
    messages = [
        _FakeMessage(101, _deal_text("B0REPLAY01")),
        _FakeMessage(102, _deal_text("B0REPLAY02")),
        _FakeMessage(103, _deal_text("B0REPLAY03")),
    ]
    client = _FakeClient(messages)
    bot = AsyncMock()
    handle_post = _checkpoint_advancing_stub()

    recovered = await replay.replay_channel(client, bot, "replay_channel_1", handle_post, fetch_limit=50)

    assert recovered == 3
    # Chronological order: oldest message id first.
    assert handle_post.calls == [101, 102, 103]

    state = database.get_channel_replay_state("replay_channel_1")
    assert state["last_message_id"] == 103


@pytest.mark.asyncio
async def test_replay_is_idempotent_second_run_recovers_nothing():
    messages = [_FakeMessage(201, _deal_text("B0REPLAY04")), _FakeMessage(202, _deal_text("B0REPLAY05"))]
    client = _FakeClient(messages)
    bot = AsyncMock()
    handle_post = _checkpoint_advancing_stub()

    first = await replay.replay_channel(client, bot, "replay_channel_2", handle_post, fetch_limit=50)
    second = await replay.replay_channel(client, bot, "replay_channel_2", handle_post, fetch_limit=50)

    assert first == 2
    assert second == 0  # nothing new since the checkpoint already advanced
    assert len(handle_post.calls) == 2  # not called again for the same messages


@pytest.mark.asyncio
async def test_replay_end_to_end_no_duplicate_notification_via_real_pipeline():
    """Uses the real listener.watcher._handle_post pipeline (mocked AI) so
    dedup is the real thing guaranteeing idempotency, not just the replay
    checkpoint.
    """
    import listener.watcher as watcher_module

    text = _deal_text("B0REPLAYE2E")
    messages = [_FakeMessage(301, text)]
    client = _FakeClient(messages)
    msg = MagicMock()
    msg.chat_id, msg.message_id = 1, 2
    bot = AsyncMock()
    bot.send_message = AsyncMock(return_value=msg)

    verdict = DealVerdict(deal_quality="great", reason="x", suggested_target=450, category="other", provider="gemini")
    with patch.object(watcher_module, "analyze_deal", new=AsyncMock(return_value=verdict)):
        recovered = await replay.replay_channel(client, bot, "e2e_channel", watcher_module._handle_post, fetch_limit=50)
        # Re-running replay (e.g. a reconnect happening moments later) must
        # not re-forward the same deal.
        recovered_again = await replay.replay_channel(
            client, bot, "e2e_channel", watcher_module._handle_post, fetch_limit=50
        )

    assert recovered == 1
    assert recovered_again == 0
    bot.send_message.assert_called_once()  # exactly one notification, ever


@pytest.mark.asyncio
async def test_replay_does_not_advance_checkpoint_past_a_failing_message():
    """replay_channel itself only counts recovered messages -- the actual
    checkpoint is set by _handle_post on success (see the next test for
    that, using the real pipeline). Here: a stub that always raises must
    stop replay immediately with zero recovered.
    """
    messages = [_FakeMessage(401, _deal_text("B0REPLAY06")), _FakeMessage(402, _deal_text("B0REPLAY07"))]
    client = _FakeClient(messages)
    bot = AsyncMock()

    async def _failing_handle_post(bot, text, channel, message_id=None, channel_id=None):
        raise RuntimeError("simulated unexpected failure")

    recovered = await replay.replay_channel(client, bot, "failing_channel", _failing_handle_post, fetch_limit=50)

    assert recovered == 0


@pytest.mark.asyncio
async def test_handle_post_checkpoint_does_not_advance_past_an_exception():
    """The real listener.watcher._handle_post is what actually sets the
    replay checkpoint, and only after successful completion -- an
    unexpected exception during processing must leave the checkpoint at
    whatever it was before this message (here: unset).
    """
    import listener.watcher as watcher_module

    with patch.object(watcher_module, "_process_post", new=AsyncMock(side_effect=RuntimeError("boom"))):
        with pytest.raises(RuntimeError):
            await watcher_module._handle_post(
                AsyncMock(), _deal_text("B0CHECKPOINT1"), "checkpoint_fail_channel", message_id=999
            )

    assert database.get_channel_replay_state("checkpoint_fail_channel") is None


@pytest.mark.asyncio
async def test_replay_stops_at_first_failure_but_keeps_earlier_successes():
    """Uses the real _handle_post (mocked AI) so the checkpoint left behind
    reflects genuine success/failure semantics, not a bare test stub.
    """
    import listener.watcher as watcher_module

    messages = [
        _FakeMessage(501, _deal_text("B0REPLAY08")),
        _FakeMessage(502, "this one will raise"),
        _FakeMessage(503, _deal_text("B0REPLAY09")),
    ]
    client = _FakeClient(messages)
    bot = AsyncMock()
    bot.send_message = AsyncMock(return_value=MagicMock(chat_id=1, message_id=2))

    verdict = DealVerdict(deal_quality="great", reason="x", suggested_target=450, category="other", provider="gemini")
    original_process_post = watcher_module._process_post

    async def _process_post_maybe_fail(bot, text, channel, stat_date, timing):
        if "this one will raise" in text:
            raise RuntimeError("simulated failure on message 502")
        return await original_process_post(bot, text, channel, stat_date, timing)

    with patch.object(watcher_module, "analyze_deal", new=AsyncMock(return_value=verdict)), patch.object(
        watcher_module, "_process_post", new=_process_post_maybe_fail
    ):
        recovered = await replay.replay_channel(
            client, bot, "partial_fail_channel", watcher_module._handle_post, fetch_limit=50
        )

    assert recovered == 1  # only message 501 succeeded before the failure
    state = database.get_channel_replay_state("partial_fail_channel")
    assert state["last_message_id"] == 501  # checkpoint stopped exactly at the last real success


@pytest.mark.asyncio
async def test_replay_is_independent_per_channel():
    messages_a = [_FakeMessage(601, _deal_text("B0REPLAYA1"))]
    messages_b = [_FakeMessage(701, _deal_text("B0REPLAYB1"))]
    bot = AsyncMock()
    handle_post = _checkpoint_advancing_stub()

    await replay.replay_channel(_FakeClient(messages_a), bot, "channel_a", handle_post, fetch_limit=50)
    await replay.replay_channel(_FakeClient(messages_b), bot, "channel_b", handle_post, fetch_limit=50)

    state_a = database.get_channel_replay_state("channel_a")
    state_b = database.get_channel_replay_state("channel_b")
    assert state_a["last_message_id"] == 601
    assert state_b["last_message_id"] == 701


@pytest.mark.asyncio
async def test_replay_all_logs_summary_and_updates_status(caplog):
    import logging

    channels = ["chan_x", "chan_y"]
    bot = AsyncMock()
    handle_post = AsyncMock()

    def _client_for(channel):
        if channel == "chan_x":
            return _FakeMessage(1, _deal_text("B0STATUSX"))
        return None

    class _MultiClient:
        async def get_messages(self, channel, limit=50):
            m = _client_for(channel)
            return [m] if m else []

    with caplog.at_level(logging.INFO, logger="fanzi.listener.replay"):
        results = await replay.replay_all(_MultiClient(), bot, channels, handle_post, reason="startup")

    assert results == {"chan_x": 1, "chan_y": 0}
    assert any("Replay complete" in r.message for r in caplog.records)
    assert any("Replay started (startup)" in r.message for r in caplog.records)

    status = replay.get_status()
    assert status["state"] == "idle"
    assert status["last_replay_at"] is not None
    assert status["recovered_today"] >= 1


@pytest.mark.asyncio
async def test_reconnect_triggers_replay_on_reconnect_transition(monkeypatch):
    connection_states = iter([True, False, False, True])  # connected, drops, still down, reconnects

    class _FakeReconnectClient:
        def is_connected(self):
            return next(connection_states)

        async def get_messages(self, channel, limit=50):
            return []

    client = _FakeReconnectClient()
    bot = AsyncMock()
    handle_post = AsyncMock()

    call_count = 0

    async def _fast_sleep(_seconds):
        nonlocal call_count
        call_count += 1
        # Let 3 sleep+is_connected iterations complete (consuming False,
        # False, True from connection_states -- the last one is the
        # reconnect transition) before stopping the loop.
        if call_count >= 4:
            raise asyncio.CancelledError()

    monkeypatch.setattr("listener.replay.asyncio.sleep", _fast_sleep)

    with patch("listener.replay.replay_all", new=AsyncMock(wraps=replay.replay_all)) as replay_all_mock:
        with pytest.raises(asyncio.CancelledError):
            await replay.watch_for_reconnects(client, bot, ["some_channel"], handle_post, poll_interval=0)

    replay_all_mock.assert_called_once()
    assert replay_all_mock.call_args.kwargs["reason"] == "reconnect"


def test_status_section_appears_in_status_message():
    import health

    with patch("health.channels_store.get_effective_channels", return_value=[]):
        message = health.format_status_message()

    assert "Replay" in message
    assert "Last replay:" in message
    assert "Recovered messages today:" in message
    assert "Replay state:" in message
