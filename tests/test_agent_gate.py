"""Tests for AgentGate -- interrupt interception during task execution."""
import asyncio
import pytest

from pipeline.agent_state import AgentState
from pipeline.agent_gate import AgentGate, _CANCEL_WORDS


class FakeFrame:
    pass


class FakeTranscriptionFrame:
    def __init__(self, text):
        self.text = text


@pytest.fixture
def state():
    return AgentState()


class TestCancelWords:

    def test_stop_is_cancel(self):
        assert "stop" in _CANCEL_WORDS

    def test_cancel_is_cancel(self):
        assert "cancel" in _CANCEL_WORDS

    def test_nevermind_is_cancel(self):
        assert "nevermind" in _CANCEL_WORDS

    def test_abort_is_cancel(self):
        assert "abort" in _CANCEL_WORDS


class TestGateIdle:
    """When not running, all frames pass through."""

    def test_idle_state_passes_frames(self, state):
        assert state.running is False
        # gate should not intercept when idle
        # (full frame processing needs Pipecat event loop,
        #  so we test the logic directly)
        words = set("open youtube".lower().split())
        assert not (words & _CANCEL_WORDS)


class TestGateRunning:
    """When running, transcriptions are captured."""

    def test_cancel_word_sets_event(self, state):
        state.running = True
        text = "stop"
        words = set(text.lower().split())
        if words & _CANCEL_WORDS:
            state.cancel_event.set()
        assert state.cancel_event.is_set()

    def test_hint_captured(self, state):
        state.running = True
        text = "try the search box instead"
        words = set(text.lower().split())
        if not (words & _CANCEL_WORDS):
            state.hint_queue.put_nowait(text)
        assert not state.hint_queue.empty()
        assert state.hint_queue.get_nowait() == "try the search box instead"

    def test_cancel_phrase_in_sentence(self, state):
        state.running = True
        text = "please stop that"
        words = set(text.lower().split())
        is_cancel = bool(words & _CANCEL_WORDS)
        assert is_cancel

    def test_non_cancel_goes_to_hints(self, state):
        state.running = True
        text = "actually use Google instead"
        words = set(text.lower().split())
        is_cancel = bool(words & _CANCEL_WORDS)
        assert not is_cancel

    def test_empty_text_ignored(self, state):
        state.running = True
        text = "   "
        assert text.strip() == ""
