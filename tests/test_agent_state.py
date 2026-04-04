"""Tests for AgentState."""
import asyncio
import pytest

from pipeline.agent_state import AgentState


def test_initial_state():
    state = AgentState()
    assert state.running is False
    assert state.task_type == ""
    assert not state.cancel_event.is_set()
    assert state.hint_queue.empty()


def test_reset_clears_all():
    state = AgentState()
    state.running = True
    state.task_type = "screen"
    state.cancel_event.set()
    state.hint_queue.put_nowait("try the search box")

    state.reset()

    assert state.running is False
    assert state.task_type == ""
    assert not state.cancel_event.is_set()
    assert state.hint_queue.empty()


def test_reset_drains_multiple_hints():
    state = AgentState()
    state.hint_queue.put_nowait("hint 1")
    state.hint_queue.put_nowait("hint 2")
    state.hint_queue.put_nowait("hint 3")

    state.reset()
    assert state.hint_queue.empty()


def test_reset_on_clean_state():
    state = AgentState()
    state.reset()
    assert state.running is False
    assert state.hint_queue.empty()
