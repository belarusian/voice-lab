"""Tests for tool schemas and registration."""
import pytest

from pipeline.dispatch import build_tools, register_tools, _SCREEN_TOOL, _CODE_TOOL
from pipeline.agent_state import AgentState


class TestBuildTools:

    def test_no_functions_no_tools(self):
        assert build_tools() == []

    def test_screen_only(self):
        tools = build_tools(screen_fn=lambda i: "ok")
        assert len(tools) == 1
        assert tools[0]["function"]["name"] == "screen_action"

    def test_code_only(self):
        tools = build_tools(code_fn=lambda i, w: "ok")
        assert len(tools) == 1
        assert tools[0]["function"]["name"] == "code_action"

    def test_both(self):
        tools = build_tools(screen_fn=lambda i: "ok", code_fn=lambda i, w: "ok")
        assert len(tools) == 2
        names = {t["function"]["name"] for t in tools}
        assert names == {"screen_action", "code_action"}


class TestToolSchemas:

    def test_screen_action_schema(self):
        assert _SCREEN_TOOL["type"] == "function"
        assert _SCREEN_TOOL["function"]["name"] == "screen_action"
        params = _SCREEN_TOOL["function"]["parameters"]
        assert "instruction" in params["properties"]
        assert "instruction" in params["required"]

    def test_code_action_schema(self):
        assert _CODE_TOOL["type"] == "function"
        assert _CODE_TOOL["function"]["name"] == "code_action"
        params = _CODE_TOOL["function"]["parameters"]
        assert "instruction" in params["properties"]
        assert "instruction" in params["required"]

    def test_schemas_are_openai_compatible(self):
        for tool in [_SCREEN_TOOL, _CODE_TOOL]:
            assert "type" in tool
            assert "function" in tool
            fn = tool["function"]
            assert "name" in fn
            assert "description" in fn
            assert "parameters" in fn
            assert fn["parameters"]["type"] == "object"


class TestRegisterTools:

    def test_registers_screen_only(self):
        registered = {}

        class FakeLLM:
            def register_function(self, name, handler, cancel_on_interruption=True):
                registered[name] = handler

        register_tools(FakeLLM(), AgentState(), screen_fn=lambda i: "ok")
        assert "screen_action" in registered
        assert "code_action" not in registered

    def test_registers_both(self):
        registered = {}

        class FakeLLM:
            def register_function(self, name, handler, cancel_on_interruption=True):
                registered[name] = handler

        register_tools(FakeLLM(), AgentState(),
                       screen_fn=lambda i: "ok", code_fn=lambda i, w: "ok")
        assert "screen_action" in registered
        assert "code_action" in registered

    def test_registers_nothing_without_functions(self):
        registered = {}

        class FakeLLM:
            def register_function(self, name, handler, cancel_on_interruption=True):
                registered[name] = handler

        register_tools(FakeLLM(), AgentState())
        assert registered == {}

    def test_cancel_on_interruption_false(self):
        flags = {}

        class FakeLLM:
            def register_function(self, name, handler, cancel_on_interruption=True):
                flags[name] = cancel_on_interruption

        register_tools(FakeLLM(), AgentState(),
                       screen_fn=lambda i: "ok", code_fn=lambda i, w: "ok")
        assert flags["screen_action"] is False
        assert flags["code_action"] is False
