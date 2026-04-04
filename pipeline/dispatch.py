"""Tool definitions and function call dispatch for the voice agent.

Uses Pipecat's native function calling: the LLM emits tool_calls,
Pipecat dispatches to registered handlers, results go back to the
LLM which formulates a natural spoken response.

The actual tool implementations (screen_fn, code_fn) are supplied
by the caller -- voice-lab never imports the subsystems directly.
"""
import asyncio
from concurrent.futures import ThreadPoolExecutor

from pipeline.agent_state import AgentState

_executor = ThreadPoolExecutor(max_workers=1)

# -- OpenAI-compatible tool schemas --

_SCREEN_TOOL = {
    "type": "function",
    "function": {
        "name": "screen_action",
        "description": (
            "Execute a screen action using the browser or desktop app. "
            "Use when the user asks to open a website, click something, "
            "find information visually, or interact with an application."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "instruction": {
                    "type": "string",
                    "description": "Natural language instruction for the screen action",
                }
            },
            "required": ["instruction"],
        },
    },
}

_CODE_TOOL = {
    "type": "function",
    "function": {
        "name": "code_action",
        "description": (
            "Execute code, analyze data, run a program, or investigate "
            "something computationally. Use when the user asks to write "
            "code, calculate something, or do research."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "instruction": {
                    "type": "string",
                    "description": "Natural language instruction for the code task",
                },
                "workspace": {
                    "type": "string",
                    "description": (
                        "Directory path for the project to work in. "
                        "Omit for general questions with no project context."
                    ),
                },
            },
            "required": ["instruction"],
        },
    },
}


def build_tools(*, screen_fn=None, code_fn=None):
    """Return tool schemas for whichever functions are supplied."""
    tools = []
    if screen_fn is not None:
        tools.append(_SCREEN_TOOL)
    if code_fn is not None:
        tools.append(_CODE_TOOL)
    return tools


def register_tools(llm, state: AgentState, *, screen_fn=None, code_fn=None):
    """Register tool handlers with the LLM service.

    screen_fn(instruction: str) -> str
    code_fn(instruction: str, workspace: str | None) -> str

    cancel_on_interruption=False because we handle interruptions
    ourselves through the AgentGate.
    """

    if screen_fn is not None:
        async def handle_screen(params):
            state.running = True
            state.task_type = "screen"
            try:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    _executor, screen_fn,
                    params.arguments["instruction"],
                )
                if state.cancel_event.is_set():
                    result = "Cancelled."
                await params.result_callback(result)
            except Exception as e:
                await params.result_callback(f"Screen action failed: {e}")
            finally:
                state.reset()

        llm.register_function("screen_action", handle_screen,
                              cancel_on_interruption=False)

    if code_fn is not None:
        async def handle_code(params):
            state.running = True
            state.task_type = "code"
            try:
                loop = asyncio.get_event_loop()
                workspace = params.arguments.get("workspace")
                result = await loop.run_in_executor(
                    _executor, code_fn,
                    params.arguments["instruction"], workspace,
                )
                if state.cancel_event.is_set():
                    result = "Cancelled."
                await params.result_callback(result)
            except Exception as e:
                await params.result_callback(f"Code action failed: {e}")
            finally:
                state.reset()

        llm.register_function("code_action", handle_code,
                              cancel_on_interruption=False)
