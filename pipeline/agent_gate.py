"""Agent gate -- intercepts transcriptions while a task is running.

When idle, all frames pass through normally. When a screen or code task
is executing, incoming TranscriptionFrames are captured as hints (or
cancel signals) instead of flowing to the LLM for a new generation.
"""
from pipecat.frames.frames import Frame, TextFrame, TranscriptionFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from pipeline.agent_state import AgentState

_CANCEL_WORDS = {"stop", "cancel", "nevermind", "never mind", "abort", "quit"}


class AgentGate(FrameProcessor):

    def __init__(self, state: AgentState, **kwargs):
        super().__init__(**kwargs)
        self._state = state

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame) and self._state.running:
            text = frame.text.strip()
            if not text:
                await self.push_frame(frame, direction)
                return

            # Check for cancel intent
            words = set(text.lower().split())
            if words & _CANCEL_WORDS:
                self._state.cancel_event.set()
                return

            # Otherwise capture as hint for the running task
            await self._state.hint_queue.put(text)
            return

        # Not running or not a transcription -- pass through
        await self.push_frame(frame, direction)
