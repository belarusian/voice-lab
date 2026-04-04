"""Shared state between voice agent pipeline processors."""
import asyncio
from dataclasses import dataclass, field


@dataclass
class AgentState:
    """Mutable state shared across dispatch, gate, handlers, and progress."""

    running: bool = False
    task_type: str = ""  # "screen" | "code" | ""
    cancel_event: asyncio.Event = field(default_factory=asyncio.Event)
    hint_queue: asyncio.Queue = field(default_factory=asyncio.Queue)

    def reset(self):
        self.running = False
        self.task_type = ""
        self.cancel_event.clear()
        while not self.hint_queue.empty():
            try:
                self.hint_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
