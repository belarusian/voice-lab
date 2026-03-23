"""Remote faster-whisper STT service for Pipecat.

Sends complete utterances over WebSocket to stt_server.py on the CUDA box,
gets transcription back.
"""
import json
from collections.abc import AsyncGenerator

import websockets
from pipecat.frames.frames import ErrorFrame, Frame, TranscriptionFrame
from pipecat.services.settings import STTSettings
from pipecat.services.stt_service import SegmentedSTTService
from pipecat.utils.time import time_now_iso8601


class RemoteWhisperSTT(SegmentedSTTService):
    def __init__(self, *, url: str, **kwargs):
        super().__init__(
            settings=STTSettings(model=None, language=None),
            **kwargs,
        )
        self._url = url

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        try:
            async with websockets.connect(self._url) as ws:
                await ws.send(audio)
                resp = json.loads(await ws.recv())
                text = resp.get("text", "").strip()
                if text:
                    yield TranscriptionFrame(text, self._user_id, time_now_iso8601())
        except Exception as e:
            yield ErrorFrame(f"STT error: {e}")
