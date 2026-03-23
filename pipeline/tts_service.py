"""Remote Piper TTS service for Pipecat.

Sends text over WebSocket to tts_server.py on the CUDA box,
gets raw PCM audio back.
"""
import json
from collections.abc import AsyncGenerator

import websockets
from pipecat.frames.frames import ErrorFrame, Frame, TTSAudioRawFrame
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import TTSService


class RemotePiperTTS(TTSService):
    def __init__(self, *, url: str, **kwargs):
        super().__init__(
            settings=TTSSettings(model=None, voice=None, language=None),
            **kwargs,
        )
        self._url = url

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        try:
            async with websockets.connect(self._url) as ws:
                await ws.send(json.dumps({"text": text}))
                pcm = await ws.recv()
                yield TTSAudioRawFrame(
                    audio=pcm,
                    sample_rate=self.sample_rate,
                    num_channels=1,
                    context_id=context_id,
                )
        except Exception as e:
            yield ErrorFrame(f"TTS error: {e}")
