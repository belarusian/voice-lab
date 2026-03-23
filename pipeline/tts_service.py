"""Remote Piper TTS service for Pipecat.

Sends text over WebSocket to tts_server.py on the CUDA box,
gets raw PCM audio back.
"""
import json
import re
from collections.abc import AsyncGenerator

import websockets
from pipecat.frames.frames import ErrorFrame, Frame, TTSAudioRawFrame
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import TTSService


class RemotePiperTTS(TTSService):
    def __init__(self, *, url: str, sample_rate: int = 22050, **kwargs):
        super().__init__(
            settings=TTSSettings(model=None, voice=None, language=None),
            sample_rate=sample_rate,
            **kwargs,
        )
        self._url = url

    @staticmethod
    def _strip_emoji(text: str) -> str:
        return re.sub(r"[^\w\s.,!?;:'\"-]", "", text, flags=re.UNICODE).strip()

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        text = self._strip_emoji(text)
        if not text:
            return
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
