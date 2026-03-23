"""Kokoro TTS wrapper with emoji stripping and speed control."""
import re
from collections.abc import AsyncGenerator

import numpy as np
from pipecat.frames.frames import ErrorFrame, Frame, TTSAudioRawFrame
from pipecat.services.kokoro.tts import KokoroTTSService

_EMOJI_RE = re.compile(r"[^\w\s.,!?;:'\"\-\(\)/&@#]", re.UNICODE)


class CleanKokoroTTS(KokoroTTSService):
    """KokoroTTSService with emoji stripping and configurable speed."""

    def __init__(self, *, speed: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self._speed = speed

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        text = _EMOJI_RE.sub("", text).strip()
        if not text:
            return
        try:
            await self.start_tts_usage_metrics(text)
            stream = self._kokoro.create_stream(
                text,
                voice=self._settings.voice,
                lang=self._settings.language,
                speed=self._speed,
            )
            async for samples, sample_rate in stream:
                await self.stop_ttfb_metrics()
                audio_int16 = (samples * 32767).astype(np.int16).tobytes()
                audio_data = await self._resampler.resample(
                    audio_int16, sample_rate, self.sample_rate
                )
                yield TTSAudioRawFrame(
                    audio=audio_data,
                    sample_rate=self.sample_rate,
                    num_channels=1,
                    context_id=context_id,
                )
        except Exception as e:
            yield ErrorFrame(error=f"TTS error: {e}")
        finally:
            await self.stop_ttfb_metrics()
