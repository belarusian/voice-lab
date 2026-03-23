"""Kokoro TTS wrapper with speed control and audio buffering."""
import re
from collections.abc import AsyncGenerator

import numpy as np
from pipecat.frames.frames import ErrorFrame, Frame, TTSAudioRawFrame
from pipecat.services.kokoro.tts import KokoroTTSService

_EMOJI_RE = re.compile(
    "["
    "\U0001f600-\U0001f64f"
    "\U0001f300-\U0001f5ff"
    "\U0001f680-\U0001f6ff"
    "\U0001f1e0-\U0001f1ff"
    "\U0001f900-\U0001f9ff"
    "\U0001fa00-\U0001fa6f"
    "\U0001fa70-\U0001faff"
    "\U00002702-\U000027b0"
    "\U0000fe00-\U0000fe0f"
    "\U0000200d"
    "\U000020e3"
    "\U00002600-\U000026ff"
    "\U00002300-\U000023ff"
    "]+",
)


def _clean_for_speech(text: str) -> str:
    """Strip emoji and asterisks. Leave all real punctuation intact."""
    text = _EMOJI_RE.sub("", text)
    text = text.replace("*", "")
    return text


class CleanKokoroTTS(KokoroTTSService):
    """KokoroTTSService with speed control and optional buffering."""

    def __init__(self, *, speed: float = 1.0, buffer_secs: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self._speed = speed
        self._buffer_secs = buffer_secs

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        text = _clean_for_speech(text)
        if not text.strip():
            return
        try:
            await self.start_tts_usage_metrics(text)
            stream = self._kokoro.create_stream(
                text,
                voice=self._settings.voice,
                lang=self._settings.language,
                speed=self._speed,
            )

            min_chunk = int(self.sample_rate * self._buffer_secs) * 2
            buffer = bytearray()

            async for samples, sample_rate in stream:
                await self.stop_ttfb_metrics()
                audio_int16 = (samples * 32767).astype(np.int16).tobytes()
                audio_data = await self._resampler.resample(
                    audio_int16, sample_rate, self.sample_rate
                )

                if min_chunk <= 0:
                    yield TTSAudioRawFrame(
                        audio=audio_data,
                        sample_rate=self.sample_rate,
                        num_channels=1,
                        context_id=context_id,
                    )
                else:
                    buffer.extend(audio_data)
                    if len(buffer) >= min_chunk:
                        yield TTSAudioRawFrame(
                            audio=bytes(buffer),
                            sample_rate=self.sample_rate,
                            num_channels=1,
                            context_id=context_id,
                        )
                        buffer.clear()

            if buffer:
                yield TTSAudioRawFrame(
                    audio=bytes(buffer),
                    sample_rate=self.sample_rate,
                    num_channels=1,
                    context_id=context_id,
                )
        except Exception as e:
            yield ErrorFrame(error=f"TTS error: {e}")
        finally:
            await self.stop_ttfb_metrics()
