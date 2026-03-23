#!/usr/bin/env python3
"""Standalone pipeline test -- bypasses Pipecat entirely.

Tests: mic -> energy VAD -> STT server -> print transcription
Run:   python test_pipeline.py
"""
import asyncio
import json
import math
import struct
import sys

import pyaudio
import websockets

STT_URL = "ws://10.106.1.91:9001/transcribe"
RATE = 16000
CHUNK = 480  # 30ms at 16kHz

# Energy VAD thresholds
SPEECH_THRESHOLD = 300   # RMS above this = speech
SILENCE_THRESHOLD = 150  # RMS below this = silence
SPEECH_FRAMES = 5        # consecutive speech frames to start
SILENCE_FRAMES = 25      # consecutive silence frames to stop (~750ms)


def rms(data: bytes) -> float:
    samples = struct.unpack(f"{len(data) // 2}h", data)
    return math.sqrt(sum(s * s for s in samples) / len(samples))


async def main():
    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK,
    )

    print(f"Listening on mic -- speak to test (ctrl-c to quit)")
    print(f"STT server: {STT_URL}")
    print()

    speech_buffer = []
    speech_count = 0
    silence_count = 0
    in_speech = False

    try:
        while True:
            data = stream.read(CHUNK, exception_on_overflow=False)
            level = rms(data)

            if level > SPEECH_THRESHOLD:
                speech_count += 1
                silence_count = 0
                if speech_count >= SPEECH_FRAMES and not in_speech:
                    in_speech = True
                    print("  [speech started]", flush=True)
                if in_speech:
                    speech_buffer.append(data)
            else:
                silence_count += 1
                if in_speech:
                    if silence_count >= SILENCE_FRAMES:
                        in_speech = False
                        speech_count = 0
                        print("  [speech ended, sending to STT...]", flush=True)

                        audio = b"".join(speech_buffer)
                        speech_buffer = []

                        try:
                            async with websockets.connect(STT_URL) as ws:
                                await ws.send(audio)
                                resp = json.loads(await ws.recv())
                                text = resp.get("text", "").strip()
                                if text:
                                    print(f"  >> {text}")
                                else:
                                    print("  >> (empty transcription)")
                        except Exception as e:
                            print(f"  >> STT error: {e}")
                    else:
                        speech_buffer.append(data)
                else:
                    speech_count = 0

    except KeyboardInterrupt:
        print("\ndone")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()


if __name__ == "__main__":
    asyncio.run(main())
