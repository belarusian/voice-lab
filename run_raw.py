#!/usr/bin/env python3
"""voice-lab -- full pipeline: mic -> VAD -> STT -> LLM -> TTS -> speaker.

No framework. Raw async Python + PyAudio + WebSockets + httpx.
"""
import asyncio
import json
import math
import struct
import sys

import httpx
import numpy as np
import pyaudio
import websockets
import yaml

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Audio capture / playback
# ---------------------------------------------------------------------------

RATE_IN = 16000
CHUNK = 480  # 30ms at 16kHz

# Energy VAD
SPEECH_THRESHOLD = 300
SPEECH_FRAMES = 5
SILENCE_FRAMES = 25


def rms(data: bytes) -> float:
    samples = struct.unpack(f"{len(data) // 2}h", data)
    return math.sqrt(sum(s * s for s in samples) / len(samples))


def play_audio(pa: pyaudio.PyAudio, pcm: bytes, rate: int):
    out = pa.open(format=pyaudio.paInt16, channels=1, rate=rate, output=True)
    out.write(pcm)
    out.stop_stream()
    out.close()


# ---------------------------------------------------------------------------
# STT: send audio to remote faster-whisper
# ---------------------------------------------------------------------------

async def transcribe(audio: bytes, url: str) -> str:
    async with websockets.connect(url) as ws:
        await ws.send(audio)
        resp = json.loads(await ws.recv())
        return resp.get("text", "").strip()


# ---------------------------------------------------------------------------
# LLM: stream chat from llama.cpp (OpenAI-compat)
# ---------------------------------------------------------------------------

async def chat_stream(text: str, history: list, cfg: dict):
    """Yields token strings as they arrive. Appends to history."""
    history.append({"role": "user", "content": text})
    messages = [{"role": "system", "content": cfg["system_prompt"]}] + history

    full = ""
    async with httpx.AsyncClient(timeout=30.0) as client:
        async with client.stream(
            "POST",
            f"{cfg['base_url']}/chat/completions",
            json={"model": cfg["model"], "messages": messages, "stream": True},
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                delta = json.loads(line[6:])["choices"][0]["delta"].get("content", "")
                if delta:
                    full += delta
                    yield delta

    history.append({"role": "assistant", "content": full})


# ---------------------------------------------------------------------------
# TTS: send text to remote piper, get PCM back
# ---------------------------------------------------------------------------

async def synthesize(text: str, url: str) -> bytes | None:
    try:
        async with websockets.connect(url) as ws:
            await ws.send(json.dumps({"text": text}))
            resp = await ws.recv()
            if isinstance(resp, str):
                err = json.loads(resp)
                print(f"  [tts error] {err.get('error', resp)}")
                return None
            return resp
    except Exception as e:
        print(f"  [tts error] {e}")
        return None


# ---------------------------------------------------------------------------
# Sentence splitter for streaming LLM -> TTS
# ---------------------------------------------------------------------------

import re

def split_sentences(text: str):
    """Returns (complete_sentences, remainder)."""
    parts = re.split(r"(?<=[.!?])\s+", text)
    if len(parts) > 1:
        return parts[:-1], parts[-1]
    return [], text


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

async def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    cfg = load_config(config_path)

    pa = pyaudio.PyAudio()
    mic = pa.open(
        format=pyaudio.paInt16, channels=1, rate=RATE_IN,
        input=True, frames_per_buffer=CHUNK,
    )

    history = []
    tts_rate = cfg["tts"]["sample_rate"]

    print("voice-lab running -- speak into your mic (ctrl-c to quit)")
    print(f"  STT: {cfg['stt']['url']}")
    print(f"  LLM: {cfg['llm']['base_url']} ({cfg['llm']['model']})")
    print(f"  TTS: {cfg['tts']['url']}")
    print()

    speech_buffer = []
    speech_count = 0
    silence_count = 0
    in_speech = False

    try:
        while True:
            data = mic.read(CHUNK, exception_on_overflow=False)
            level = rms(data)

            if level > SPEECH_THRESHOLD:
                speech_count += 1
                silence_count = 0
                if speech_count >= SPEECH_FRAMES and not in_speech:
                    in_speech = True
                if in_speech:
                    speech_buffer.append(data)
            else:
                silence_count += 1
                if in_speech:
                    if silence_count >= SILENCE_FRAMES:
                        in_speech = False
                        speech_count = 0
                        audio = b"".join(speech_buffer)
                        speech_buffer = []

                        # STT
                        text = await transcribe(audio, cfg["stt"]["url"])
                        if not text:
                            continue
                        print(f"  [you] {text}")

                        # LLM -> TTS (sentence-level streaming)
                        buffer = ""
                        full_response = ""
                        async for token in chat_stream(text, history, cfg["llm"]):
                            buffer += token
                            full_response += token
                            sentences, buffer = split_sentences(buffer)
                            for sentence in sentences:
                                pcm = await synthesize(sentence, cfg["tts"]["url"])
                                if pcm:
                                    play_audio(pa, pcm, tts_rate)

                        # Flush remainder
                        if buffer.strip():
                            pcm = await synthesize(buffer, cfg["tts"]["url"])
                            if pcm:
                                play_audio(pa, pcm, tts_rate)

                        print(f"  [ast] {full_response}")
                    else:
                        speech_buffer.append(data)
                else:
                    speech_count = 0

    except KeyboardInterrupt:
        print("\nshutting down")
    finally:
        mic.stop_stream()
        mic.close()
        pa.terminate()


if __name__ == "__main__":
    asyncio.run(main())
