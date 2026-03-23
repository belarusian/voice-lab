#!/usr/bin/env python3
"""TTS server -- Piper over WebSocket.

Run on inference machine:
    pip install fastapi uvicorn[standard] piper-tts
    python tts_server.py [--port 9002] [--voice en_US-lessac-medium]

First run downloads the voice model from HuggingFace.
"""
import argparse
import io
import json
import wave

import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket
from piper import PiperVoice

app = FastAPI()
voice: PiperVoice | None = None
voice_sample_rate: int = 22050


@app.websocket("/synthesize")
async def synthesize(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            msg = json.loads(await ws.receive_text())
            text = msg.get("text", "")
            if not text:
                continue

            try:
                # Piper writes WAV to a file-like object
                buf = io.BytesIO()
                with wave.open(buf, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)  # 16-bit PCM
                    wf.setframerate(voice_sample_rate)
                    voice.synthesize(text, wf)

                # Extract raw PCM from WAV
                buf.seek(0)
                with wave.open(buf, "rb") as wf:
                    pcm = wf.readframes(wf.getnframes())

                await ws.send_bytes(pcm)
            except Exception as e:
                print(f"[TTS] synthesis error: {e}", flush=True)
                await ws.send_text(json.dumps({"error": str(e)}))
    except Exception:
        pass


@app.get("/health")
async def health():
    return {"status": "ok", "voice_loaded": voice is not None, "sample_rate": voice_sample_rate}


def main():
    global voice, voice_sample_rate
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=9002)
    parser.add_argument("--voice", default="en_US-lessac-medium")
    args = parser.parse_args()

    print(f"loading piper voice: {args.voice}")
    voice = PiperVoice.load(args.voice)
    voice_sample_rate = voice.config.sample_rate
    print(f"voice ready (sample_rate={voice_sample_rate})")

    uvicorn.run(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
