#!/usr/bin/env python3
"""STT server -- faster-whisper over WebSocket.

Run on a CUDA machine:
    pip install fastapi uvicorn[standard] faster-whisper
    python stt_server.py [--port 9001] [--model large-v3]

First run downloads the model from HuggingFace (~3 GB for large-v3).
"""
import argparse
import io
import json
import wave

import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket
from faster_whisper import WhisperModel

app = FastAPI()
model: WhisperModel | None = None


@app.websocket("/transcribe")
async def transcribe(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            audio_bytes = await ws.receive_bytes()
            # Pipecat's SegmentedSTTService sends WAV-wrapped audio
            if audio_bytes[:4] == b"RIFF":
                with wave.open(io.BytesIO(audio_bytes), "rb") as wf:
                    raw = wf.readframes(wf.getnframes())
                audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            else:
                audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            segments, _ = model.transcribe(audio, language="en", vad_filter=True)
            text = " ".join(seg.text for seg in segments).strip()
            await ws.send_text(json.dumps({"text": text}))
    except Exception:
        pass


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model is not None}


def main():
    global model
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=9001)
    parser.add_argument("--model", default="large-v3")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--compute-type", default="float16")
    args = parser.parse_args()

    print(f"loading whisper {args.model} on {args.device} ({args.compute_type})")
    model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type)
    print("model ready")

    uvicorn.run(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
