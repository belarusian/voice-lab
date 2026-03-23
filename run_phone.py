#!/usr/bin/env python3
"""voice-lab phone server -- Twilio calls -> Pipecat pipeline.

Listens on :8765. Nginx on EC2 proxies voice.compsci.boutique here
through the WireGuard tunnel.

POST /incoming  -> TwiML (tells Twilio to open media stream)
WS   /ws        -> Twilio media stream -> STT -> LLM -> TTS -> back to caller
"""
import asyncio
import json
import os
import sys

import uvicorn
import yaml
from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import PlainTextResponse

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.serializers.twilio import TwilioFrameSerializer
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)

from pipeline.stt_service import RemoteWhisperSTT
from pipeline.tts_service import RemotePiperTTS

app = FastAPI()

config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
with open(config_path) as f:
    cfg = yaml.safe_load(f)

TWILIO_ACCOUNT_SID = os.environ.get("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN", "")

TWIML = """<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Connect>
        <Stream url="wss://voice.compsci.boutique/ws" />
    </Connect>
</Response>"""


@app.post("/incoming")
async def incoming(request: Request):
    """Twilio webhook -- return TwiML to open a media stream."""
    return PlainTextResponse(TWIML, media_type="application/xml")


@app.websocket("/ws")
async def ws_twilio(websocket: WebSocket):
    """Handle Twilio media stream WebSocket."""
    await websocket.accept()

    # Wait for the Twilio "start" event to get stream_sid and call_sid.
    start_data = None
    async for message in websocket.iter_text():
        data = json.loads(message)
        if data.get("event") == "start":
            start_data = data.get("start", {})
            break

    if not start_data:
        await websocket.close()
        return

    stream_sid = start_data.get("streamSid", "")
    call_sid = start_data.get("callSid", "")

    print(f"[phone] call connected: call_sid={call_sid} stream_sid={stream_sid}")

    serializer = TwilioFrameSerializer(
        stream_sid=stream_sid,
        call_sid=call_sid,
        account_sid=TWILIO_ACCOUNT_SID,
        auth_token=TWILIO_AUTH_TOKEN,
        params=TwilioFrameSerializer.InputParams(
            auto_hang_up=bool(TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN),
        ),
    )

    transport = FastAPIWebsocketTransport(
        websocket=websocket,
        params=FastAPIWebsocketParams(
            serializer=serializer,
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_in_sample_rate=16000,
            audio_out_sample_rate=16000,
            vad_enabled=True,
            vad_analyzer=SileroVADAnalyzer(
                params=VADParams(min_volume=0.3, confidence=0.7),
            ),
        ),
    )

    stt = RemoteWhisperSTT(url=cfg["stt"]["url"])

    llm = OpenAILLMService(
        api_key="not-needed",
        base_url=cfg["llm"]["base_url"],
        model=cfg["llm"]["model"],
    )

    tts = RemotePiperTTS(
        url=cfg["tts"]["url"],
        sample_rate=cfg["tts"]["sample_rate"],
    )

    system_prompt = cfg["llm"].get("phone_system_prompt", cfg["llm"]["system_prompt"])
    context = OpenAILLMContext(
        messages=[{"role": "system", "content": system_prompt}]
    )
    context_aggregator = llm.create_context_aggregator(context)

    pipeline = Pipeline([
        transport.input(),
        stt,
        context_aggregator.user(),
        llm,
        tts,
        transport.output(),
        context_aggregator.assistant(),
    ])

    task = PipelineTask(
        pipeline,
        params=PipelineParams(allow_interruptions=True),
    )

    runner = PipelineRunner()

    @transport.event_handler("on_client_disconnected")
    async def on_disconnected(transport, websocket):
        print(f"[phone] call disconnected: call_sid={call_sid}")
        await task.cancel()

    await runner.run(task)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8765)
