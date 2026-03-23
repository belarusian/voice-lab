# Voice AI -- Connecting Phone Calls to Our Pipeline

## The Goal

When someone calls (620) 202-7770, they talk to our voice AI instead of
hearing a demo message. The voice pipeline (STT, LLM, TTS) already works
with a local microphone. Your job is to make it work over a phone call
via Twilio.

## What Is Already Done (Do Not Redo)

- Voice pipeline running on Sunny (faster-whisper STT on :9001, Piper TTS
  on :9002, llama.cpp LLM on 10.106.1.184:8080)
- EC2 reverse proxy at 54.243.75.156 with WireGuard tunnel to Sunny
  (10.200.0.2) -- same one used for chat.compsci.boutique
- DNS: voice.compsci.boutique -> 54.243.75.156
- Nginx on EC2: `/incoming` and `/ws` both proxy to 10.200.0.2:8765
  (WebSocket-ready)
- SSL: Let's Encrypt cert, live and auto-renewing
- Twilio: phone number (620) 202-7770 configured to POST to
  https://voice.compsci.boutique/incoming on every inbound call

Right now, calls 502 because nothing is listening on Sunny port 8765.
That is what you build.

## What You Build

A server on Sunny (port 8765) that does two things:

1. Responds to POST `/incoming` with TwiML that tells Twilio to open a
   WebSocket media stream to `wss://voice.compsci.boutique/ws`
2. Handles that WebSocket -- receives caller audio, runs it through the
   existing STT -> LLM -> TTS pipeline, sends response audio back

Pipecat (which we already use) has a Twilio transport:
`pipecat.transports.network.twilio`. It should replace `LocalAudioTransport`
from the current `run.py`. The pipeline itself (STT, LLM, TTS wiring) stays
the same.

## Files to Read

All in this repo:

- `TASK.md` -- full task guide with architecture diagrams, audio format
  notes, systemd service template, and access details
- `run.py` -- current Pipecat pipeline (local mic/speaker). Your
  `run_phone.py` mirrors this but swaps the transport
- `config.yaml` -- STT/LLM/TTS endpoint URLs
- `pipeline/stt_service.py` -- RemoteWhisperSTT (WebSocket client)
- `pipeline/tts_service.py` -- RemotePiperTTS (WebSocket client)
- `servers/` -- the STT and TTS server code running on Sunny

## Access

- Sunny: `ssh sasha@10.106.1.91`
- EC2 (if needed, but you probably won't need it):
  `ssh -i ~/.ssh/cc-proxy.pem ubuntu@54.243.75.156`
- Twilio credentials: provided as env vars `TWILIO_ACCOUNT_SID` and
  `TWILIO_AUTH_TOKEN`

## Do Not Touch

- chat.compsci.boutique nginx config or any Matrix/Synapse/Element/Coturn
  services
- WireGuard tunnel config
- The EC2 instance beyond reading logs if needed

Start with `TASK.md`. Everything is in there.
