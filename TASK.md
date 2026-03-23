# Task: Connect Voice Pipeline to Twilio Phone Number

## Overview

We have a working voice AI pipeline (STT -> LLM -> TTS) running on local
hardware. We need to connect it to a Twilio phone number so callers can
talk to the AI over the phone instead of a local microphone.

The public-facing address will be: **voice.compsci.boutique**

---

## What Already Exists

### Infrastructure (do NOT rebuild)

An EC2 proxy + WireGuard tunnel is already running. It currently serves
`chat.compsci.boutique` (a Matrix/Synapse instance). We reuse the same
EC2 instance for `voice.compsci.boutique`.

```
Public Internet
      |
      v
[Route53 DNS]
  chat.compsci.boutique  -> 54.243.75.156  (already live)
  voice.compsci.boutique -> 54.243.75.156  (you add this)
      |
      v
[EC2 t3.micro] -- 54.243.75.156
  - nginx reverse proxy + SSL (Let's Encrypt)
  - WireGuard tunnel endpoint (10.200.0.1)
      |
      | WireGuard tunnel (encrypted, UDP)
      |
      v
[Sunny -- 10.106.1.91] -- WireGuard IP: 10.200.0.2
  - faster-whisper STT server (port 9001)
  - Piper TTS server (port 9002)
  - Matrix/Synapse (port 8008) -- unrelated, leave it alone
  - Element Web (port 8080) -- unrelated, leave it alone
```

The LLM runs on a separate machine:
```
[10.106.1.184] -- llama.cpp with OpenAI-compat API (port 8080)
```

### Voice Pipeline (already working)

The voice-lab project in this directory has a working Pipecat pipeline:

```
mic -> VAD -> STT (faster-whisper) -> LLM (llama.cpp) -> TTS (Piper) -> speaker
```

Key files:
- `run.py` -- Pipecat pipeline (main entry point for local mic/speaker)
- `run_raw.py` -- raw async alternative (no framework)
- `config.yaml` -- endpoint URLs for STT, LLM, TTS
- `pipeline/stt_service.py` -- RemoteWhisperSTT WebSocket client
- `pipeline/tts_service.py` -- RemotePiperTTS WebSocket client
- `servers/stt_server.py` -- faster-whisper WebSocket server (runs on Sunny)
- `servers/tts_server.py` -- Piper TTS WebSocket server (runs on Sunny)
- `start-servers.ps1` -- launches STT/TTS servers on Sunny (PowerShell)

The pipeline currently uses `LocalAudioTransport` (microphone in, speaker out).
The task is to replace that with Twilio WebSocket transport so audio comes
from phone calls instead.

### Twilio Account

Active, upgraded (not trial). Credentials will be provided as environment
variables:

```bash
export TWILIO_ACCOUNT_SID="..."
export TWILIO_AUTH_TOKEN="..."
```

---

## Architecture: How Twilio Connects

```
Caller dials phone number
      |
      v
[Twilio]
  - Answers the call
  - Opens a WebSocket media stream to our server
  - Sends caller audio as mulaw/8kHz
  - Receives response audio back over the same WebSocket
      |
      v (WebSocket over HTTPS)
[voice.compsci.boutique]
      |
      v (nginx proxy through WireGuard)
[Sunny -- 10.200.0.2:PORT]
      |
      v
[Pipecat pipeline]
  - Receives Twilio audio stream
  - STT (faster-whisper on :9001)
  - LLM (llama.cpp on 10.106.1.184:8080)
  - TTS (Piper on :9002)
  - Sends response audio back through the WebSocket
  - Twilio plays it to the caller
```

Twilio Media Streams sends audio as mulaw 8kHz over WebSocket. The pipeline
needs to handle the format conversion (mulaw 8kHz <-> PCM 16kHz for Whisper,
PCM 22050Hz from Piper back to mulaw 8kHz for Twilio).

---

## Step-by-Step Implementation

### Phase 1: DNS + Nginx + SSL + Twilio -- ALREADY DONE

All of this has been completed. Do not redo it.

- Route53 A record: voice.compsci.boutique -> 54.243.75.156
- Nginx config: `/etc/nginx/sites-available/voice.compsci.boutique`
  - `/incoming` -> proxy to 10.200.0.2:8765 (Twilio webhook)
  - `/ws` -> proxy to 10.200.0.2:8765 (Twilio WebSocket media stream)
- SSL: Let's Encrypt cert, expires 2026-06-21, auto-renews
- Twilio phone number: (620) 202-7770 / +16202027770
  - SID: PN21bfec22d331c417c82c2029ac30ffbc
  - Voice webhook: https://voice.compsci.boutique/incoming (POST)
  - Calls to this number will POST to our server, which returns TwiML
    to open a WebSocket media stream to wss://voice.compsci.boutique/ws

Twilio can reach `https://voice.compsci.boutique/incoming` (webhook)
and `wss://voice.compsci.boutique/ws` (media stream). Both are live.
Calls currently 502 because the phone server (Phase 2) is not built yet.

---

### Phase 2: Phone Server on Sunny

This is the main new code. A server on Sunny (port 8765) that:
1. Handles the `/incoming` webhook (returns TwiML)
2. Handles the `/ws` WebSocket (bidirectional audio with Twilio)
3. Runs the voice pipeline on the audio stream

#### 2.1 Approach

Pipecat has built-in Twilio support. Check `pipecat.transports.network.twilio`
-- it provides a transport that replaces `LocalAudioTransport` and handles
the Twilio Media Stream protocol (mulaw encoding, WebSocket framing).

The pipeline structure stays the same:

```
[TwilioTransport input] -> STT -> LLM -> TTS -> [TwilioTransport output]
```

Instead of mic/speaker, audio flows through the Twilio WebSocket.

When Twilio POSTs to `/incoming`, the server returns TwiML that opens a
media stream:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Connect>
        <Stream url="wss://voice.compsci.boutique/ws" />
    </Connect>
</Response>
```

#### 2.2 Create the Phone Entry Point

Create a new file (e.g., `run_phone.py`) that:

1. Starts an HTTP/WebSocket server on port 8765
2. On POST `/incoming`: returns the TwiML connecting to `/ws`
3. On WebSocket `/ws`: creates a Pipecat pipeline with TwilioTransport
   using the same STT, LLM, TTS services from config.yaml

Use `run.py` as reference for the pipeline structure. The only change is
the transport layer.

#### 2.3 Audio Format Notes

- Twilio sends: mulaw, 8kHz, mono
- faster-whisper expects: PCM 16-bit, 16kHz, mono
- Piper outputs: PCM 16-bit, 22050Hz, mono
- Twilio expects back: mulaw, 8kHz, mono

If Pipecat's TwilioTransport handles the conversion automatically, great.
If not, you will need resampling (scipy.signal.resample or similar) and
mulaw encoding/decoding (audioop or the equivalent).

#### 2.4 System Prompt

Update the LLM system prompt for phone context. In `config.yaml`, add a
phone-specific prompt or create a separate config:

```yaml
llm:
  phone_system_prompt: >
    You are a voice assistant answering phone calls. Keep responses
    concise and natural for spoken conversation. If you do not know
    something, say so. If the caller needs a human, tell them you
    will connect them.
```

#### 2.5 Run as a Service

Create a systemd service on Sunny (WSL2) so it starts automatically:

```ini
[Unit]
Description=Voice AI Phone Server
After=network.target

[Service]
Type=simple
User=sasha
WorkingDirectory=/path/to/voice-lab
ExecStart=/path/to/voice-lab/.venv/bin/python run_phone.py
Restart=on-failure
RestartSec=5
Environment="TWILIO_ACCOUNT_SID=..."
Environment="TWILIO_AUTH_TOKEN=..."

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable voice-phone
sudo systemctl start voice-phone
```

---

### Phase 3: Test End-to-End

1. Verify STT/TTS servers are running on Sunny:
   ```bash
   curl http://10.106.1.91:9001/health
   curl http://10.106.1.91:9002/health
   ```

2. Verify LLM is running on .184:
   ```bash
   curl http://10.106.1.184:8080/health
   ```

3. Verify the phone server is running:
   ```bash
   curl https://voice.compsci.boutique/incoming
   ```
   Should return TwiML.

4. Call the Twilio phone number from a real phone. You should hear the AI
   respond to your speech.

5. Check logs if something fails:
   ```bash
   sudo journalctl -u voice-phone -f
   ```

---

## Access Summary

| Machine | How to Reach | What Runs There |
|---------|-------------|-----------------|
| EC2 proxy | `ssh -i ~/.ssh/cc-proxy.pem ubuntu@54.243.75.156` | nginx, WireGuard, SSL termination |
| Sunny | `ssh sasha@10.106.1.91` (from local network) or `10.200.0.2` (from EC2 via tunnel) | STT server, TTS server, phone server, Synapse |
| .184 | `10.106.1.184` (SSH has auth issues, may need fixing) | llama.cpp LLM inference |

## Existing Infrastructure Details

| Resource | Value |
|----------|-------|
| EC2 Instance | i-098f0e5284ff29151 |
| Elastic IP | 54.243.75.156 |
| Security Group | sg-003ad10e58a6cfb4e |
| Route53 Hosted Zone | Z08724971MVBZ0TJGIWI3 |
| WireGuard tunnel | EC2 10.200.0.1 <-> Sunny 10.200.0.2 |
| SSH key | ~/.ssh/cc-proxy.pem |
| SSL | Let's Encrypt (certbot auto-renews) |

## EC2 Security Group (already open)

Ports 80, 443, 8448, 3478, 5349, 51820 are already open. No changes needed
for this task -- Twilio connects over HTTPS (443) which is already allowed.

## What NOT to Touch

- Matrix/Synapse (port 8008) -- separate service, leave it alone
- Element Web (port 8080) -- separate service, leave it alone
- Coturn (ports 3478, 5349) -- TURN server for Matrix, leave it alone
- WireGuard config -- tunnel is working, do not modify
- Existing nginx configs for chat.compsci.boutique -- do not modify
