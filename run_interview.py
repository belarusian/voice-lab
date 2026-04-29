#!/usr/bin/env python3
"""voice-lab interview practice -- Pipecat pipeline: mic -> STT -> LLM -> TTS -> speaker.

The LLM acts as a GitHub recruiter coach, asking behavioral and technical questions,
then coaching you on how to answer them using your resume.

    python run_interview.py              # default config
    python run_interview.py config.yaml  # custom config

Usage:
    - Speak naturally; the coach will ask questions one at a time
    - Answer as if you're in a real recruiter screen
    - The coach will then teach you how to strengthen your answer using your resume
    - Practice your improved answer, then move to the next question
    - Say "quit" or "exit" to end the session
    - Say "feedback" to get a comprehensive performance summary

Coaching cycle per question:
    1. Coach asks a question
    2. You give your answer
    3. Coach coaches you on how to answer better using your resume
    4. You practice your improved answer
    5. Coach gives brief positive reinforcement, then moves to next question

The coach covers:
    1. Background and motivation
    2. Technical depth (distributed systems, platform engineering)
    3. Behavioral questions (leadership, conflict, failure)
    4. Role-specific (GitHub, Codespaces, Staff level)
    5. Your questions for them
"""
import asyncio
import sys
import signal
import yaml
from pathlib import Path
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams

from pipeline.kokoro_tts import CleanKokoroTTS
from pipeline.stt_service import RemoteWhisperSTT
from pipeline.call_logger import (
    CallTranscript,
    AssistantTranscriptLogger,
    CallerTranscriptLogger,
    save_and_push,
)


async def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Load system prompt from external file
    prompt_path = Path(__file__).parent / "interview_system_prompt.md"
    with open(prompt_path) as f:
        INTERVIEW_SYSTEM_PROMPT = f.read()

    # -- TTS --
    tts_cfg = cfg.get("tts", {})
    tts = CleanKokoroTTS(
        speed=tts_cfg.get("speed", 1.0),
        settings=CleanKokoroTTS.Settings(
            voice=tts_cfg.get("voice", "af_heart"),
        ),
    )

    # -- Transport --
    transport = LocalAudioTransport(
        params=LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_analyzer=SileroVADAnalyzer(
                params=VADParams(min_volume=0.15, confidence=0.6),
            ),
            audio_out_sample_rate=tts.sample_rate,
        ),
    )

    # -- STT --
    stt = RemoteWhisperSTT(url=cfg["stt"]["url"])

    # -- LLM --
    llm = OpenAILLMService(
        api_key="not-needed",
        base_url=cfg["llm"]["base_url"],
        model=cfg["llm"]["model"],
    )

    # -- Context --
    context = OpenAILLMContext(
        messages=[{"role": "system", "content": INTERVIEW_SYSTEM_PROMPT}]
    )
    context_aggregator = llm.create_context_aggregator(context)

    # -- Transcript capture --
    transcript = CallTranscript("interview", Path.home() / "voice-lab-interviews")
    caller_logger = CallerTranscriptLogger(transcript)
    assistant_logger = AssistantTranscriptLogger(transcript)

    # -- Pipeline --
    pipeline = Pipeline([
        transport.input(),
        stt,
        caller_logger,
        context_aggregator.user(),
        llm,
        assistant_logger,
        tts,
        transport.output(),
        context_aggregator.assistant(),
    ])

    task = PipelineTask(
        pipeline,
        params=PipelineParams(allow_interruptions=True),
    )

    runner = PipelineRunner()

    # Track session end to save transcript
    session_complete = asyncio.Event()

    @transport.event_handler("on_client_disconnected")
    async def on_disconnected(transport, websocket):
        print("[interview] session ended, saving transcript...")
        try:
            await save_and_push(transcript)
        except Exception as e:
            print(f"[interview] error saving transcript: {e}")
        finally:
            session_complete.set()

    def handle_signal(sig, frame):
        print("\n[interview] interrupt received, cleaning up...")
        try:
            # Flush any pending transcripts
            transcript._flush_sunny()
            # Save transcript directly (not via git)
            filepath = transcript.save()
            if filepath:
                print(f"[interview] saved transcript to {filepath}")
            else:
                print("[interview] no transcript to save (no entries captured)")
        except Exception as e:
            print(f"[interview] error saving transcript: {e}")
        finally:
            session_complete.set()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    try:
        await runner.run(task)
    finally:
        await session_complete.wait()


if __name__ == "__main__":
    asyncio.run(main())
