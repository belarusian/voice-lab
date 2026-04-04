#!/usr/bin/env python3
"""Voice agent -- Pipecat pipeline with tool dispatch.

The voice pipeline: mic -> STT -> LLM -> TTS -> speaker.
Tool functions (screen, code) are supplied by the caller.
Standalone runs conversation-only (no tools).

    # conversation only
    python run_agent.py

    # with tools -- called from neo-lab or similar
    from run_agent import run_agent
    asyncio.run(run_agent(screen_fn=..., code_fn=...))
"""
import asyncio
import sys

import yaml
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
from pipeline.agent_state import AgentState
from pipeline.agent_gate import AgentGate
from pipeline.dispatch import build_tools, register_tools


async def run_agent(*, screen_fn=None, code_fn=None,
                    config_path="config.yaml"):
    """Start the voice agent pipeline.

    screen_fn(instruction: str) -> str
    code_fn(instruction: str, workspace: str | None) -> str

    If no functions supplied, runs as conversation-only.
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    agent_cfg = cfg.get("agent", {})

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

    # -- Context (tools only if functions supplied) --
    tools = build_tools(screen_fn=screen_fn, code_fn=code_fn)
    system_prompt = agent_cfg.get("system_prompt", cfg["llm"]["system_prompt"])

    context = OpenAILLMContext(
        messages=[{"role": "system", "content": system_prompt}],
        tools=tools or None,
    )
    context_aggregator = llm.create_context_aggregator(context)

    # -- State + tool registration --
    state = AgentState()
    register_tools(llm, state, screen_fn=screen_fn, code_fn=code_fn)

    # -- Pipeline --
    gate = AgentGate(state)

    pipeline = Pipeline([
        transport.input(),
        stt,
        gate,
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

    tool_names = [t["function"]["name"] for t in tools]
    print("voice agent ready")
    if tool_names:
        print(f"  tools: {', '.join(tool_names)}")
    else:
        print("  conversation only (no tools)")
    print()

    runner = PipelineRunner()
    await runner.run(task)


if __name__ == "__main__":
    asyncio.run(run_agent())
