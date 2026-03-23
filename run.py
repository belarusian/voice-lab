#!/usr/bin/env python3
"""voice-lab -- Pipecat pipeline: mic -> STT -> LLM -> TTS -> speaker."""
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

from pipeline.stt_service import RemoteWhisperSTT
from pipeline.tts_service import RemotePiperTTS


async def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    transport = LocalAudioTransport(
        params=LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_analyzer=SileroVADAnalyzer(
                params=VADParams(min_volume=0.15, confidence=0.6),
            ),
            audio_out_sample_rate=cfg["tts"]["sample_rate"],
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

    context = OpenAILLMContext(
        messages=[{"role": "system", "content": cfg["llm"]["system_prompt"]}]
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
    await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
