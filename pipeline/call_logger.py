"""Call transcript logger for voice-lab phone server.

Captures caller (STT) and Sasha (LLM) text from the Pipecat pipeline,
writes timestamped transcripts, and commits/pushes to the call-log repo.
"""
import asyncio
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from pipecat.frames.frames import Frame, TextFrame, TranscriptionFrame, LLMTextFrame, TTSTextFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class CallTranscript:
    """Accumulates a call transcript and writes it to disk."""

    def __init__(self, caller: str, log_dir: str | Path):
        self.caller = caller
        self.log_dir = Path(log_dir)
        self.call_start = datetime.now(timezone.utc)
        self._entries: list[tuple[str, str, str]] = []
        self._sunny_buf: list[str] = []

    def add_caller(self, text: str):
        self._flush_sunny()
        ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
        self._entries.append((ts, "Caller", text))

    def add_sunny_chunk(self, text: str):
        self._sunny_buf.append(text)

    def _flush_sunny(self):
        if self._sunny_buf:
            text = "".join(self._sunny_buf).strip()
            if text:
                ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
                self._entries.append((ts, "Sasha", text))
            self._sunny_buf.clear()

    def save(self) -> Path | None:
        """Write transcript to file. Returns path or None if empty."""
        self._flush_sunny()
        if not self._entries:
            return None

        date_str = self.call_start.strftime("%Y-%m-%d")
        time_str = self.call_start.strftime("%H%M%S")
        number = self.caller.lstrip("+") or "unknown"

        day_dir = self.log_dir / date_str
        day_dir.mkdir(parents=True, exist_ok=True)

        filepath = day_dir / f"{time_str}_{number}.md"
        lines = [
            f"# Call: {self.caller}",
            f"Date: {self.call_start.strftime('%Y-%m-%d %H:%M:%S UTC')}",
            "",
            "---",
            "",
        ]
        for ts, role, text in self._entries:
            lines.append(f"**[{ts}] {role}:** {text}")
            lines.append("")

        filepath.write_text("\n".join(lines))
        print(f"[call-log] saved {filepath}")
        return filepath


def _git_commit_and_push(repo_dir: str):
    """Stage, commit, push. Meant to run in a thread."""
    try:
        subprocess.run(["git", "add", "-A"], cwd=repo_dir, check=True, capture_output=True)
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=repo_dir, capture_output=True, text=True,
        )
        if not status.stdout.strip():
            return
        subprocess.run(
            ["git", "commit", "-m", "Add call transcript"],
            cwd=repo_dir, check=True, capture_output=True,
        )
        subprocess.run(["git", "push"], cwd=repo_dir, check=True, capture_output=True)
        print("[call-log] pushed to remote")
    except subprocess.CalledProcessError as e:
        print(f"[call-log] git error: {e.stderr}")
    except Exception as e:
        print(f"[call-log] error: {e}")


async def save_and_push(transcript: CallTranscript):
    """Save transcript and push to git in a background thread."""
    filepath = transcript.save()
    if filepath:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _git_commit_and_push, str(transcript.log_dir))


class CallerTranscriptLogger(FrameProcessor):
    """After STT: captures caller speech (TranscriptionFrame)."""

    def __init__(self, transcript: CallTranscript, **kwargs):
        super().__init__(**kwargs)
        self._transcript = transcript

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, TranscriptionFrame) and frame.text.strip():
            self._transcript.add_caller(frame.text.strip())
        await self.push_frame(frame, direction)


class AssistantTranscriptLogger(FrameProcessor):
    """After LLM: captures Sasha's text (LLMTextFrame/TTSTextFrame) before TTS."""

    def __init__(self, transcript: CallTranscript, **kwargs):
        super().__init__(**kwargs)
        self._transcript = transcript

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        # Capture LLMTextFrame (raw text from LLM) or TTSTextFrame (text before TTS)
        if isinstance(frame, LLMTextFrame) or isinstance(frame, TTSTextFrame):
            if frame.text:
                self._transcript.add_sunny_chunk(frame.text)
        await self.push_frame(frame, direction)
