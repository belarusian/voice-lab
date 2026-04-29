#!/usr/bin/env python3
"""Generate written feedback from an interview transcript.

Reads a transcript file and generates a summary of:
- Strengths
- Areas for improvement
- Key metrics/impact that should be emphasized
- Suggestions for next practice session

Optionally runs the feedback through your LLM for more nuanced insights.
"""
import sys
import asyncio
import re
from pathlib import Path

# Load private interview data from separate repo
_INTERVIEWS_PATH = Path.home() / "Ideas" / "interviews"
if _INTERVIEWS_PATH.exists():
    sys.path.insert(0, str(_INTERVIEWS_PATH))

from feedback_keywords import topic_keywords


def parse_transcript(filepath: str) -> list[tuple[str, str, str]]:
    """Parse a transcript markdown file into [(timestamp, role, text), ...]."""
    entries = []
    current_role = None
    current_text = []
    current_ts = None

    # Match lines like: **[00:33:46] Caller:** I love you.
    pattern = re.compile(r'^\*\*\[(\d{2}:\d{2}:\d{2})\] ([^:]+):\*\* (.*)$')

    with open(filepath) as f:
        for line in f:
            line = line.rstrip()
            match = pattern.match(line)
            if match:
                # Save previous entry
                if current_role and current_text:
                    entries.append((current_ts or "", current_role, " ".join(current_text)))
                    current_text = []

                # Parse new entry
                current_ts = match.group(1)
                current_role = match.group(2)
                current_text = [match.group(3)]
            elif current_role and line.strip():
                current_text.append(line.strip())

        # Don't forget last entry
        if current_role and current_text:
            entries.append((current_ts or "", current_role, " ".join(current_text)))

    return entries


def generate_feedback(entries: list[tuple[str, str, str]], filepath: str) -> str:
    """Generate minimal metadata for LLM review."""
    if not entries:
        return "No transcript found to analyze."

    caller_turns = [e for e in entries if e[1] == "Caller"]
    assistant_turns = [e for e in entries if e[1] == "Sasha"]

    lines = [
        f"Transcript metadata:",
        f"- Date: {Path(filepath).stem.split('_')[0]}",
        f"- Total turns: {len(entries)}",
        f"- Candidate answers: {len(caller_turns)}",
        f"- Coach responses: {len(assistant_turns)}",
    ]

    # Identify topics discussed (uses private topic_keywords)
    all_text = " ".join(e[2] for e in caller_turns).lower()
    topics = [t for t, kws in topic_keywords.items() if any(kw in all_text for kw in kws)]
    if topics:
        lines.append(f"- Topics discussed: {', '.join(topics)}")

    return "\n".join(lines)


async def llm_review(feedback: str, transcript: str, config: dict) -> str:
    """Send feedback and transcript to LLM for nuanced analysis."""
    stt_url = config.get("stt", {}).get("url", "ws://localhost:9001/transcribe")
    llm_url = config.get("llm", {}).get("base_url", "http://localhost:8080/v1")

    # Extract model from config
    model = config.get("llm", {}).get("model", "default")

    # Build prompt
    prompt = f"""You are a career coach reviewing an interview practice session.

Here's the raw transcript:
{transcript}

And here's an initial analysis:
{feedback}

Please provide deeper insights on:
1. What was said well - specific strengths to reinforce
2. What could be improved - concrete suggestions
3. Which resume bullets should be emphasized more
4. How well did the answers connect to the target role?
5. Any red flags or concerns the interviewer might have

Keep your response conversational and actionable.
"""

    try:
        # Use the same LLM as voice-lab
        import httpx
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{llm_url}/chat/completions",
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": "You are a career coach. Be specific and actionable in your feedback."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.7,
                    "max_tokens": 1000
                },
                timeout=60.0
            )
            result = response.json()
            llm_feedback = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            return f"\n## LLM-Enhanced Feedback\n\n{llm_feedback}\n"
    except Exception as e:
        return f"\n## LLM Feedback (failed: {e})\n\nCould not reach LLM. Here's the initial analysis instead:\n\n{feedback}"


async def main():
    if len(sys.argv) < 2:
        print("Usage: python feedback_from_transcript.py <transcript.md> [--llm]")
        print("  --llm: Run feedback through LLM for enhanced insights")
        sys.exit(1)

    filepath = sys.argv[1]
    use_llm = "--llm" in sys.argv

    if not Path(filepath).exists():
        print(f"Error: File not found: {filepath}")
        sys.exit(1)

    # Load config
    config_path = Path(__file__).parent / "config.yaml"
    config = {}
    if config_path.exists():
        import yaml
        with open(config_path) as f:
            config = yaml.safe_load(f)

    entries = parse_transcript(filepath)
    
    # Build transcript for LLM
    transcript_text = "\n".join(f"[{ts}] {role}: {text}" for ts, role, text in entries)
    
    feedback = generate_feedback(entries, filepath)
    
    if use_llm:
        print("[feedback] Sending to LLM for enhanced analysis...")
        llm_result = await llm_review(feedback, transcript_text, config)
        feedback = feedback + "\n" + llm_result
    
    print(feedback)


if __name__ == "__main__":
    asyncio.run(main())
