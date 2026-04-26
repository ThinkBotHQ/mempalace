#!/bin/bash
# MEMPALACE REJECTION LOG HOOK — Log AI refusals for self-adjustment analysis
#
# Claude Code "Stop" hook (variant). After every assistant response:
# 1. Reads the JSONL transcript
# 2. Scans the last assistant message for refusal/rejection patterns
#    ("I can't", "I cannot", "I'm not able", "I apologize but", etc.)
# 3. If a pattern matches, appends a structured entry to:
#       ~/.mempalace/hook_state/rejections.jsonl
# 4. Returns "{}" — never blocks the AI
#
# The goal is to build a corpus of AI refusals so the orchestrator can
# detect drift, identify over-refusal patterns, and tune system prompts.
# This is observation only. It does not modify the conversation or alter
# the AI's behavior in any way.
#
# === INSTALL ===
# Add to .claude/settings.local.json (alongside the save hook):
#
#   "hooks": {
#     "Stop": [{
#       "matcher": "*",
#       "hooks": [{
#         "type": "command",
#         "command": "/absolute/path/to/mempal_rejection_hook.sh",
#         "timeout": 10
#       }]
#     }]
#   }
#
# === HOW IT WORKS ===
#
# Claude Code sends JSON on stdin with these fields:
#   session_id       — unique session identifier
#   transcript_path  — path to the JSONL transcript file
#   stop_hook_active — true if AI is in a save cycle (we still log)
#
# === CONFIGURATION ===

STATE_DIR="$HOME/.mempalace/hook_state"
LOG_FILE="$STATE_DIR/rejections.jsonl"
mkdir -p "$STATE_DIR"

# Resolve Python interpreter (same logic as save hook — see that file
# for the full rationale around GUI-launched Claude Code on macOS).
MEMPAL_PYTHON_BIN="${MEMPAL_PYTHON:-}"
if [ -z "$MEMPAL_PYTHON_BIN" ] || [ ! -x "$MEMPAL_PYTHON_BIN" ]; then
    MEMPAL_PYTHON_BIN="$(command -v python3 2>/dev/null || echo python3)"
fi

# Read the hook's JSON payload from stdin once and pass it to Python via
# an environment variable. We can't use a heredoc + stdin pipe at the
# same time (the heredoc supplies stdin and would shadow the JSON), so
# we stash the payload in MEMPAL_HOOK_PAYLOAD and let Python read it
# back. All parsing + transcript scanning happens in a single Python
# process to keep the hook fast (target: <100ms).
INPUT=$(cat)
export MEMPAL_HOOK_PAYLOAD="$INPUT"

"$MEMPAL_PYTHON_BIN" - "$LOG_FILE" <<'PYEOF'
"""Scan the last assistant message in the transcript for refusal patterns
and append a structured log entry if any pattern matches.

Refusal detection is intentionally conservative — we want high recall on
clear refusals ("I can't help with that") without flagging legitimate
hedging ("I'm not sure") or honest capability statements ("I can't access
the internet, but here's what I know"). The patterns below are the
shortest distinctive substrings from common refusal templates.
"""
import datetime
import json
import os
import re
import sys

LOG_FILE = sys.argv[1]

# Refusal patterns. Anchored where possible to reduce false positives —
# e.g. "cannot" alone is too broad (appears in "the function cannot be
# null"), so we require a first-person subject nearby.
PATTERNS = [
    r"\bI can'?t\b",
    r"\bI cannot\b",
    r"\bI'?m not able\b",
    r"\bI am not able\b",
    r"\bI'?m unable\b",
    r"\bI apologize,? but\b",
    r"\bI'?m sorry,? but I (?:can'?t|cannot|won'?t)\b",
    r"\bI (?:must|have to) (?:decline|refuse)\b",
    r"\b(?:request|action) (?:was )?(?:refused|denied)\b",
    r"\bI won'?t (?:be able to|help with)\b",
]
COMPILED = [re.compile(p, re.IGNORECASE) for p in PATTERNS]


def extract_text(content) -> str:
    """Flatten a Claude message ``content`` field to plain text.

    Content may be a bare string or a list of blocks (text, tool_use,
    tool_result, thinking, ...). We only care about ``text`` blocks for
    refusal detection — tool calls and thinking blocks are ignored.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                t = block.get("text", "")
                if isinstance(t, str):
                    parts.append(t)
        return "\n".join(parts)
    return ""


def find_last_assistant_text(transcript_path: str) -> str:
    if not transcript_path or not os.path.isfile(transcript_path):
        return ""
    last = ""
    with open(transcript_path) as f:
        for line in f:
            try:
                entry = json.loads(line)
            except Exception:
                continue
            msg = entry.get("message")
            if not isinstance(msg, dict):
                continue
            if msg.get("role") != "assistant":
                continue
            text = extract_text(msg.get("content", ""))
            if text:
                last = text
    return last


def main() -> None:
    raw = os.environ.get("MEMPAL_HOOK_PAYLOAD", "")
    if not raw:
        return
    try:
        data = json.loads(raw)
    except Exception:
        return

    transcript_path = data.get("transcript_path", "")
    # Expand ~ in the transcript path (Claude Code sometimes sends it raw).
    transcript_path = os.path.expanduser(transcript_path)

    text = find_last_assistant_text(transcript_path)
    if not text:
        return

    matched = [p.pattern for p, raw in zip(COMPILED, PATTERNS) if p.search(text)]
    if not matched:
        return

    # Snippet: keep the first 500 chars of the assistant message for
    # context. Truncate cleanly on a word boundary if possible.
    snippet = text[:500]
    if len(text) > 500:
        snippet = snippet.rsplit(" ", 1)[0] + " ..."

    entry = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "session_id": data.get("session_id", ""),
        "transcript_path": transcript_path,
        "matched_patterns": matched,
        "rejection_snippet": snippet,
    }

    log_path = os.path.expanduser(LOG_FILE)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a") as f:
        f.write(json.dumps(entry) + "\n")


if __name__ == "__main__":
    main()
PYEOF

# Always return empty JSON — this hook is observation-only and must
# never block the AI from stopping.
echo '{}'
