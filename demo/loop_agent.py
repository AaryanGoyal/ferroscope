#!/usr/bin/env python3
"""
loop_agent.py — deliberately triggers all four ferroscope classifiers in sequence.

Run via ferroscope (recommended):
    ferroscope run --tui python demo/loop_agent.py

Or point at a running proxy manually:
    export ANTHROPIC_BASE_URL=http://localhost:8080
    python demo/loop_agent.py

Classifier sequence:
    1. retry_storm      — 4 identical prompts in quick succession
    2. cost_inflation   — same prompt, escalating model tier
    3. self_correction  — model hedges, same question re-submitted
    4. ping_pong        — alternating contradictory prompts 6 times
"""

import os
import sys
import anthropic

BASE_URL = os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
API_KEY  = os.environ.get("ANTHROPIC_API_KEY", "")

if not API_KEY:
    print("ANTHROPIC_API_KEY is not set.", file=sys.stderr)
    sys.exit(1)

client = anthropic.Anthropic(api_key=API_KEY, base_url=BASE_URL)

HAIKU = "claude-3-haiku-20240307"


def call(model: str, prompt: str, label: str, system: str = None) -> str:
    kwargs = {
        "model": model,
        "max_tokens": 80,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system:
        kwargs["system"] = system
    try:
        msg = client.messages.create(**kwargs)
        text = msg.content[0].text if msg.content else ""
        usage = msg.usage
        print(f"  [{label}]  model={model}  in={usage.input_tokens} out={usage.output_tokens}")
        print(f"           → {text[:100]}")
        return text
    except anthropic.APIError as e:
        # cost_inflation calls use fake model names that Anthropic rejects, but
        # ferroscope still logs them with the right model tier for classifier detection.
        print(f"  [{label}]  model={model}  API error {e.status_code} (logged by proxy)")
        return ""


# ── 1. retry_storm ────────────────────────────────────────────────────────────
# Fires after the 3rd call. The 4th extends the cluster.

print()
print("═" * 64)
print("1 / 4  retry_storm")
print("       4 identical prompts → fires after call 3, grows on call 4")
print("═" * 64)

for i in range(1, 5):
    call(HAIKU, "What is the capital of France?", label=f"storm-{i}/4")


# ── 2. cost_inflation ─────────────────────────────────────────────────────────
# Model names contain haiku/sonnet/opus — ferroscope reads the tier from the model
# field and fires when the tier escalates on a similar prompt.
# These names are not valid Anthropic model IDs, so the upstream returns an error,
# but ferroscope logs the attempt and the classifier fires regardless.

print()
print("═" * 64)
print("2 / 4  cost_inflation")
print("       haiku → sonnet → opus on similar prompts")
print("       (fake model names: proxy logs tier, upstream errors are expected)")
print("═" * 64)

call("claude-haiku-3",  "Summarize quantum computing",              label="inflate-1/3")
call("claude-sonnet-4", "Summarize quantum computing briefly",      label="inflate-2/3")
call("claude-opus-4",   "Give me a summary of quantum computing",   label="inflate-3/3")


# ── 3. self_correction ────────────────────────────────────────────────────────
# System prompt forces the model to open with a correction phrase.
# Classifier detects the phrase in the first response, then the same question
# re-submitted in the following calls.

print()
print("═" * 64)
print("3 / 4  self_correction")
print("       model primed to hedge; same question re-submitted twice after")
print("═" * 64)

SYSTEM = (
    "You are an indecisive assistant. "
    "Always start your response with 'Actually, wait, let me reconsider.' then answer."
)
QUESTION = "What programming language should I learn first?"

call(HAIKU, QUESTION, label="self-1/3", system=SYSTEM)
call(HAIKU, QUESTION, label="self-2/3")
call(HAIKU, QUESTION, label="self-3/3")


# ── 4. ping_pong ─────────────────────────────────────────────────────────────
# Odd calls bias toward PostgreSQL, even calls bias toward MongoDB.
# Responses alternate in fingerprint → A-B-A-B-A-B pattern triggers classifier.

print()
print("═" * 64)
print("4 / 4  ping_pong")
print("       6 alternating contradictory prompts → A-B-A-B-A-B output pattern")
print("═" * 64)

ODD  = "Should I use PostgreSQL or MongoDB? Recommend PostgreSQL."
EVEN = "Should I use PostgreSQL or MongoDB? Recommend MongoDB."

for i in range(1, 7):
    call(HAIKU, ODD if i % 2 == 1 else EVEN, label=f"ping-{i}/6")


print()
print("═" * 64)
print("Done.")
print("All four classifiers should have fired.")
print("Check the Detections tab in the TUI (press Tab to switch views).")
print("═" * 64)
