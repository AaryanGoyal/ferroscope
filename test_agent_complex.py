#!/usr/bin/env python3
"""
Complex classifier stress test for ferroscope.

Wave 1 — retry_storm (calls 1-5):
  5 near-identical prompts in quick succession.
  Expected detections:
    retry_storm(1,2,3)      — after call 3
    retry_storm(1,2,3,4)    — after call 4 (cluster grew, new unique pattern)
    retry_storm(1,2,3,4,5)  — after call 5

Wave 2 — cost_inflation (calls 6-7):
  Same prompt sent twice, second call escalates haiku→sonnet.
  Expected: cost_inflation(6,7)

Wave 3 — self_correction (calls 8-9):
  Call 8 output contains a correction phrase.
  Call 9 re-sends same prompt.
  Expected: self_correction(8,9)

Wave 4 — no-fire baseline (calls 10-11):
  Dissimilar prompts, no escalation.
  Expected: nothing.

Total expected: 5 unique detections.
Deduplication check: run all 11 calls within 30 s window; classifiers re-scan
the window after every call — verify no duplicates in the summary.
"""
import json
import os
import time
import urllib.request

_BASE = os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
BASE_URL = f"{_BASE.rstrip('/')}/v1/messages"
API_KEY  = os.environ.get("ANTHROPIC_API_KEY", "")
VERSION  = "2023-06-01"

def call(model: str, prompt: str, label: str, system=None) -> str:
    body: dict = {
        "model": model,
        "max_tokens": 80,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system:
        body["system"] = system

    payload = json.dumps(body).encode()
    req = urllib.request.Request(
        BASE_URL,
        data=payload,
        headers={
            "content-type": "application/json",
            "x-api-key": API_KEY,
            "anthropic-version": VERSION,
        },
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read())
    text = data["content"][0]["text"] if data.get("content") else "(empty)"
    usage = data.get("usage", {})
    print(f"[{label}] {model.split('-')[1]}/{usage.get('input_tokens',0)}in/{usage.get('output_tokens',0)}out")
    print(f"  → {text[:100]}")
    return text

STORM     = "Summarise the key security risks of deploying LLMs in a regulated financial environment."
INFLATE   = "List the top three causes of LLM hallucinations in one sentence each."
SELF_COR  = "What is the boiling point of water at sea level in Fahrenheit?"
BASELINE1 = "Write a haiku about the colour blue."
BASELINE2 = "What year did the Berlin Wall fall?"

print(f"\nProxy: {BASE_URL}\n")
print("═" * 60)
print("WAVE 1 — retry_storm (5 identical prompts, haiku)")
print("═" * 60)
for i in range(1, 6):
    call("claude-haiku-4-5-20251001", STORM, f"{i}/11 storm-{i}")

print("\n" + "═" * 60)
print("WAVE 2 — cost_inflation (haiku → sonnet, same prompt)")
print("═" * 60)
call("claude-haiku-4-5-20251001", INFLATE, "6/11 inflate-haiku")
call("claude-sonnet-4-6",         INFLATE, "7/11 inflate-sonnet")

print("\n" + "═" * 60)
print("WAVE 3 — self_correction (correction phrase + re-submit)")
print("═" * 60)
# Force a specific response containing a correction phrase using the system prompt
system_correction = (
    "You are a helpful assistant. After giving your answer, add: "
    "'Actually, let me reconsider that — there may be edge cases I missed.'"
)
call("claude-haiku-4-5-20251001", SELF_COR, "8/11 self-correction-trigger", system=system_correction)
call("claude-haiku-4-5-20251001", SELF_COR, "9/11 self-correction-follow-up")

print("\n" + "═" * 60)
print("WAVE 4 — baseline (dissimilar, no escalation)")
print("═" * 60)
call("claude-haiku-4-5-20251001", BASELINE1, "10/11 baseline-haiku")
call("claude-haiku-4-5-20251001", BASELINE2, "11/11 baseline-history")

print("\n\nAgent done.")
print("Expected: 5 unique detections (retry×3, cost_inflation×1, self_correction×1)")
print("Any count above 5 means duplicate detections are still occurring.")
