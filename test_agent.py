#!/usr/bin/env python3
"""
Test agent for ferroscope run.
Makes 5 LLM calls:
  - calls 1-3: nearly identical prompts  → triggers retry_storm
  - call 4:    haiku → sonnet escalation  → triggers cost_inflation
  - call 5:    distinct prompt (baseline)
Reads ANTHROPIC_BASE_URL from env (injected by ferroscope run).
"""
import json
import os
import urllib.request

_BASE = os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
BASE_URL = f"{_BASE.rstrip('/')}/v1/messages"
API_KEY  = os.environ.get("ANTHROPIC_API_KEY", "")
VERSION  = "2023-06-01"


def call(model: str, prompt: str, label: str) -> str:
    payload = json.dumps({
        "model": model,
        "max_tokens": 64,
        "messages": [{"role": "user", "content": prompt}],
    }).encode()

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
        body = json.loads(resp.read())
    text = body["content"][0]["text"] if body.get("content") else "(empty)"
    usage = body.get("usage", {})
    print(f"[{label}] model={model} in={usage.get('input_tokens',0)} out={usage.get('output_tokens',0)}")
    print(f"         → {text[:80]}")
    return text


REPEATED = "Summarise the main risks of using LLMs in production in one sentence."

print(f"\nProxy: {BASE_URL}\n")

# calls 1-3: identical prompt — should trigger retry_storm
call("claude-haiku-4-5-20251001", REPEATED, "1/5 retry-storm-A")
call("claude-haiku-4-5-20251001", REPEATED, "2/5 retry-storm-B")
call("claude-haiku-4-5-20251001", REPEATED, "3/5 retry-storm-C")

# call 4: same prompt, upgraded model — should trigger cost_inflation
call("claude-sonnet-4-6", REPEATED, "4/5 cost-inflation")

# call 5: totally different prompt — baseline
call("claude-haiku-4-5-20251001", "What is 7 × 8? Reply with just the number.", "5/5 baseline")

print("\nAgent done.")
