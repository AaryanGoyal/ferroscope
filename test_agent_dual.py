#!/usr/bin/env python3
"""
Dual-provider live test for ferroscope run.

Sends real calls through the proxy to both Anthropic and OpenAI
from the same process, verifying both routes are handled and logged.

Requires:
  ANTHROPIC_API_KEY — set in .env or environment
  OPENAI_API_KEY    — set in .env or environment
  ANTHROPIC_BASE_URL / OPENAI_BASE_URL — injected by ferroscope run

Run:
  ferroscope run -- python3 test_agent_dual.py
"""
import json
import os
import sys
import urllib.request
import urllib.error

# ferroscope run injects base URLs; we append the provider path.
ANTHROPIC_BASE = os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
OPENAI_BASE    = os.environ.get("OPENAI_BASE_URL",    "https://api.openai.com")
ANTHROPIC_URL  = f"{ANTHROPIC_BASE.rstrip('/')}/v1/messages"
OPENAI_URL     = f"{OPENAI_BASE.rstrip('/')}/v1/chat/completions"

ANTHROPIC_KEY  = os.environ.get("ANTHROPIC_API_KEY", "")
OPENAI_KEY     = os.environ.get("OPENAI_API_KEY", "")


def anthropic_call(prompt: str, model: str, label: str) -> str:
    payload = json.dumps({
        "model": model,
        "max_tokens": 64,
        "messages": [{"role": "user", "content": prompt}],
    }).encode()
    req = urllib.request.Request(
        ANTHROPIC_URL,
        data=payload,
        headers={
            "content-type": "application/json",
            "x-api-key": ANTHROPIC_KEY,
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req) as resp:
            body = json.loads(resp.read())
        text = body["content"][0]["text"] if body.get("content") else "(empty)"
        usage = body.get("usage", {})
        print(f"[{label}] anthropic/{model.split('-')[1]} "
              f"in={usage.get('input_tokens',0)} out={usage.get('output_tokens',0)}")
        print(f"  → {text[:90]}")
        return text
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        print(f"[{label}] ERROR {e.code}: {body[:120]}", file=sys.stderr)
        return ""


def openai_call(prompt: str, model: str, label: str) -> str:
    payload = json.dumps({
        "model": model,
        "max_tokens": 64,
        "messages": [{"role": "user", "content": prompt}],
    }).encode()
    req = urllib.request.Request(
        OPENAI_URL,
        data=payload,
        headers={
            "content-type": "application/json",
            "authorization": f"Bearer {OPENAI_KEY}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req) as resp:
            body = json.loads(resp.read())
        text = (body.get("choices") or [{}])[0].get("message", {}).get("content", "(empty)")
        usage = body.get("usage", {})
        print(f"[{label}] openai/{model} "
              f"in={usage.get('prompt_tokens',0)} out={usage.get('completion_tokens',0)}")
        print(f"  → {text[:90]}")
        return text
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        print(f"[{label}] ERROR {e.code}: {body[:120]}", file=sys.stderr)
        return ""


print(f"\nAnthropicProxy : {ANTHROPIC_URL}")
print(f"OpenAI Proxy   : {OPENAI_URL}\n")

print("═" * 60)
print("Anthropic calls")
print("═" * 60)
anthropic_call("What is 3 + 4? Reply with just the number.",
               "claude-haiku-4-5-20251001", "1/6 anthropic-buffered-1")
anthropic_call("What is 3 + 4? Reply with just the number.",
               "claude-haiku-4-5-20251001", "2/6 anthropic-buffered-2")
anthropic_call("Name three colours of the rainbow in a comma-separated list.",
               "claude-haiku-4-5-20251001", "3/6 anthropic-buffered-3")

print("\n" + "═" * 60)
print("OpenAI calls")
print("═" * 60)
openai_call("What is 3 + 4? Reply with just the number.",
            "gpt-4o-mini", "4/6 openai-buffered-1")
openai_call("What is 3 + 4? Reply with just the number.",
            "gpt-4o-mini", "5/6 openai-buffered-2")
openai_call("Name three colours of the rainbow in a comma-separated list.",
            "gpt-4o-mini", "6/6 openai-buffered-3")

print("\nDual-provider agent done.")
print("Expected: 6 calls total (3 Anthropic + 3 OpenAI) in the session summary.")
print("retry_storm should fire for both provider groups (3 identical prompts each).")
