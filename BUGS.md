# Ferroscope — Known Bugs

---

## Bug 1: OpenAI streaming token counts are always 0

**File:** `src/proxy.rs` — `forward()`

OpenAI only includes token counts in the final SSE chunk when the request contains
`"stream_options": {"include_usage": true}`. Ferroscope forwards the client's request
body verbatim without injecting this field. If the client omits it (which is the default
for most SDKs and agents), `prompt_tokens` and `completion_tokens` are 0 in the DB for
every OpenAI streaming call — making cost tracking and classifier input entirely wrong.

**Fix:** In `forward()`, when `is_streaming && provider == OpenAI`, deserialise the
request body, inject `stream_options.include_usage = true`, and re-serialise before
sending upstream. The client never sees the injected field; the extra final chunk is
transparent to them.

---

## Bug 2: `latency_ms` measures total stream duration, not time-to-first-token

**File:** `src/proxy.rs` — `stream_response()`

`start.elapsed()` is sampled inside the spawned task after `drop(tx)` — i.e., after
the entire stream has finished. For a 10-second streaming response the logged latency is
~10s, but what matters in practice is TTFT (time to first token), which might be 200ms.
The current metric makes the TUI latency column misleading for every streaming call.

**Fix:** Record a `first_chunk_at: Option<Instant>` inside the loop. Set it on the
first successful `Ok(bytes)` iteration. After the stream closes, compute
`ttft_ms = first_chunk_at.elapsed()` and store that as `latency_ms` for streaming
calls. Keep total stream duration as a separate optional field if useful.

---

## Bug 3: Multi-byte characters split across chunk boundaries are silently dropped from the DB

**File:** `src/proxy.rs` — `stream_response()`, line ~163

```rust
if let Ok(s) = std::str::from_utf8(&bytes) {
    raw.push_str(s);
}
```

`from_utf8` fails if a multi-byte UTF-8 character (CJK, emoji, etc.) is split across
two consecutive network chunks. The bytes still reach the client correctly via
`tx.send(Ok(bytes))`, but `raw` silently skips the fragment, corrupting `output_text`
reconstruction and any downstream classifier that reads it.

**Fix:** Replace `raw: String` with a `utf8_decoder: encoding_rs::Decoder` (or use
`std::str::Utf8Chunks` / a stateful `Utf8Decoder`) that carries leftover bytes between
iterations. Alternatively, accumulate into `raw_bytes: Vec<u8>` and do a single
`String::from_utf8_lossy(&raw_bytes)` after the loop — lossy conversion is acceptable
for observability purposes.

---

## Bug 4: Streaming task accumulates the full response body in memory

**File:** `src/proxy.rs` — `stream_response()`

Every byte forwarded to the client is also appended to `raw: String`, which grows
without bound until the stream finishes. A 100K-token response can be several MB. Under
high concurrency — many simultaneous long streams — this is a significant and invisible
memory sink. There is no cap or incremental eviction.

**Fix:** Parse SSE events incrementally as chunks arrive rather than accumulating the
full raw body. Maintain only the running state needed for reconstruction: a text buffer
for `output_text`, and integer accumulators for token counts. Discard each SSE event
once parsed. Peak memory per stream becomes O(response text length) rather than
O(raw SSE body length), which is typically 3–5× smaller.

---

## Bug 5: Streaming tasks have no timeout — a stalled upstream leaks memory indefinitely

**File:** `src/proxy.rs` — `stream_response()`

The `tokio::spawn` task has no timeout. If the upstream stalls mid-stream (network
partition, model hang, slow client holding `tx` full), the task stays alive indefinitely,
holding `raw` and all associated state in memory. Under adversarial or degraded
conditions this is an unbounded leak.

**Fix:** Wrap the streaming loop with `tokio::time::timeout`:

```rust
tokio::time::timeout(Duration::from_secs(120), async {
    while let Some(chunk) = byte_stream.next().await { ... }
}).await.unwrap_or_else(|_| tracing::warn!("stream timed out"));
```

The timeout value should be configurable via a CLI flag (`--stream-timeout-secs`).
On timeout, drop `tx` so the client gets a clean EOF rather than hanging.
