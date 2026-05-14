# Ferroscope — Known Bugs

---

## Bug 1: OpenAI streaming token counts are always 0 ✅ FIXED

**File:** `src/proxy.rs` — `forward()`

OpenAI only includes token counts in the final SSE chunk when the request contains
`"stream_options": {"include_usage": true}`. Ferroscope forwarded the client's request
body verbatim without injecting this field. If the client omitted it (the default for
most SDKs and agents), `prompt_tokens` and `completion_tokens` were 0 in the DB for
every OpenAI streaming call — making cost tracking and classifier input entirely wrong.

**Fix applied:** `forward()` now injects `stream_options.include_usage = true` into the
upstream request body when `is_streaming && provider == OpenAI`. Existing `stream_options`
fields from the client are preserved via `entry().or_insert`. The body is re-serialised
only in that case; all other paths forward the original bytes unchanged.

---

## Bug 2: `latency_ms` measures total stream duration, not time-to-first-token ✅ FIXED

**File:** `src/proxy.rs` — `stream_response()`

`start.elapsed()` was sampled after `drop(tx)` — after the entire stream finished. For a
10-second streaming response the logged latency was ~10s, but what matters in practice is
TTFT (time to first token), which might be 200ms. The metric was misleading for every
streaming call in the TUI and DB.

**Fix applied:** `first_chunk_at: Option<Instant>` is recorded on the first `Ok(bytes)`
in the loop. `latency_ms` is now `first_chunk_at.duration_since(start)` — true TTFT.
Falls back to total elapsed only if no chunk ever arrived (error / empty response).

---

## Bug 3: Multi-byte characters split across chunk boundaries are silently dropped ✅ FIXED

**File:** `src/proxy.rs` — `stream_response()`

The old code called `std::str::from_utf8(&bytes)` on each raw chunk and silently skipped
the push if it returned an error. This happens whenever a multi-byte UTF-8 character
(CJK, emoji, etc.) is split across two consecutive network chunks. The bytes still
reached the client correctly, but `output_text` in the DB was corrupted and downstream
classifiers could read truncated text.

**Fix applied:** `StreamAccumulator` maintains a `byte_buf: Vec<u8>` that carries
incomplete UTF-8 byte sequences between `push()` calls. On each call it determines the
valid UTF-8 prefix length via `from_utf8`'s error `.valid_up_to()`, decodes exactly that
many bytes, and leaves the remainder in `byte_buf` for the next chunk. The character is
reconstructed correctly regardless of where the chunk boundary falls.

---

## Bug 4: Streaming task accumulates the full response body in memory ✅ FIXED

**File:** `src/proxy.rs` — `stream_response()`

Every byte forwarded to the client was also appended to `raw: String`, which grew without
bound until the stream finished. A 100K-token response is several MB. Under high
concurrency many simultaneous long streams silently multiplied this cost.

**Fix applied:** `raw: String` is replaced entirely by `StreamAccumulator`, which parses
SSE events line-by-line as chunks arrive and discards each line once processed. Only the
running state is kept: `line_buf` (one partial line), `byte_buf` (at most 3 bytes for a
pending multibyte sequence), `text` (output text), and two token counters. Peak memory
per stream is O(output text length), typically 3–5× smaller than the raw SSE body.

---

## Bug 5: Streaming tasks have no timeout — a stalled upstream leaks memory indefinitely

**File:** `src/proxy.rs` — `stream_response()`

The `tokio::spawn` task has no timeout. If the upstream stalls mid-stream (network
partition, model hang, slow client holding `tx` full), the task stays alive indefinitely,
holding all associated state in memory. Under adversarial or degraded conditions this is
an unbounded leak.

**Fix:** Wrap the streaming loop with `tokio::time::timeout`:

```rust
tokio::time::timeout(Duration::from_secs(120), async {
    while let Some(chunk) = byte_stream.next().await { ... }
}).await.unwrap_or_else(|_| tracing::warn!("stream timed out"));
```

The timeout value should be configurable via a CLI flag (`--stream-timeout-secs`).
On timeout, drop `tx` so the client gets a clean EOF rather than hanging.
