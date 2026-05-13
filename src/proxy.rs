use std::sync::Arc;
use std::time::Instant;

use axum::body::Body;
use axum::extract::State;
use axum::http::{HeaderMap, Response, StatusCode};
use bytes::Bytes;
use futures::StreamExt;
use serde_json::Value;
use sha2::{Digest, Sha256};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

use crate::db::CallRecord;
use crate::AppState;

const UPSTREAM: &str = "https://api.anthropic.com/v1/messages";

// Hop-by-hop headers stripped from the client → upstream direction.
const STRIP_REQUEST_HEADERS: &[&str] =
    &["host", "content-length", "transfer-encoding", "connection"];

// Hop-by-hop headers stripped from the upstream → client direction.
// content-length must be removed because reqwest decompresses the body,
// making the upstream's content-length wrong; axum recomputes the correct one.
const STRIP_RESPONSE_HEADERS: &[&str] = &[
    "content-length",
    "transfer-encoding",
    "content-encoding",
    "connection",
    "keep-alive",
];

pub async fn handle_messages(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Bytes,
) -> Response<Body> {
    match forward(state, headers, body).await {
        Ok(r) => r,
        Err(e) => {
            tracing::error!("proxy error: {e:#}");
            Response::builder()
                .status(StatusCode::BAD_GATEWAY)
                .header("content-type", "application/json")
                .body(Body::from(format!("{{\"error\":\"{e}\"}}")))
                .unwrap()
        }
    }
}

async fn forward(
    state: Arc<AppState>,
    headers: HeaderMap,
    body: Bytes,
) -> anyhow::Result<Response<Body>> {
    let start = Instant::now();

    let body_json: Value = serde_json::from_slice(&body).unwrap_or(Value::Null);
    let model = body_json["model"]
        .as_str()
        .unwrap_or("unknown")
        .to_string();
    let is_streaming = body_json["stream"].as_bool().unwrap_or(false);
    let messages_str = body_json["messages"].to_string();
    let prompt_hash = sha256_hex(&messages_str);

    let loop_detected = {
        let mut det = state.loop_detector.lock().await;
        match det.check_and_record(&messages_str) {
            Some(w) => {
                tracing::warn!(
                    similar_count = w.similar_count,
                    max_similarity = format!("{:.2}", w.max_similarity),
                    "agentic loop detected"
                );
                true
            }
            None => false,
        }
    };

    let mut req = state.http_client.post(UPSTREAM);
    for (name, value) in &headers {
        if !STRIP_REQUEST_HEADERS.contains(&name.as_str()) {
            req = req.header(name, value);
        }
    }
    let upstream = req.body(body).send().await?;

    let status = upstream.status();
    // Collect upstream response headers before consuming the body.
    let resp_headers: Vec<(String, String)> = upstream
        .headers()
        .iter()
        .filter_map(|(k, v)| Some((k.to_string(), v.to_str().ok()?.to_string())))
        .collect();

    let input_text = format_messages(&body_json["messages"]);

    if is_streaming {
        stream_response(state, upstream, status, resp_headers, model, prompt_hash, loop_detected, input_text, start).await
    } else {
        buffered_response(state, upstream, status, resp_headers, model, prompt_hash, loop_detected, input_text, start).await
    }
}

async fn stream_response(
    state: Arc<AppState>,
    upstream: reqwest::Response,
    status: reqwest::StatusCode,
    resp_headers: Vec<(String, String)>,
    model: String,
    prompt_hash: String,
    loop_detected: bool,
    input_text: String,
    start: Instant,
) -> anyhow::Result<Response<Body>> {
    let (tx, rx) = mpsc::channel::<anyhow::Result<Bytes>>(64);
    let db = state.db.clone();

    tokio::spawn(async move {
        let mut byte_stream = upstream.bytes_stream();
        let mut raw = String::new();

        while let Some(chunk) = byte_stream.next().await {
            match chunk {
                Ok(bytes) => {
                    if let Ok(s) = std::str::from_utf8(&bytes) {
                        raw.push_str(s);
                    }
                    if tx.send(Ok(bytes)).await.is_err() {
                        return; // client disconnected
                    }
                }
                Err(e) => {
                    let _ = tx.send(Err(anyhow::anyhow!(e))).await;
                    return;
                }
            }
        }
        drop(tx);

        let (prompt_tokens, output_tokens, output_text) = parse_sse(&raw);
        let latency_ms = start.elapsed().as_millis() as i64;
        let record = CallRecord {
            timestamp: chrono::Utc::now().to_rfc3339(),
            cost_usd: compute_cost(&model, prompt_tokens, output_tokens),
            model,
            prompt_tokens,
            output_tokens,
            latency_ms,
            prompt_hash,
            loop_detected,
            input_text,
            output_text,
        };
        tracing::info!(
            model = record.model,
            prompt_tokens,
            output_tokens,
            latency_ms,
            cost_usd = record.cost_usd,
            "call logged (stream)"
        );
        if let Err(e) = db.insert_call(&record) {
            tracing::error!("db insert: {e}");
        }
    });

    let body = Body::from_stream(ReceiverStream::new(rx));
    build_response(status, resp_headers, body)
}

async fn buffered_response(
    state: Arc<AppState>,
    upstream: reqwest::Response,
    status: reqwest::StatusCode,
    resp_headers: Vec<(String, String)>,
    model: String,
    prompt_hash: String,
    loop_detected: bool,
    input_text: String,
    start: Instant,
) -> anyhow::Result<Response<Body>> {
    let bytes = upstream.bytes().await?;
    let latency_ms = start.elapsed().as_millis() as i64;

    let (prompt_tokens, output_tokens, output_text) = serde_json::from_slice::<Value>(&bytes)
        .map(|v| {
            let pt = v["usage"]["input_tokens"].as_i64().unwrap_or(0);
            let ot = v["usage"]["output_tokens"].as_i64().unwrap_or(0);
            let text = extract_response_text(&v);
            (pt, ot, text)
        })
        .unwrap_or_default();

    let record = CallRecord {
        timestamp: chrono::Utc::now().to_rfc3339(),
        cost_usd: compute_cost(&model, prompt_tokens, output_tokens),
        model,
        prompt_tokens,
        output_tokens,
        latency_ms,
        prompt_hash,
        loop_detected,
        input_text,
        output_text,
    };
    tracing::info!(
        model = record.model,
        prompt_tokens,
        output_tokens,
        latency_ms,
        cost_usd = record.cost_usd,
        "call logged"
    );
    if let Err(e) = state.db.insert_call(&record) {
        tracing::error!("db insert: {e}");
    }

    build_response(status, resp_headers, Body::from(bytes))
}

fn build_response(
    status: reqwest::StatusCode,
    headers: Vec<(String, String)>,
    body: Body,
) -> anyhow::Result<Response<Body>> {
    let axum_status = axum::http::StatusCode::from_u16(status.as_u16())?;
    let mut builder = Response::builder().status(axum_status);
    for (name, value) in &headers {
        if !STRIP_RESPONSE_HEADERS.contains(&name.as_str()) {
            builder = builder.header(name.as_str(), value.as_str());
        }
    }
    Ok(builder.body(body)?)
}

// ── helpers ──────────────────────────────────────────────────────────────────

fn sha256_hex(s: &str) -> String {
    hex::encode(Sha256::digest(s.as_bytes()))
}

/// Parse a raw Anthropic SSE body → (input_tokens, output_tokens, output_text).
fn parse_sse(raw: &str) -> (i64, i64, String) {
    let mut input = 0i64;
    let mut output = 0i64;
    let mut text = String::new();

    for line in raw.lines() {
        let Some(data) = line.strip_prefix("data: ") else {
            continue;
        };
        let Ok(v) = serde_json::from_str::<Value>(data) else {
            continue;
        };
        match v["type"].as_str() {
            Some("message_start") => {
                if let Some(t) = v["message"]["usage"]["input_tokens"].as_i64() {
                    input = t;
                }
            }
            Some("message_delta") => {
                if let Some(t) = v["usage"]["output_tokens"].as_i64() {
                    output = t;
                }
            }
            Some("content_block_delta") => {
                if v["delta"]["type"].as_str() == Some("text_delta") {
                    if let Some(t) = v["delta"]["text"].as_str() {
                        text.push_str(t);
                    }
                }
            }
            _ => {}
        }
    }
    (input, output, text)
}

/// Pull the text content out of a non-streaming Anthropic response.
fn extract_response_text(v: &Value) -> String {
    v["content"]
        .as_array()
        .map(|arr| {
            arr.iter()
                .filter_map(|block| {
                    if block["type"].as_str() == Some("text") {
                        block["text"].as_str().map(str::to_owned)
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>()
                .join("\n")
        })
        .unwrap_or_default()
}

/// Render the messages array as readable "[role]\ncontent" blocks.
fn format_messages(messages: &Value) -> String {
    let Some(arr) = messages.as_array() else {
        return String::new();
    };
    arr.iter()
        .map(|m| {
            let role = m["role"].as_str().unwrap_or("?");
            let content = match &m["content"] {
                Value::String(s) => s.clone(),
                Value::Array(parts) => parts
                    .iter()
                    .filter_map(|p| match p["type"].as_str() {
                        Some("text") => p["text"].as_str().map(str::to_owned),
                        Some("image") => Some("[image]".to_owned()),
                        Some("tool_use") => Some(format!(
                            "[tool_use: {}]",
                            p["name"].as_str().unwrap_or("?")
                        )),
                        Some("tool_result") => Some("[tool_result]".to_owned()),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
                    .join("\n"),
                _ => String::new(),
            };
            format!("[{role}]\n{content}")
        })
        .collect::<Vec<_>>()
        .join("\n\n")
}

/// Per-million-token pricing (input, output) in USD.
fn model_pricing(model: &str) -> (f64, f64) {
    if model.contains("claude-opus-4") || model.contains("claude-3-opus") {
        (15.0, 75.0)
    } else if model.contains("claude-sonnet-4") || model.contains("claude-3-5-sonnet") {
        (3.0, 15.0)
    } else if model.contains("claude-haiku-4")
        || model.contains("claude-3-5-haiku")
        || model.contains("claude-3-haiku")
    {
        (0.80, 4.0)
    } else if model.contains("gpt-4o-mini") {
        (0.15, 0.60)
    } else if model.contains("gpt-4o") {
        (2.5, 10.0)
    } else {
        (3.0, 15.0) // conservative fallback
    }
}

fn compute_cost(model: &str, input_tokens: i64, output_tokens: i64) -> f64 {
    let (ir, or_) = model_pricing(model);
    (input_tokens as f64 * ir + output_tokens as f64 * or_) / 1_000_000.0
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── sha256_hex ────────────────────────────────────────────────────────────

    #[test]
    fn sha256_known_vector() {
        // NIST known-answer: SHA-256("hello")
        assert_eq!(
            sha256_hex("hello"),
            "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
        );
    }

    #[test]
    fn sha256_empty_string() {
        assert_eq!(
            sha256_hex(""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn sha256_different_inputs_differ() {
        assert_ne!(sha256_hex("abc"), sha256_hex("abd"));
    }

    // ── model pricing / compute_cost ──────────────────────────────────────────

    #[test]
    fn cost_zero_tokens_is_zero() {
        assert_eq!(compute_cost("claude-haiku-4-5", 0, 0), 0.0);
    }

    #[test]
    fn cost_haiku_one_million_each() {
        // $0.80/M input + $4.00/M output = $4.80
        let c = compute_cost("claude-haiku-4-5", 1_000_000, 1_000_000);
        assert!((c - 4.80).abs() < 1e-6, "got {c}");
    }

    #[test]
    fn cost_sonnet_one_million_each() {
        // $3.00/M input + $15.00/M output = $18.00
        let c = compute_cost("claude-sonnet-4-6", 1_000_000, 1_000_000);
        assert!((c - 18.0).abs() < 1e-6, "got {c}");
    }

    #[test]
    fn cost_opus_one_million_each() {
        // $15.00/M input + $75.00/M output = $90.00
        let c = compute_cost("claude-opus-4-7", 1_000_000, 1_000_000);
        assert!((c - 90.0).abs() < 1e-6, "got {c}");
    }

    #[test]
    fn cost_gpt4o_one_million_each() {
        // $2.50/M input + $10.00/M output = $12.50
        let c = compute_cost("gpt-4o", 1_000_000, 1_000_000);
        assert!((c - 12.5).abs() < 1e-6, "got {c}");
    }

    #[test]
    fn cost_unknown_model_uses_positive_fallback() {
        let c = compute_cost("some-unknown-model-v99", 1_000_000, 0);
        assert!(c > 0.0);
    }

    // ── format_messages ───────────────────────────────────────────────────────

    #[test]
    fn format_messages_null_returns_empty() {
        assert_eq!(format_messages(&serde_json::json!(null)), "");
    }

    #[test]
    fn format_messages_empty_array() {
        assert_eq!(format_messages(&serde_json::json!([])), "");
    }

    #[test]
    fn format_messages_string_content() {
        let v = serde_json::json!([
            {"role": "user",      "content": "hello"},
            {"role": "assistant", "content": "hi there"}
        ]);
        let out = format_messages(&v);
        assert!(out.contains("[user]\nhello"), "got: {out}");
        assert!(out.contains("[assistant]\nhi there"), "got: {out}");
    }

    #[test]
    fn format_messages_array_content_text_and_image() {
        let v = serde_json::json!([{
            "role": "user",
            "content": [
                {"type": "text",  "text": "describe this"},
                {"type": "image", "source": {"type": "base64", "data": "..."}}
            ]
        }]);
        let out = format_messages(&v);
        assert!(out.contains("describe this"), "got: {out}");
        assert!(out.contains("[image]"), "got: {out}");
    }

    #[test]
    fn format_messages_tool_use_block() {
        let v = serde_json::json!([{
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "t1", "name": "web_search", "input": {}}
            ]
        }]);
        let out = format_messages(&v);
        assert!(out.contains("[tool_use: web_search]"), "got: {out}");
    }

    // ── extract_response_text ─────────────────────────────────────────────────

    #[test]
    fn extract_text_single_block() {
        let v = serde_json::json!({
            "content": [{"type": "text", "text": "hello world"}]
        });
        assert_eq!(extract_response_text(&v), "hello world");
    }

    #[test]
    fn extract_text_multiple_text_blocks_joined() {
        let v = serde_json::json!({
            "content": [
                {"type": "text", "text": "first"},
                {"type": "text", "text": "second"}
            ]
        });
        let out = extract_response_text(&v);
        assert!(out.contains("first") && out.contains("second"), "got: {out}");
    }

    #[test]
    fn extract_text_skips_non_text_blocks() {
        let v = serde_json::json!({
            "content": [
                {"type": "tool_use", "name": "fn", "id": "x", "input": {}},
                {"type": "text", "text": "after tool"}
            ]
        });
        let out = extract_response_text(&v);
        assert_eq!(out, "after tool");
    }

    #[test]
    fn extract_text_no_content_key() {
        let v = serde_json::json!({"type": "error", "error": {"message": "bad key"}});
        assert_eq!(extract_response_text(&v), "");
    }

    // ── parse_sse ─────────────────────────────────────────────────────────────

    #[test]
    fn parse_sse_empty_string() {
        let (inp, out, text) = parse_sse("");
        assert_eq!((inp, out, text.as_str()), (0, 0, ""));
    }

    #[test]
    fn parse_sse_full_stream() {
        let raw = concat!(
            "event: message_start\n",
            "data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":8}}}\n",
            "\n",
            "event: content_block_delta\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Hello\"}}\n",
            "\n",
            "event: content_block_delta\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\" world\"}}\n",
            "\n",
            "event: message_delta\n",
            "data: {\"type\":\"message_delta\",\"delta\":{},\"usage\":{\"output_tokens\":2}}\n",
            "\n",
            "event: message_stop\n",
            "data: {\"type\":\"message_stop\"}\n",
        );
        let (inp, out, text) = parse_sse(raw);
        assert_eq!(inp, 8);
        assert_eq!(out, 2);
        assert_eq!(text, "Hello world");
    }

    #[test]
    fn parse_sse_ignores_malformed_json_lines() {
        let raw = "data: not-json\ndata: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":3}}}\n";
        let (inp, _, _) = parse_sse(raw);
        assert_eq!(inp, 3);
    }

    #[test]
    fn parse_sse_accumulates_multiple_text_deltas() {
        let deltas = (0..5)
            .map(|i| {
                format!(
                    "data: {{\"type\":\"content_block_delta\",\"index\":0,\"delta\":{{\"type\":\"text_delta\",\"text\":\"{i}\"}}}}\n"
                )
            })
            .collect::<String>();
        let (_, _, text) = parse_sse(&deltas);
        assert_eq!(text, "01234");
    }
}
