use std::sync::Arc;
use std::time::Instant;

use axum::body::Body;
use axum::extract::State;
use axum::http::{HeaderMap, Response, StatusCode};
use axum::routing::post;
use axum::Router;
use bytes::Bytes;
use futures::StreamExt;
use serde_json::Value;
use sha2::{Digest, Sha256};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

use crate::db::CallRecord;
use crate::{classifiers, AppState};

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

#[derive(Clone, Copy)]
enum Provider {
    Anthropic,
    OpenAI,
}

pub fn make_app(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/v1/messages", post(handle_messages))
        .route("/v1/chat/completions", post(handle_chat_completions))
        .with_state(state)
}

fn error_response(e: anyhow::Error) -> Response<Body> {
    tracing::error!("proxy error: {e:#}");
    Response::builder()
        .status(StatusCode::BAD_GATEWAY)
        .header("content-type", "application/json")
        .body(Body::from(format!("{{\"error\":\"{e}\"}}")))
        .unwrap()
}

pub async fn handle_messages(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Bytes,
) -> Response<Body> {
    match forward(state, headers, body, Provider::Anthropic).await {
        Ok(r) => r,
        Err(e) => error_response(e),
    }
}

pub async fn handle_chat_completions(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Bytes,
) -> Response<Body> {
    match forward(state, headers, body, Provider::OpenAI).await {
        Ok(r) => r,
        Err(e) => error_response(e),
    }
}

async fn forward(
    state: Arc<AppState>,
    headers: HeaderMap,
    body: Bytes,
    provider: Provider,
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

    let upstream_url = match provider {
        Provider::Anthropic => &state.anthropic_upstream,
        Provider::OpenAI => &state.openai_upstream,
    };

    let mut req = state.http_client.post(upstream_url);
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
        stream_response(state, upstream, status, resp_headers, model, prompt_hash, loop_detected, input_text, start, provider).await
    } else {
        buffered_response(state, upstream, status, resp_headers, model, prompt_hash, loop_detected, input_text, start, provider).await
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
    provider: Provider,
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

        let (prompt_tokens, output_tokens, output_text) = match provider {
            Provider::Anthropic => parse_anthropic_sse(&raw),
            Provider::OpenAI => parse_openai_sse(&raw),
        };
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
            classifier: None,
        };
        tracing::info!(
            model = record.model,
            prompt_tokens,
            output_tokens,
            latency_ms,
            cost_usd = record.cost_usd,
            "call logged (stream)"
        );
        match db.insert_call(&record) {
            Ok(_) => run_classifiers(&db),
            Err(e) => tracing::error!("db insert: {e}"),
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
    provider: Provider,
) -> anyhow::Result<Response<Body>> {
    let bytes = upstream.bytes().await?;
    let latency_ms = start.elapsed().as_millis() as i64;

    let (prompt_tokens, output_tokens, output_text) = serde_json::from_slice::<Value>(&bytes)
        .map(|v| match provider {
            Provider::Anthropic => {
                let pt = v["usage"]["input_tokens"].as_i64().unwrap_or(0);
                let ot = v["usage"]["output_tokens"].as_i64().unwrap_or(0);
                let text = extract_anthropic_response_text(&v);
                (pt, ot, text)
            }
            Provider::OpenAI => {
                let pt = v["usage"]["prompt_tokens"].as_i64().unwrap_or(0);
                let ot = v["usage"]["completion_tokens"].as_i64().unwrap_or(0);
                let text = extract_openai_response_text(&v);
                (pt, ot, text)
            }
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
        classifier: None,
    };
    tracing::info!(
        model = record.model,
        prompt_tokens,
        output_tokens,
        latency_ms,
        cost_usd = record.cost_usd,
        "call logged"
    );
    match state.db.insert_call(&record) {
        Ok(_) => run_classifiers(&state.db),
        Err(e) => tracing::error!("db insert: {e}"),
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
fn parse_anthropic_sse(raw: &str) -> (i64, i64, String) {
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

/// Parse a raw OpenAI SSE body → (prompt_tokens, completion_tokens, output_text).
fn parse_openai_sse(raw: &str) -> (i64, i64, String) {
    let mut prompt_tokens = 0i64;
    let mut completion_tokens = 0i64;
    let mut text = String::new();

    for line in raw.lines() {
        let Some(data) = line.strip_prefix("data: ") else {
            continue;
        };
        if data == "[DONE]" {
            break;
        }
        let Ok(v) = serde_json::from_str::<Value>(data) else {
            continue;
        };
        // Accumulate text from choices[].delta.content
        if let Some(choices) = v["choices"].as_array() {
            for choice in choices {
                if let Some(content) = choice["delta"]["content"].as_str() {
                    text.push_str(content);
                }
            }
        }
        // When a line has usage field, grab token counts
        if let Some(usage) = v.get("usage") {
            if !usage.is_null() {
                if let Some(pt) = usage["prompt_tokens"].as_i64() {
                    prompt_tokens = pt;
                }
                if let Some(ct) = usage["completion_tokens"].as_i64() {
                    completion_tokens = ct;
                }
            }
        }
    }
    (prompt_tokens, completion_tokens, text)
}

/// Pull the text content out of a non-streaming Anthropic response.
fn extract_anthropic_response_text(v: &Value) -> String {
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

/// Pull the text content out of a non-streaming OpenAI response.
fn extract_openai_response_text(v: &Value) -> String {
    v["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or("")
        .to_string()
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
    } else if model.contains("gpt-4.1") {
        (2.0, 8.0)
    } else if model.contains("gpt-4-turbo") {
        (10.0, 30.0)
    } else {
        (3.0, 15.0) // conservative fallback
    }
}

fn compute_cost(model: &str, input_tokens: i64, output_tokens: i64) -> f64 {
    let (ir, or_) = model_pricing(model);
    (input_tokens as f64 * ir + output_tokens as f64 * or_) / 1_000_000.0
}

fn run_classifiers(db: &crate::db::Database) {
    match classifiers::run_all(db) {
        Ok(result) => {
            for d in &result.detections {
                // Skip exact duplicates: same classifier + same call_ids already recorded.
                match db.detection_exists(&d.classifier, &d.call_ids) {
                    Ok(true) => continue,
                    Err(e) => { tracing::error!("detection_exists: {e}"); continue; }
                    Ok(false) => {}
                }

                tracing::warn!(classifier = %d.classifier, detail = %d.detail, "classifier fired");
                if let Err(e) = db.insert_detection(d) {
                    tracing::error!("detection insert: {e}");
                }
                // Tag each involved call with this classifier (last writer wins for display).
                for id_str in d.call_ids.split(',') {
                    if let Ok(id) = id_str.parse::<i64>() {
                        if let Err(e) = db.update_call_classifier(id, &d.classifier) {
                            tracing::error!("call classifier update: {e}");
                        }
                    }
                }
            }
        }
        Err(e) => tracing::error!("classifiers error: {e}"),
    }
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
    fn cost_gpt4_1_one_million_each() {
        // $2.00/M input + $8.00/M output = $10.00
        let c = compute_cost("gpt-4.1", 1_000_000, 1_000_000);
        assert!((c - 10.0).abs() < 1e-6, "got {c}");
    }

    #[test]
    fn cost_gpt4_turbo_one_million_each() {
        // $10.00/M input + $30.00/M output = $40.00
        let c = compute_cost("gpt-4-turbo", 1_000_000, 1_000_000);
        assert!((c - 40.0).abs() < 1e-6, "got {c}");
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

    // ── extract_anthropic_response_text ───────────────────────────────────────

    #[test]
    fn extract_text_single_block() {
        let v = serde_json::json!({
            "content": [{"type": "text", "text": "hello world"}]
        });
        assert_eq!(extract_anthropic_response_text(&v), "hello world");
    }

    #[test]
    fn extract_text_multiple_text_blocks_joined() {
        let v = serde_json::json!({
            "content": [
                {"type": "text", "text": "first"},
                {"type": "text", "text": "second"}
            ]
        });
        let out = extract_anthropic_response_text(&v);
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
        let out = extract_anthropic_response_text(&v);
        assert_eq!(out, "after tool");
    }

    #[test]
    fn extract_text_no_content_key() {
        let v = serde_json::json!({"type": "error", "error": {"message": "bad key"}});
        assert_eq!(extract_anthropic_response_text(&v), "");
    }

    // ── extract_openai_response_text ──────────────────────────────────────────

    #[test]
    fn extract_openai_response_text_normal() {
        let v = serde_json::json!({
            "choices": [{"message": {"role": "assistant", "content": "Hello!"}}]
        });
        assert_eq!(extract_openai_response_text(&v), "Hello!");
    }

    #[test]
    fn extract_openai_response_text_empty_choices() {
        let v = serde_json::json!({"choices": []});
        assert_eq!(extract_openai_response_text(&v), "");
    }

    // ── parse_anthropic_sse ───────────────────────────────────────────────────

    #[test]
    fn parse_anthropic_sse_empty_string() {
        let (inp, out, text) = parse_anthropic_sse("");
        assert_eq!((inp, out, text.as_str()), (0, 0, ""));
    }

    #[test]
    fn parse_anthropic_sse_full_stream() {
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
        let (inp, out, text) = parse_anthropic_sse(raw);
        assert_eq!(inp, 8);
        assert_eq!(out, 2);
        assert_eq!(text, "Hello world");
    }

    #[test]
    fn parse_anthropic_sse_ignores_malformed_json_lines() {
        let raw = "data: not-json\ndata: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":3}}}\n";
        let (inp, _, _) = parse_anthropic_sse(raw);
        assert_eq!(inp, 3);
    }

    #[test]
    fn parse_anthropic_sse_accumulates_multiple_text_deltas() {
        let deltas = (0..5)
            .map(|i| {
                format!(
                    "data: {{\"type\":\"content_block_delta\",\"index\":0,\"delta\":{{\"type\":\"text_delta\",\"text\":\"{i}\"}}}}\n"
                )
            })
            .collect::<String>();
        let (_, _, text) = parse_anthropic_sse(&deltas);
        assert_eq!(text, "01234");
    }

    // ── parse_openai_sse ──────────────────────────────────────────────────────

    #[test]
    fn parse_openai_sse_empty_string() {
        let (pt, ct, text) = parse_openai_sse("");
        assert_eq!((pt, ct, text.as_str()), (0, 0, ""));
    }

    #[test]
    fn parse_openai_sse_full_stream_with_usage() {
        let raw = concat!(
            "data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"\"},\"finish_reason\":null}]}\n",
            "\n",
            "data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hi\"},\"finish_reason\":null}]}\n",
            "\n",
            "data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\" there\"},\"finish_reason\":null}],\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":2,\"total_tokens\":7}}\n",
            "\n",
            "data: [DONE]\n",
        );
        let (pt, ct, text) = parse_openai_sse(raw);
        assert_eq!(pt, 5);
        assert_eq!(ct, 2);
        assert_eq!(text, "Hi there");
    }

    #[test]
    fn parse_openai_sse_accumulates_deltas() {
        let deltas = (0..5)
            .map(|i| {
                format!(
                    "data: {{\"id\":\"c\",\"choices\":[{{\"index\":0,\"delta\":{{\"content\":\"{i}\"}}}}]}}\n"
                )
            })
            .collect::<String>();
        let (_, _, text) = parse_openai_sse(&deltas);
        assert_eq!(text, "01234");
    }
}

// ── Integration tests (proxy handler + mock upstream) ─────────────────────────
//
// These tests exercise the full request→forward→response cycle using wiremock
// to stand in for the Anthropic API. No real network calls are made.
//
// Live tests (requires ANTHROPIC_API_KEY) are marked #[ignore] and can be run
// with:  cargo test live -- --ignored
#[cfg(test)]
mod integration {
    use std::sync::Arc;

    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use http_body_util::BodyExt;
    use tokio::sync::Mutex;
    use tower::ServiceExt;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    use crate::db::Database;
    use crate::loop_detector::LoopDetector;
    use crate::AppState;
    use super::make_app;

    // ── helpers ───────────────────────────────────────────────────────────────

    fn make_state(anthropic_upstream_url: String) -> Arc<AppState> {
        Arc::new(AppState {
            db: Database::new(":memory:").unwrap(),
            loop_detector: Mutex::new(LoopDetector::new()),
            http_client: reqwest::Client::new(),
            anthropic_upstream: anthropic_upstream_url,
            openai_upstream: "https://api.openai.com/v1/chat/completions".to_string(),
        })
    }

    fn make_state_with_openai(anthropic_upstream_url: String, openai_upstream_url: String) -> Arc<AppState> {
        Arc::new(AppState {
            db: Database::new(":memory:").unwrap(),
            loop_detector: Mutex::new(LoopDetector::new()),
            http_client: reqwest::Client::new(),
            anthropic_upstream: anthropic_upstream_url,
            openai_upstream: openai_upstream_url,
        })
    }

    async fn collect_body(body: Body) -> bytes::Bytes {
        body.collect().await.unwrap().to_bytes()
    }

    const BUFFERED_REQ: &str = r#"{
        "model": "claude-test",
        "max_tokens": 10,
        "messages": [{"role": "user", "content": "hi"}]
    }"#;

    const STREAM_REQ: &str = r#"{
        "model": "claude-test",
        "max_tokens": 10,
        "stream": true,
        "messages": [{"role": "user", "content": "hi"}]
    }"#;

    const OAI_BUFFERED_REQ: &str = r#"{
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "hi"}]
    }"#;

    const OAI_STREAM_REQ: &str = r#"{
        "model": "gpt-4o",
        "stream": true,
        "messages": [{"role": "user", "content": "hi"}]
    }"#;

    // ── buffered path ─────────────────────────────────────────────────────────

    #[tokio::test]
    async fn buffered_complete_response_returned_to_client() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "application/json")
                    .set_body_json(serde_json::json!({
                        "id": "msg_01",
                        "type": "message",
                        "role": "assistant",
                        "model": "claude-test",
                        "content": [{"type": "text", "text": "Hello!"}],
                        "stop_reason": "end_turn",
                        "usage": {"input_tokens": 5, "output_tokens": 3}
                    })),
            )
            .mount(&server)
            .await;

        let state = make_state(format!("{}/v1/messages", server.uri()));
        let resp = make_app(state.clone())
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/messages")
                    .header("content-type", "application/json")
                    .body(Body::from(BUFFERED_REQ))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);

        let bytes = collect_body(resp.into_body()).await;
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();

        assert_eq!(json["content"][0]["text"], "Hello!", "response text truncated");
        assert_eq!(json["usage"]["input_tokens"], 5);
        assert_eq!(json["usage"]["output_tokens"], 3);

        // DB row logged with correct token counts and extracted output text.
        let rows = state.db.query_recent(1).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].prompt_tokens, 5);
        assert_eq!(rows[0].output_tokens, 3);
        assert_eq!(rows[0].output_text, "Hello!");
    }

    #[tokio::test]
    async fn buffered_upstream_status_forwarded_on_error() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .respond_with(
                ResponseTemplate::new(401)
                    .insert_header("content-type", "application/json")
                    .set_body_json(serde_json::json!({
                        "type": "error",
                        "error": {"type": "authentication_error", "message": "bad key"}
                    })),
            )
            .mount(&server)
            .await;

        let state = make_state(format!("{}/v1/messages", server.uri()));
        let resp = make_app(state)
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/messages")
                    .header("content-type", "application/json")
                    .body(Body::from(BUFFERED_REQ))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);

        let bytes = collect_body(resp.into_body()).await;
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(json["error"]["type"], "authentication_error");
    }

    // ── streaming path ────────────────────────────────────────────────────────

    const SSE_BODY: &str = concat!(
        "event: message_start\n",
        "data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":5}}}\n\n",
        "event: content_block_start\n",
        "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
        "event: content_block_delta\n",
        "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Hi\"}}\n\n",
        "event: content_block_delta\n",
        "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\" there\"}}\n\n",
        "event: message_delta\n",
        "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":2}}\n\n",
        "event: message_stop\n",
        "data: {\"type\":\"message_stop\"}\n\n",
    );

    #[tokio::test]
    async fn streaming_full_sse_stream_forwarded_to_client() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(SSE_BODY),
            )
            .mount(&server)
            .await;

        let state = make_state(format!("{}/v1/messages", server.uri()));
        let resp = make_app(state.clone())
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/messages")
                    .header("content-type", "application/json")
                    .body(Body::from(STREAM_REQ))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);

        let bytes = collect_body(resp.into_body()).await;
        let body_str = std::str::from_utf8(&bytes).unwrap();

        // Every SSE event must reach the client intact.
        assert!(body_str.contains("message_start"),  "missing message_start");
        assert!(body_str.contains("content_block_delta"), "missing delta events");
        assert!(body_str.contains("message_stop"),   "missing message_stop");
        // Text chunks both arrive.
        assert!(body_str.contains("\"Hi\""),    "missing first text delta");
        assert!(body_str.contains("\" there\""), "missing second text delta");

        // The spawned logger needs a moment to write after the stream closes.
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;

        let rows = state.db.query_recent(1).unwrap();
        assert_eq!(rows.len(), 1, "streaming call must be logged");
        assert_eq!(rows[0].prompt_tokens, 5);
        assert_eq!(rows[0].output_tokens, 2);
        assert_eq!(rows[0].output_text, "Hi there", "output text must be reconstructed from deltas");
    }

    #[tokio::test]
    async fn streaming_upstream_status_forwarded_on_error() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .respond_with(
                ResponseTemplate::new(429)
                    .insert_header("content-type", "application/json")
                    .set_body_json(serde_json::json!({
                        "type": "error",
                        "error": {"type": "rate_limit_error", "message": "too many requests"}
                    })),
            )
            .mount(&server)
            .await;

        let state = make_state(format!("{}/v1/messages", server.uri()));
        let resp = make_app(state)
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/messages")
                    .header("content-type", "application/json")
                    .body(Body::from(STREAM_REQ))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::TOO_MANY_REQUESTS);
    }

    // ── OpenAI buffered path ──────────────────────────────────────────────────

    #[tokio::test]
    async fn openai_buffered_complete_response_returned_to_client() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "application/json")
                    .set_body_json(serde_json::json!({
                        "id": "chatcmpl-1",
                        "object": "chat.completion",
                        "choices": [{
                            "index": 0,
                            "message": {"role": "assistant", "content": "Hello!"},
                            "finish_reason": "stop"
                        }],
                        "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}
                    })),
            )
            .mount(&server)
            .await;

        let state = make_state_with_openai(
            "https://api.anthropic.com/v1/messages".to_string(),
            format!("{}/v1/chat/completions", server.uri()),
        );
        let resp = make_app(state.clone())
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(OAI_BUFFERED_REQ))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);

        let bytes = collect_body(resp.into_body()).await;
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();

        assert_eq!(json["choices"][0]["message"]["content"], "Hello!");
        assert_eq!(json["usage"]["prompt_tokens"], 5);
        assert_eq!(json["usage"]["completion_tokens"], 3);

        // DB row logged with correct token counts and extracted output text.
        let rows = state.db.query_recent(1).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].prompt_tokens, 5);
        assert_eq!(rows[0].output_tokens, 3);
        assert_eq!(rows[0].output_text, "Hello!");
    }

    // ── OpenAI streaming path ─────────────────────────────────────────────────

    const OAI_SSE_BODY: &str = concat!(
        "data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"\"},\"finish_reason\":null}]}\n",
        "\n",
        "data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hi\"},\"finish_reason\":null}]}\n",
        "\n",
        "data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\" there\"},\"finish_reason\":null}],\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":2,\"total_tokens\":7}}\n",
        "\n",
        "data: [DONE]\n",
    );

    #[tokio::test]
    async fn openai_streaming_full_sse_stream_forwarded_to_client() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(OAI_SSE_BODY),
            )
            .mount(&server)
            .await;

        let state = make_state_with_openai(
            "https://api.anthropic.com/v1/messages".to_string(),
            format!("{}/v1/chat/completions", server.uri()),
        );
        let resp = make_app(state.clone())
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(OAI_STREAM_REQ))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);

        let bytes = collect_body(resp.into_body()).await;
        let body_str = std::str::from_utf8(&bytes).unwrap();

        // SSE chunks must reach the client.
        assert!(body_str.contains("chatcmpl-1"), "missing id");
        assert!(body_str.contains("\"Hi\""), "missing first text delta");
        assert!(body_str.contains("\" there\""), "missing second text delta");
        assert!(body_str.contains("[DONE]"), "missing done marker");

        // The spawned logger needs a moment to write after the stream closes.
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;

        let rows = state.db.query_recent(1).unwrap();
        assert_eq!(rows.len(), 1, "streaming call must be logged");
        assert_eq!(rows[0].prompt_tokens, 5);
        assert_eq!(rows[0].output_tokens, 2);
        assert_eq!(rows[0].output_text, "Hi there", "output text must be reconstructed from deltas");
    }

    #[tokio::test]
    async fn openai_buffered_upstream_error_forwarded() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(
                ResponseTemplate::new(401)
                    .insert_header("content-type", "application/json")
                    .set_body_json(serde_json::json!({
                        "error": {"type": "invalid_api_key", "message": "bad key"}
                    })),
            )
            .mount(&server)
            .await;

        let state = make_state_with_openai(
            "https://api.anthropic.com/v1/messages".to_string(),
            format!("{}/v1/chat/completions", server.uri()),
        );
        let resp = make_app(state)
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(OAI_BUFFERED_REQ))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }

    // ── both providers from the same app instance ─────────────────────────────

    #[tokio::test]
    async fn both_providers_log_to_same_db() {
        let server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "application/json")
                    .set_body_json(serde_json::json!({
                        "id": "msg_01", "type": "message", "role": "assistant",
                        "model": "claude-haiku-4-5", "stop_reason": "end_turn",
                        "content": [{"type": "text", "text": "anthropic reply"}],
                        "usage": {"input_tokens": 5, "output_tokens": 3}
                    })),
            )
            .mount(&server)
            .await;

        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "application/json")
                    .set_body_json(serde_json::json!({
                        "id": "chatcmpl-1", "object": "chat.completion",
                        "choices": [{"index": 0, "message": {"role": "assistant", "content": "openai reply"}, "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6}
                    })),
            )
            .mount(&server)
            .await;

        let state = make_state_with_openai(
            format!("{}/v1/messages", server.uri()),
            format!("{}/v1/chat/completions", server.uri()),
        );
        let app = make_app(state.clone());

        // Fire Anthropic call.
        app.clone()
            .oneshot(
                Request::builder()
                    .method("POST").uri("/v1/messages")
                    .header("content-type", "application/json")
                    .body(Body::from(BUFFERED_REQ)).unwrap(),
            )
            .await.unwrap();

        // Fire OpenAI call.
        app.oneshot(
            Request::builder()
                .method("POST").uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(OAI_BUFFERED_REQ)).unwrap(),
        )
        .await.unwrap();

        let rows = state.db.query_recent(10).unwrap();
        assert_eq!(rows.len(), 2, "both calls must be logged to the same DB");
        // newest first — OpenAI was second
        assert_eq!(rows[0].output_text, "openai reply");
        assert_eq!(rows[0].prompt_tokens, 4);
        assert_eq!(rows[0].output_tokens, 2);
        assert_eq!(rows[1].output_text, "anthropic reply");
        assert_eq!(rows[1].prompt_tokens, 5);
        assert_eq!(rows[1].output_tokens, 3);
    }

    // ── live tests (require real API keys) ────────────────────────────────────

    #[tokio::test]
    #[ignore = "requires ANTHROPIC_API_KEY; run: cargo test live -- --ignored"]
    async fn live_buffered_round_trip() {
        let key = std::env::var("ANTHROPIC_API_KEY").expect("set ANTHROPIC_API_KEY");

        let state = make_state("https://api.anthropic.com/v1/messages".to_string());
        let resp = make_app(state.clone())
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/messages")
                    .header("content-type", "application/json")
                    .header("x-api-key", key)
                    .header("anthropic-version", "2023-06-01")
                    .body(Body::from(r#"{"model":"claude-haiku-4-5","max_tokens":20,"messages":[{"role":"user","content":"Reply with: pong"}]}"#))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = collect_body(resp.into_body()).await;
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(json["type"], "message", "unexpected response: {json}");
        assert!(!json["content"].as_array().unwrap().is_empty(), "empty content");
        let text = json["content"][0]["text"].as_str().unwrap_or("");
        assert!(!text.is_empty(), "response text is empty");

        let rows = state.db.query_recent(1).unwrap();
        assert_eq!(rows.len(), 1);
        assert!(rows[0].prompt_tokens > 0);
        assert!(rows[0].output_tokens > 0);
        assert!(!rows[0].output_text.is_empty());
    }

    #[tokio::test]
    #[ignore = "requires ANTHROPIC_API_KEY; run: cargo test live -- --ignored"]
    async fn live_streaming_round_trip() {
        let key = std::env::var("ANTHROPIC_API_KEY").expect("set ANTHROPIC_API_KEY");

        let state = make_state("https://api.anthropic.com/v1/messages".to_string());
        let resp = make_app(state.clone())
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/messages")
                    .header("content-type", "application/json")
                    .header("x-api-key", key)
                    .header("anthropic-version", "2023-06-01")
                    .body(Body::from(r#"{"model":"claude-haiku-4-5","max_tokens":20,"stream":true,"messages":[{"role":"user","content":"Reply with: pong"}]}"#))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = collect_body(resp.into_body()).await;
        let body_str = std::str::from_utf8(&bytes).unwrap();
        assert!(body_str.contains("message_start"), "missing SSE events");
        assert!(body_str.contains("message_stop"),  "stream not terminated");

        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        let rows = state.db.query_recent(1).unwrap();
        assert_eq!(rows.len(), 1);
        assert!(rows[0].prompt_tokens > 0);
        assert!(rows[0].output_tokens > 0);
        assert!(!rows[0].output_text.is_empty());
    }

    #[tokio::test]
    #[ignore = "requires OPENAI_API_KEY with billing credits; run: cargo test live -- --ignored"]
    async fn live_openai_buffered_round_trip() {
        let key = std::env::var("OPENAI_API_KEY").expect("set OPENAI_API_KEY");

        let state = make_state_with_openai(
            "https://api.anthropic.com/v1/messages".to_string(),
            "https://api.openai.com/v1/chat/completions".to_string(),
        );
        let resp = make_app(state.clone())
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .header("authorization", format!("Bearer {key}"))
                    .body(Body::from(r#"{"model":"gpt-4o-mini","max_tokens":20,"messages":[{"role":"user","content":"Reply with: pong"}]}"#))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = collect_body(resp.into_body()).await;
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(json["object"], "chat.completion", "unexpected response: {json}");
        let text = json["choices"][0]["message"]["content"].as_str().unwrap_or("");
        assert!(!text.is_empty(), "response text is empty");

        let rows = state.db.query_recent(1).unwrap();
        assert_eq!(rows.len(), 1);
        assert!(rows[0].prompt_tokens > 0);
        assert!(rows[0].output_tokens > 0);
        assert!(!rows[0].output_text.is_empty());
    }

    #[tokio::test]
    #[ignore = "requires OPENAI_API_KEY with billing credits; run: cargo test live -- --ignored"]
    async fn live_openai_streaming_round_trip() {
        let key = std::env::var("OPENAI_API_KEY").expect("set OPENAI_API_KEY");

        let state = make_state_with_openai(
            "https://api.anthropic.com/v1/messages".to_string(),
            "https://api.openai.com/v1/chat/completions".to_string(),
        );
        let resp = make_app(state.clone())
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .header("authorization", format!("Bearer {key}"))
                    .body(Body::from(r#"{"model":"gpt-4o-mini","max_tokens":20,"stream":true,"stream_options":{"include_usage":true},"messages":[{"role":"user","content":"Reply with: pong"}]}"#))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = collect_body(resp.into_body()).await;
        let body_str = std::str::from_utf8(&bytes).unwrap();
        assert!(body_str.contains("chat.completion.chunk"), "missing SSE chunks");
        assert!(body_str.contains("[DONE]"), "stream not terminated");

        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        let rows = state.db.query_recent(1).unwrap();
        assert_eq!(rows.len(), 1);
        assert!(rows[0].prompt_tokens > 0, "prompt_tokens not logged (need stream_options.include_usage)");
        assert!(!rows[0].output_text.is_empty());
    }
}
