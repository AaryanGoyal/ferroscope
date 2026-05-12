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

    if is_streaming {
        stream_response(state, upstream, status, resp_headers, model, prompt_hash, loop_detected, start).await
    } else {
        buffered_response(state, upstream, status, resp_headers, model, prompt_hash, loop_detected, start).await
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

        let (prompt_tokens, output_tokens) = parse_sse_usage(&raw);
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
    start: Instant,
) -> anyhow::Result<Response<Body>> {
    let bytes = upstream.bytes().await?;
    let latency_ms = start.elapsed().as_millis() as i64;

    let (prompt_tokens, output_tokens) = serde_json::from_slice::<Value>(&bytes)
        .map(|v| {
            (
                v["usage"]["input_tokens"].as_i64().unwrap_or(0),
                v["usage"]["output_tokens"].as_i64().unwrap_or(0),
            )
        })
        .unwrap_or((0, 0));

    let record = CallRecord {
        timestamp: chrono::Utc::now().to_rfc3339(),
        cost_usd: compute_cost(&model, prompt_tokens, output_tokens),
        model,
        prompt_tokens,
        output_tokens,
        latency_ms,
        prompt_hash,
        loop_detected,
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

/// Extract token counts from a raw Anthropic SSE response body.
fn parse_sse_usage(raw: &str) -> (i64, i64) {
    let mut input = 0i64;
    let mut output = 0i64;

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
            _ => {}
        }
    }
    (input, output)
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
