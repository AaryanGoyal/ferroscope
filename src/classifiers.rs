use chrono::Utc;

use crate::db::{CallRow, Detection};
use crate::loop_detector::normalized_levenshtein;

const SIM_THRESHOLD: f64 = 0.85;
const FINGERPRINT: usize = 300;

pub struct ClassifierResult {
    pub detections: Vec<Detection>,
}

// ── retry_storm ───────────────────────────────────────────────────────────────

/// 3 or more similar prompts (sim > 0.85) within a 30-second window.
pub fn check_retry_storm(calls: &[CallRow]) -> Option<Detection> {
    // calls are ordered oldest→newest (query_recent_calls_window returns ASC).
    if calls.len() < 3 {
        return None;
    }

    // For each call, count how many *previous* calls in the window are similar.
    let fps: Vec<String> = calls
        .iter()
        .map(|c| c.input_text.chars().take(FINGERPRINT).collect())
        .collect();

    // Find a group of ≥3 calls that are mutually similar to the anchor.
    for anchor in 0..fps.len() {
        let mut group: Vec<usize> = vec![anchor];
        for j in (anchor + 1)..fps.len() {
            if normalized_levenshtein(&fps[anchor], &fps[j]) >= SIM_THRESHOLD {
                group.push(j);
            }
        }
        if group.len() >= 3 {
            let call_ids: Vec<String> = group.iter().map(|&i| calls[i].id.to_string()).collect();
            let total_cost: f64 = group.iter().map(|&i| calls[i].cost_usd).sum();
            return Some(Detection {
                timestamp: Utc::now().to_rfc3339(),
                classifier: "retry_storm".to_string(),
                call_ids: call_ids.join(","),
                detail: format!(
                    "{} near-identical prompts within 30 s (similarity ≥{:.0}%)",
                    group.len(),
                    SIM_THRESHOLD * 100.0
                ),
                suggested_fix:
                    "Add exponential back-off or a duplicate-suppression guard before retrying."
                        .to_string(),
                cost_usd: total_cost,
            });
        }
    }
    None
}

// ── cost_inflation ────────────────────────────────────────────────────────────

fn model_tier(model: &str) -> u8 {
    if model.contains("haiku") || model.contains("gpt-4o-mini") {
        1
    } else if model.contains("sonnet") || model.contains("gpt-4o") {
        2
    } else if model.contains("opus") || model.contains("gpt-4-turbo") {
        3
    } else {
        0 // unknown — skip
    }
}

/// Similar prompt sent consecutively with the model tier escalating.
/// Returns one Detection per escalating consecutive pair found in the window.
pub fn check_cost_inflation(calls: &[CallRow]) -> Vec<Detection> {
    let mut detections = Vec::new();
    if calls.len() < 2 {
        return detections;
    }
    for i in 0..calls.len() - 1 {
        let a = &calls[i];
        let b = &calls[i + 1];
        let tier_a = model_tier(&a.model);
        let tier_b = model_tier(&b.model);
        if tier_a == 0 || tier_b == 0 || tier_b <= tier_a {
            continue;
        }
        let fp_a: String = a.input_text.chars().take(FINGERPRINT).collect();
        let fp_b: String = b.input_text.chars().take(FINGERPRINT).collect();
        if normalized_levenshtein(&fp_a, &fp_b) >= SIM_THRESHOLD {
            detections.push(Detection {
                timestamp: Utc::now().to_rfc3339(),
                classifier: "cost_inflation".to_string(),
                call_ids: format!("{},{}", a.id, b.id),
                detail: format!(
                    "Model escalated from tier {} ({}) to tier {} ({}) for a similar prompt",
                    tier_a, a.model, tier_b, b.model
                ),
                suggested_fix:
                    "Pin the model explicitly; only escalate when a cheaper model fails."
                        .to_string(),
                cost_usd: a.cost_usd + b.cost_usd,
            });
        }
    }
    detections
}

// ── self_correction ───────────────────────────────────────────────────────────

const CORRECTION_PHRASES: &[&str] = &[
    "actually,",
    "wait, let me",
    "let me reconsider",
    "i made an error",
    "i was wrong",
    "correction:",
    "sorry, i",
    "upon reflection",
    "i need to correct",
    "to clarify,",
];

fn contains_correction(text: &str) -> bool {
    let lower = text.to_lowercase();
    CORRECTION_PHRASES.iter().any(|p| lower.contains(p))
}

/// Output has a correction phrase AND the next call has a similar prompt.
pub fn check_self_correction(calls: &[CallRow]) -> Option<Detection> {
    if calls.len() < 2 {
        return None;
    }
    for i in 0..calls.len() - 1 {
        let a = &calls[i];
        let b = &calls[i + 1];
        if !contains_correction(&a.output_text) {
            continue;
        }
        let fp_a: String = a.input_text.chars().take(FINGERPRINT).collect();
        let fp_b: String = b.input_text.chars().take(FINGERPRINT).collect();
        if normalized_levenshtein(&fp_a, &fp_b) >= SIM_THRESHOLD {
            return Some(Detection {
                timestamp: Utc::now().to_rfc3339(),
                classifier: "self_correction".to_string(),
                call_ids: format!("{},{}", a.id, b.id),
                detail: format!(
                    "Call #{} output contained self-correction phrase; call #{} resubmitted similar prompt",
                    a.id, b.id
                ),
                suggested_fix:
                    "Have the agent validate its output before retrying, or add a reflection step."
                        .to_string(),
                cost_usd: a.cost_usd + b.cost_usd,
            });
        }
    }
    None
}

// ── ping_pong ─────────────────────────────────────────────────────────────────

const PING_PONG_SAME: f64 = 0.80; // call[N] vs call[N-2] must be similar
const PING_PONG_DIFF: f64 = 0.40; // call[N] vs call[N-1] must be dissimilar

/// Detects an agent oscillating A-B-A-B-... between two responses.
/// Requires at least 3 consecutive calls forming one A-B-A triplet.
pub fn check_ping_pong(calls: &[CallRow]) -> Option<Detection> {
    if calls.len() < 3 {
        return None;
    }

    let fps: Vec<String> = calls
        .iter()
        .map(|c| c.output_text.chars().take(FINGERPRINT).collect())
        .collect();

    // Find the first index where a valid oscillation triplet ends (index i >= 2).
    // Then extend the run as long as the pattern holds.
    let mut run_start: Option<usize> = None;
    let mut run_end: usize = 0;
    let mut oscillation_count: usize = 0;

    for i in 2..fps.len() {
        let sim_same = normalized_levenshtein(&fps[i], &fps[i - 2]); // N vs N-2
        let sim_diff = normalized_levenshtein(&fps[i], &fps[i - 1]); // N vs N-1

        if sim_same > PING_PONG_SAME && sim_diff < PING_PONG_DIFF {
            if run_start.is_none() {
                run_start = Some(i - 2);
            }
            run_end = i;
            oscillation_count += 1;
        }
    }

    let start_idx = run_start?; // None if no triplet found

    // Collect call indices in the oscillating run.
    let involved: Vec<usize> = (start_idx..=run_end).collect();
    let call_ids: Vec<String> = involved.iter().map(|&i| calls[i].id.to_string()).collect();
    let total_cost: f64 = involved.iter().map(|&i| calls[i].cost_usd).sum();

    // Compute elapsed seconds between first and last call in the run.
    let secs = {
        let t0 = chrono::DateTime::parse_from_rfc3339(&calls[start_idx].timestamp)
            .map(|dt| dt.timestamp())
            .unwrap_or(0);
        let t1 = chrono::DateTime::parse_from_rfc3339(&calls[run_end].timestamp)
            .map(|dt| dt.timestamp())
            .unwrap_or(0);
        (t1 - t0).max(0)
    };

    Some(Detection {
        timestamp: Utc::now().to_rfc3339(),
        classifier: "ping_pong".to_string(),
        call_ids: call_ids.join(","),
        detail: format!(
            "Agent oscillating between two responses, {} oscillations in last {}s, costing ${:.4}",
            oscillation_count, secs, total_cost
        ),
        suggested_fix:
            "Add a decision lock or take majority vote across 3 samples before committing"
                .to_string(),
        cost_usd: total_cost,
    })
}

// ── run_all ───────────────────────────────────────────────────────────────────

/// Run all classifiers against the last 60-second window of calls.
/// Returns detected issues and which call IDs triggered which classifier.
pub fn run_all(db: &crate::db::Database) -> anyhow::Result<ClassifierResult> {
    let calls = db.query_recent_calls_window(60)?;
    let mut detections = Vec::new();

    if let Some(d) = check_retry_storm(&calls) {
        detections.push(d);
    }
    detections.extend(check_cost_inflation(&calls));
    if let Some(d) = check_self_correction(&calls) {
        detections.push(d);
    }
    if let Some(d) = check_ping_pong(&calls) {
        detections.push(d);
    }

    Ok(ClassifierResult { detections })
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::{CallRecord, Database};

    fn make_call(id: i64, model: &str, input: &str, output: &str, cost: f64) -> CallRow {
        CallRow {
            id,
            timestamp: "2026-05-14T00:00:00Z".to_string(),
            model: model.to_string(),
            prompt_tokens: 10,
            output_tokens: 10,
            latency_ms: 100,
            cost_usd: cost,
            loop_detected: false,
            input_text: input.to_string(),
            output_text: output.to_string(),
            classifier: None,
        }
    }

    fn similar_prompt() -> &'static str {
        "Please summarise the quarterly earnings report for Q1 2026 in three bullet points."
    }

    // ── retry_storm ───────────────────────────────────────────────────────────

    #[test]
    fn retry_storm_fires_on_three_identical_prompts() {
        let calls = vec![
            make_call(1, "claude-haiku-4-5", similar_prompt(), "ok", 0.001),
            make_call(2, "claude-haiku-4-5", similar_prompt(), "ok", 0.001),
            make_call(3, "claude-haiku-4-5", similar_prompt(), "ok", 0.001),
        ];
        let d = check_retry_storm(&calls).expect("should detect retry storm");
        assert_eq!(d.classifier, "retry_storm");
        assert!(d.call_ids.contains('1') || d.call_ids.contains('2'));
    }

    #[test]
    fn retry_storm_no_fire_on_two_similar() {
        let calls = vec![
            make_call(1, "claude-haiku-4-5", similar_prompt(), "ok", 0.001),
            make_call(2, "claude-haiku-4-5", similar_prompt(), "ok", 0.001),
        ];
        assert!(check_retry_storm(&calls).is_none());
    }

    #[test]
    fn retry_storm_no_fire_on_dissimilar_prompts() {
        let calls = vec![
            make_call(1, "claude-haiku-4-5", "summarise the Q1 report", "ok", 0.001),
            make_call(2, "claude-haiku-4-5", "write a haiku about autumn leaves falling", "ok", 0.001),
            make_call(3, "claude-haiku-4-5", "translate 'hello' to Japanese", "ok", 0.001),
        ];
        assert!(check_retry_storm(&calls).is_none());
    }

    #[test]
    fn retry_storm_cost_summed() {
        let calls = vec![
            make_call(1, "claude-haiku-4-5", similar_prompt(), "ok", 0.001),
            make_call(2, "claude-haiku-4-5", similar_prompt(), "ok", 0.002),
            make_call(3, "claude-haiku-4-5", similar_prompt(), "ok", 0.003),
        ];
        let d = check_retry_storm(&calls).unwrap();
        assert!((d.cost_usd - 0.006).abs() < 1e-9);
    }

    // ── cost_inflation ────────────────────────────────────────────────────────

    #[test]
    fn cost_inflation_detects_haiku_to_sonnet() {
        let calls = vec![
            make_call(1, "claude-haiku-4-5", similar_prompt(), "ok", 0.001),
            make_call(2, "claude-sonnet-4-6", similar_prompt(), "ok", 0.01),
        ];
        let ds = check_cost_inflation(&calls);
        assert_eq!(ds.len(), 1, "should detect one escalation");
        assert_eq!(ds[0].classifier, "cost_inflation");
        assert!(ds[0].detail.contains("tier 1"));
        assert!(ds[0].detail.contains("tier 2"));
    }

    #[test]
    fn cost_inflation_detects_sonnet_to_opus() {
        let calls = vec![
            make_call(1, "claude-sonnet-4-6", similar_prompt(), "ok", 0.01),
            make_call(2, "claude-opus-4-7", similar_prompt(), "ok", 0.1),
        ];
        let ds = check_cost_inflation(&calls);
        assert_eq!(ds.len(), 1);
        assert_eq!(ds[0].classifier, "cost_inflation");
    }

    #[test]
    fn cost_inflation_detects_full_haiku_sonnet_opus_chain() {
        let calls = vec![
            make_call(1, "claude-haiku-4-5",  similar_prompt(), "ok", 0.001),
            make_call(2, "claude-sonnet-4-6", similar_prompt(), "ok", 0.01),
            make_call(3, "claude-opus-4-7",   similar_prompt(), "ok", 0.1),
        ];
        let ds = check_cost_inflation(&calls);
        assert_eq!(ds.len(), 2, "should detect haiku→sonnet and sonnet→opus as separate escalations");
        assert!(ds[0].call_ids.contains("1") && ds[0].call_ids.contains("2"));
        assert!(ds[1].call_ids.contains("2") && ds[1].call_ids.contains("3"));
    }

    #[test]
    fn cost_inflation_no_fire_when_tier_same() {
        let calls = vec![
            make_call(1, "claude-haiku-4-5", similar_prompt(), "ok", 0.001),
            make_call(2, "claude-haiku-4-5", similar_prompt(), "ok", 0.001),
        ];
        assert!(check_cost_inflation(&calls).is_empty());
    }

    #[test]
    fn cost_inflation_no_fire_when_prompts_differ() {
        let calls = vec![
            make_call(1, "claude-haiku-4-5", "completely different question about cats", "ok", 0.001),
            make_call(2, "claude-sonnet-4-6", similar_prompt(), "ok", 0.01),
        ];
        assert!(check_cost_inflation(&calls).is_empty());
    }

    #[test]
    fn cost_inflation_no_fire_downgrade() {
        let calls = vec![
            make_call(1, "claude-opus-4-7", similar_prompt(), "ok", 0.1),
            make_call(2, "claude-haiku-4-5", similar_prompt(), "ok", 0.001),
        ];
        assert!(check_cost_inflation(&calls).is_empty());
    }

    // ── self_correction ───────────────────────────────────────────────────────

    #[test]
    fn self_correction_fires_on_correction_phrase_and_similar_next_prompt() {
        let output_with_correction = "The answer is 42. Actually, let me reconsider that.";
        let calls = vec![
            make_call(1, "claude-haiku-4-5", similar_prompt(), output_with_correction, 0.001),
            make_call(2, "claude-haiku-4-5", similar_prompt(), "ok", 0.001),
        ];
        let d = check_self_correction(&calls).expect("should detect self-correction");
        assert_eq!(d.classifier, "self_correction");
        assert!(d.call_ids.contains('1'));
        assert!(d.call_ids.contains('2'));
    }

    #[test]
    fn self_correction_no_fire_without_correction_phrase() {
        let calls = vec![
            make_call(1, "claude-haiku-4-5", similar_prompt(), "The answer is 42.", 0.001),
            make_call(2, "claude-haiku-4-5", similar_prompt(), "ok", 0.001),
        ];
        assert!(check_self_correction(&calls).is_none());
    }

    #[test]
    fn self_correction_no_fire_when_next_prompt_differs() {
        let output_with_correction = "Actually, I was wrong about that.";
        let calls = vec![
            make_call(1, "claude-haiku-4-5", similar_prompt(), output_with_correction, 0.001),
            make_call(2, "claude-haiku-4-5", "completely unrelated question about dinosaurs", "ok", 0.001),
        ];
        assert!(check_self_correction(&calls).is_none());
    }

    #[test]
    fn self_correction_detects_various_phrases() {
        let phrases = [
            "Wait, let me reconsider the approach here.",
            "I made an error in my previous calculation.",
            "I was wrong about the population figure.",
            "Sorry, I need to correct that statement.",
        ];
        for phrase in &phrases {
            let calls = vec![
                make_call(1, "claude-haiku-4-5", similar_prompt(), phrase, 0.001),
                make_call(2, "claude-haiku-4-5", similar_prompt(), "ok", 0.001),
            ];
            assert!(
                check_self_correction(&calls).is_some(),
                "phrase not detected: {phrase}"
            );
        }
    }

    // ── model_tier ────────────────────────────────────────────────────────────

    #[test]
    fn model_tier_mappings() {
        assert_eq!(model_tier("claude-haiku-4-5"), 1);
        assert_eq!(model_tier("claude-3-haiku-20240307"), 1);
        assert_eq!(model_tier("gpt-4o-mini"), 1);
        assert_eq!(model_tier("claude-sonnet-4-6"), 2);
        assert_eq!(model_tier("claude-3-5-sonnet-20241022"), 2);
        assert_eq!(model_tier("gpt-4o"), 2);
        assert_eq!(model_tier("claude-opus-4-7"), 3);
        assert_eq!(model_tier("claude-3-opus-20240229"), 3);
        assert_eq!(model_tier("totally-unknown-v99"), 0);
    }

    // ── ping_pong ─────────────────────────────────────────────────────────────

    // Two clearly distinct outputs and one that is identical to the first.
    fn output_a() -> &'static str {
        "The capital of France is Paris, a city known for the Eiffel Tower and fine cuisine."
    }
    fn output_b() -> &'static str {
        "Quantum entanglement describes a phenomenon where particles become correlated in ways that cannot be explained classically."
    }

    #[test]
    fn ping_pong_fires_on_aba_pattern() {
        let calls = vec![
            make_call(1, "claude-haiku-4-5", "q", output_a(), 0.001),
            make_call(2, "claude-haiku-4-5", "q", output_b(), 0.002),
            make_call(3, "claude-haiku-4-5", "q", output_a(), 0.003),
        ];
        let d = check_ping_pong(&calls).expect("should detect ping-pong on A-B-A");
        assert_eq!(d.classifier, "ping_pong");
        assert!(d.call_ids.contains('1'));
        assert!(d.call_ids.contains('2'));
        assert!(d.call_ids.contains('3'));
    }

    #[test]
    fn ping_pong_fires_on_abab_pattern() {
        let calls = vec![
            make_call(1, "claude-haiku-4-5", "q", output_a(), 0.001),
            make_call(2, "claude-haiku-4-5", "q", output_b(), 0.002),
            make_call(3, "claude-haiku-4-5", "q", output_a(), 0.003),
            make_call(4, "claude-haiku-4-5", "q", output_b(), 0.004),
        ];
        let d = check_ping_pong(&calls).expect("should detect ping-pong on A-B-A-B");
        assert_eq!(d.classifier, "ping_pong");
        // All 4 calls should be in the run
        assert!(d.call_ids.contains('1'));
        assert!(d.call_ids.contains('4'));
    }

    #[test]
    fn ping_pong_no_fire_on_only_two_calls() {
        let calls = vec![
            make_call(1, "claude-haiku-4-5", "q", output_a(), 0.001),
            make_call(2, "claude-haiku-4-5", "q", output_b(), 0.002),
        ];
        assert!(check_ping_pong(&calls).is_none());
    }

    #[test]
    fn ping_pong_no_fire_when_all_similar() {
        // A-A-A pattern: every-other pair is similar BUT adjacent pair is also similar
        // so sim_diff (N vs N-1) will be > 0.40, failing the dissimilarity check.
        let calls = vec![
            make_call(1, "claude-haiku-4-5", "q", output_a(), 0.001),
            make_call(2, "claude-haiku-4-5", "q", output_a(), 0.002),
            make_call(3, "claude-haiku-4-5", "q", output_a(), 0.003),
        ];
        assert!(check_ping_pong(&calls).is_none());
    }

    #[test]
    fn ping_pong_no_fire_when_all_dissimilar() {
        // Three completely different outputs — no consistent A pattern every other call.
        let output_c =
            "Machine learning models learn patterns from data through gradient descent optimization.";
        let calls = vec![
            make_call(1, "claude-haiku-4-5", "q", output_a(), 0.001),
            make_call(2, "claude-haiku-4-5", "q", output_b(), 0.002),
            make_call(3, "claude-haiku-4-5", "q", output_c, 0.003),
        ];
        assert!(check_ping_pong(&calls).is_none());
    }

    #[test]
    fn ping_pong_cost_summed() {
        let calls = vec![
            make_call(1, "claude-haiku-4-5", "q", output_a(), 0.010),
            make_call(2, "claude-haiku-4-5", "q", output_b(), 0.020),
            make_call(3, "claude-haiku-4-5", "q", output_a(), 0.030),
        ];
        let d = check_ping_pong(&calls).unwrap();
        assert!((d.cost_usd - 0.060).abs() < 1e-9);
    }

    #[test]
    fn ping_pong_detail_contains_oscillation_count() {
        let calls = vec![
            make_call(1, "claude-haiku-4-5", "q", output_a(), 0.001),
            make_call(2, "claude-haiku-4-5", "q", output_b(), 0.002),
            make_call(3, "claude-haiku-4-5", "q", output_a(), 0.003),
        ];
        let d = check_ping_pong(&calls).unwrap();
        // Detail should contain "1 oscillations" for one triplet
        assert!(
            d.detail.contains("1 oscillations"),
            "detail was: {}",
            d.detail
        );
        assert!(d.detail.contains("costing $"), "detail was: {}", d.detail);
    }

    // ── run_all integration ───────────────────────────────────────────────────

    #[test]
    fn run_all_on_empty_db_returns_no_detections() {
        let db = Database::new(":memory:").unwrap();
        let result = run_all(&db).unwrap();
        assert!(result.detections.is_empty());
    }

    #[test]
    fn run_all_detects_retry_storm_in_db() {
        let db = Database::new(":memory:").unwrap();
        for _ in 0..3 {
            db.insert_call(&CallRecord {
                timestamp: chrono::Utc::now().to_rfc3339(),
                model: "claude-haiku-4-5".to_string(),
                prompt_tokens: 10,
                output_tokens: 5,
                latency_ms: 100,
                prompt_hash: "abc".to_string(),
                cost_usd: 0.001,
                loop_detected: false,
                input_text: similar_prompt().to_string(),
                output_text: "ok".to_string(),
                classifier: None,
            }).unwrap();
        }
        let result = run_all(&db).unwrap();
        assert!(!result.detections.is_empty(), "expected at least one detection");
        assert!(result.detections.iter().any(|d| d.classifier == "retry_storm"));
    }
}
