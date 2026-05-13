use std::collections::VecDeque;

const WINDOW: usize = 5;
const TRIGGER_COUNT: usize = 2;
const THRESHOLD: f64 = 0.85;
// Cap compared text to bound Levenshtein to O(CAP^2) = ~90k ops worst-case.
const FINGERPRINT_CHARS: usize = 300;

pub struct LoopDetector {
    recent: VecDeque<String>,
}

pub struct LoopWarning {
    pub similar_count: usize,
    pub max_similarity: f64,
}

impl LoopDetector {
    pub fn new() -> Self {
        Self {
            recent: VecDeque::with_capacity(WINDOW),
        }
    }

    /// Record `prompt_text` and return a warning if the call pattern looks loopy.
    pub fn check_and_record(&mut self, prompt_text: &str) -> Option<LoopWarning> {
        let fp: String = prompt_text.chars().take(FINGERPRINT_CHARS).collect();

        let mut similar_count = 0;
        let mut max_similarity = 0.0f64;

        for prev in &self.recent {
            let sim = normalized_levenshtein(&fp, prev);
            if sim >= THRESHOLD {
                similar_count += 1;
            }
            max_similarity = max_similarity.max(sim);
        }

        if self.recent.len() == WINDOW {
            self.recent.pop_front();
        }
        self.recent.push_back(fp);

        if similar_count >= TRIGGER_COUNT {
            Some(LoopWarning {
                similar_count,
                max_similarity,
            })
        } else {
            None
        }
    }
}

/// Returns a value in [0.0, 1.0]: 1.0 = identical, 0.0 = nothing in common.
fn normalized_levenshtein(a: &str, b: &str) -> f64 {
    if a.is_empty() && b.is_empty() {
        return 1.0;
    }
    let max_len = a.chars().count().max(b.chars().count());
    if max_len == 0 {
        return 1.0;
    }
    1.0 - levenshtein(a, b) as f64 / max_len as f64
}

fn levenshtein(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let (m, n) = (a.len(), b.len());

    // Use two-row rolling array instead of full m×n matrix.
    let mut prev: Vec<usize> = (0..=n).collect();
    let mut curr = vec![0usize; n + 1];

    for i in 1..=m {
        curr[0] = i;
        for j in 1..=n {
            curr[j] = if a[i - 1] == b[j - 1] {
                prev[j - 1]
            } else {
                1 + prev[j - 1].min(prev[j]).min(curr[j - 1])
            };
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[n]
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── levenshtein ───────────────────────────────────────────────────────────

    #[test]
    fn lev_identical() {
        assert_eq!(levenshtein("abc", "abc"), 0);
    }

    #[test]
    fn lev_both_empty() {
        assert_eq!(levenshtein("", ""), 0);
    }

    #[test]
    fn lev_one_empty() {
        assert_eq!(levenshtein("abc", ""), 3);
        assert_eq!(levenshtein("", "abc"), 3);
    }

    #[test]
    fn lev_single_substitution() {
        assert_eq!(levenshtein("abc", "abd"), 1);
    }

    #[test]
    fn lev_classic_kitten_sitting() {
        // well-known example: kitten → sitting = 3 ops
        assert_eq!(levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn lev_insertion_deletion() {
        assert_eq!(levenshtein("cat", "cats"), 1);
        assert_eq!(levenshtein("cats", "cat"), 1);
    }

    // ── normalized_levenshtein ────────────────────────────────────────────────

    #[test]
    fn norm_lev_identical() {
        assert!((normalized_levenshtein("hello", "hello") - 1.0).abs() < 1e-9);
    }

    #[test]
    fn norm_lev_both_empty() {
        assert!((normalized_levenshtein("", "") - 1.0).abs() < 1e-9);
    }

    #[test]
    fn norm_lev_completely_different() {
        // "abc" vs "xyz": all 3 chars differ, distance=3, max_len=3 → 0.0
        let sim = normalized_levenshtein("abc", "xyz");
        assert!((sim - 0.0).abs() < 1e-9);
    }

    #[test]
    fn norm_lev_one_char_off() {
        // "abcd" vs "abce": distance=1, max_len=4 → 0.75
        let sim = normalized_levenshtein("abcd", "abce");
        assert!((sim - 0.75).abs() < 1e-9);
    }

    #[test]
    fn norm_lev_asymmetric_length() {
        // "ab" vs "abcd": distance=2, max_len=4 → 0.5
        let sim = normalized_levenshtein("ab", "abcd");
        assert!((sim - 0.5).abs() < 1e-9);
    }

    // ── LoopDetector ─────────────────────────────────────────────────────────

    #[test]
    fn no_warning_on_first_call() {
        let mut det = LoopDetector::new();
        assert!(det.check_and_record("hello world").is_none());
    }

    #[test]
    fn no_warning_on_dissimilar_calls() {
        let mut det = LoopDetector::new();
        det.check_and_record("rust programming language");
        det.check_and_record("french cuisine and cooking");
        let result = det.check_and_record("quantum mechanics wave function");
        assert!(result.is_none());
    }

    #[test]
    fn warns_after_trigger_count_similar_prompts() {
        let mut det = LoopDetector::new();
        let prompt = "What is the capital of France?";
        det.check_and_record(prompt); // window: [p]      — 0 matches
        det.check_and_record(prompt); // window: [p, p]   — 1 match, no warn
        let result = det.check_and_record(prompt); // window: [p,p,p] — 2 matches → warn
        let w = result.expect("should have warned after 2 identical prompts");
        assert!(w.similar_count >= TRIGGER_COUNT);
        assert!(w.max_similarity > 0.99);
    }

    #[test]
    fn one_similar_below_trigger_count_is_not_a_loop() {
        let mut det = LoopDetector::new();
        det.check_and_record("What is the capital of France?");
        // Second call is very different — only one prior similar entry
        let result = det.check_and_record("What is the capital of France?");
        // TRIGGER_COUNT = 2, but window only had 1 entry → similar_count = 1 → None
        assert!(result.is_none());
    }

    #[test]
    fn window_eviction_clears_old_matches() {
        let mut det = LoopDetector::new();
        let old = "old prompt that should be evicted eventually";
        // Fill window with `old`
        for _ in 0..WINDOW {
            det.check_and_record(old);
        }
        // Push `old` out by filling the window with completely different prompts
        for i in 0..WINDOW {
            det.check_and_record(&format!("distinct entry number {i} zqxvwk"));
        }
        // Window now contains only the distinct entries; `old` is gone
        let result = det.check_and_record(old);
        assert!(result.is_none(), "evicted prompt should not trigger a loop warning");
    }

    #[test]
    fn warning_includes_correct_similarity_metadata() {
        let mut det = LoopDetector::new();
        let prompt = "repeated agent task: summarise the document";
        det.check_and_record(prompt);
        det.check_and_record(prompt);
        let w = det.check_and_record(prompt).unwrap();
        assert_eq!(w.similar_count, 2);
        assert!(w.max_similarity > 0.99);
    }
}
