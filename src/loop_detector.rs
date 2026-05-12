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
