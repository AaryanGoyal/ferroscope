use std::sync::{Arc, Mutex};

use rusqlite::{Connection, params};

pub struct CallRecord {
    pub timestamp: String,
    pub model: String,
    pub prompt_tokens: i64,
    pub output_tokens: i64,
    pub latency_ms: i64,
    pub prompt_hash: String,
    pub cost_usd: f64,
    pub loop_detected: bool,
    pub input_text: String,
    pub output_text: String,
    pub classifier: Option<String>,
}

pub struct CallRow {
    pub id: i64,
    pub timestamp: String,
    pub model: String,
    pub prompt_tokens: i64,
    pub output_tokens: i64,
    pub latency_ms: i64,
    pub cost_usd: f64,
    pub loop_detected: bool,
    pub input_text: String,
    pub output_text: String,
    pub classifier: Option<String>,
}

pub struct Detection {
    pub timestamp: String,
    pub classifier: String,
    pub call_ids: String,
    pub detail: String,
    pub suggested_fix: String,
    pub cost_usd: f64,
}

pub struct DetectionRow {
    pub id: i64,
    pub timestamp: String,
    pub classifier: String,
    pub call_ids: String,
    pub detail: String,
    pub suggested_fix: String,
    pub cost_usd: f64,
}

#[derive(Default)]
pub struct Stats {
    pub total_calls: i64,
    pub total_cost_usd: f64,
    pub avg_latency_ms: f64,
    pub total_detections: i64,
}

#[derive(Clone)]
pub struct Database {
    conn: Arc<Mutex<Connection>>,
}

impl Database {
    pub fn new(path: &str) -> anyhow::Result<Self> {
        let conn = Connection::open(path)?;
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS calls (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp     TEXT    NOT NULL,
                model         TEXT    NOT NULL,
                prompt_tokens INTEGER NOT NULL,
                output_tokens INTEGER NOT NULL,
                latency_ms    INTEGER NOT NULL,
                prompt_hash   TEXT    NOT NULL,
                cost_usd      REAL    NOT NULL,
                loop_detected INTEGER NOT NULL DEFAULT 0,
                input_text    TEXT    NOT NULL DEFAULT '',
                output_text   TEXT    NOT NULL DEFAULT '',
                classifier    TEXT    DEFAULT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_calls_timestamp   ON calls(timestamp);
            CREATE INDEX IF NOT EXISTS idx_calls_prompt_hash ON calls(prompt_hash);
            CREATE TABLE IF NOT EXISTS detections (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp     TEXT    NOT NULL,
                classifier    TEXT    NOT NULL,
                call_ids      TEXT    NOT NULL DEFAULT '',
                detail        TEXT    NOT NULL DEFAULT '',
                suggested_fix TEXT    NOT NULL DEFAULT '',
                cost_usd      REAL    NOT NULL DEFAULT 0.0
            );
            CREATE INDEX IF NOT EXISTS idx_detections_timestamp ON detections(timestamp);",
        )?;
        // Non-destructive migrations for older DBs.
        conn.execute("ALTER TABLE calls ADD COLUMN loop_detected INTEGER NOT NULL DEFAULT 0", []).ok();
        conn.execute("ALTER TABLE calls ADD COLUMN input_text TEXT NOT NULL DEFAULT ''", []).ok();
        conn.execute("ALTER TABLE calls ADD COLUMN output_text TEXT NOT NULL DEFAULT ''", []).ok();
        conn.execute("ALTER TABLE calls ADD COLUMN classifier TEXT DEFAULT NULL", []).ok();
        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
        })
    }

    /// Insert a call record and return its rowid.
    pub fn insert_call(&self, r: &CallRecord) -> anyhow::Result<i64> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT INTO calls
             (timestamp, model, prompt_tokens, output_tokens, latency_ms, prompt_hash, cost_usd, loop_detected, input_text, output_text, classifier)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
            params![
                r.timestamp,
                r.model,
                r.prompt_tokens,
                r.output_tokens,
                r.latency_ms,
                r.prompt_hash,
                r.cost_usd,
                r.loop_detected as i64,
                r.input_text,
                r.output_text,
                r.classifier,
            ],
        )?;
        Ok(conn.last_insert_rowid())
    }

    /// Returns true if a detection with this exact (classifier, call_ids) already exists.
    pub fn detection_exists(&self, classifier: &str, call_ids: &str) -> anyhow::Result<bool> {
        let conn = self.conn.lock().unwrap();
        let count: i64 = conn.query_row(
            "SELECT COUNT(*) FROM detections WHERE classifier = ?1 AND call_ids = ?2",
            params![classifier, call_ids],
            |r| r.get(0),
        )?;
        Ok(count > 0)
    }

    pub fn update_call_classifier(&self, id: i64, classifier: &str) -> anyhow::Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "UPDATE calls SET classifier = ?1 WHERE id = ?2",
            params![classifier, id],
        )?;
        Ok(())
    }

    pub fn insert_detection(&self, d: &Detection) -> anyhow::Result<i64> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT INTO detections (timestamp, classifier, call_ids, detail, suggested_fix, cost_usd)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![d.timestamp, d.classifier, d.call_ids, d.detail, d.suggested_fix, d.cost_usd],
        )?;
        Ok(conn.last_insert_rowid())
    }

    pub fn query_recent(&self, limit: usize) -> anyhow::Result<Vec<CallRow>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, timestamp, model, prompt_tokens, output_tokens, latency_ms, cost_usd, loop_detected, input_text, output_text, classifier
             FROM calls ORDER BY id DESC LIMIT ?1",
        )?;
        let rows = stmt
            .query_map([limit as i64], |row| {
                Ok(CallRow {
                    id: row.get(0)?,
                    timestamp: row.get(1)?,
                    model: row.get(2)?,
                    prompt_tokens: row.get(3)?,
                    output_tokens: row.get(4)?,
                    latency_ms: row.get(5)?,
                    cost_usd: row.get(6)?,
                    loop_detected: row.get::<_, i64>(7)? != 0,
                    input_text: row.get(8)?,
                    output_text: row.get(9)?,
                    classifier: row.get(10)?,
                })
            })?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    pub fn query_recent_detections(&self, limit: usize) -> anyhow::Result<Vec<DetectionRow>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, timestamp, classifier, call_ids, detail, suggested_fix, cost_usd
             FROM detections ORDER BY id DESC LIMIT ?1",
        )?;
        let rows = stmt
            .query_map([limit as i64], |row| {
                Ok(DetectionRow {
                    id: row.get(0)?,
                    timestamp: row.get(1)?,
                    classifier: row.get(2)?,
                    call_ids: row.get(3)?,
                    detail: row.get(4)?,
                    suggested_fix: row.get(5)?,
                    cost_usd: row.get(6)?,
                })
            })?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    pub fn query_stats(&self) -> anyhow::Result<Stats> {
        let conn = self.conn.lock().unwrap();
        let (total_calls, total_cost_usd, avg_latency_ms) = conn.query_row(
            "SELECT COUNT(*), COALESCE(SUM(cost_usd), 0.0), COALESCE(AVG(latency_ms), 0.0) FROM calls",
            [],
            |row| Ok((row.get::<_, i64>(0)?, row.get::<_, f64>(1)?, row.get::<_, f64>(2)?)),
        )?;
        let total_detections = conn.query_row(
            "SELECT COUNT(*) FROM detections",
            [],
            |row| row.get::<_, i64>(0),
        )?;
        Ok(Stats {
            total_calls,
            total_cost_usd,
            avg_latency_ms,
            total_detections,
        })
    }

    /// Stats for calls inserted after `since_timestamp` (RFC3339 string).
    pub fn query_stats_since(&self, since: &str) -> anyhow::Result<Stats> {
        let conn = self.conn.lock().unwrap();
        let (total_calls, total_cost_usd, avg_latency_ms) = conn.query_row(
            "SELECT COUNT(*), COALESCE(SUM(cost_usd), 0.0), COALESCE(AVG(latency_ms), 0.0)
             FROM calls WHERE timestamp >= ?1",
            params![since],
            |row| Ok((row.get::<_, i64>(0)?, row.get::<_, f64>(1)?, row.get::<_, f64>(2)?)),
        )?;
        let total_detections = conn.query_row(
            "SELECT COUNT(*) FROM detections WHERE timestamp >= ?1",
            params![since],
            |row| row.get::<_, i64>(0),
        )?;
        Ok(Stats {
            total_calls,
            total_cost_usd,
            avg_latency_ms,
            total_detections,
        })
    }

    /// Recent calls within the past `window_secs` seconds, oldest first.
    pub fn query_recent_calls_window(&self, window_secs: i64) -> anyhow::Result<Vec<CallRow>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, timestamp, model, prompt_tokens, output_tokens, latency_ms, cost_usd, loop_detected, input_text, output_text, classifier
             FROM calls
             WHERE timestamp >= datetime('now', ?1)
             ORDER BY id ASC",
        )?;
        let arg = format!("-{window_secs} seconds");
        let rows = stmt
            .query_map([arg], |row| {
                Ok(CallRow {
                    id: row.get(0)?,
                    timestamp: row.get(1)?,
                    model: row.get(2)?,
                    prompt_tokens: row.get(3)?,
                    output_tokens: row.get(4)?,
                    latency_ms: row.get(5)?,
                    cost_usd: row.get(6)?,
                    loop_detected: row.get::<_, i64>(7)? != 0,
                    input_text: row.get(8)?,
                    output_text: row.get(9)?,
                    classifier: row.get(10)?,
                })
            })?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(rows)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mem_db() -> Database {
        Database::new(":memory:").expect("in-memory db")
    }

    fn record(model: &str) -> CallRecord {
        CallRecord {
            timestamp: "2026-05-14T00:00:00Z".to_string(),
            model: model.to_string(),
            prompt_tokens: 10,
            output_tokens: 20,
            latency_ms: 500,
            prompt_hash: "deadbeef".to_string(),
            cost_usd: 0.001,
            loop_detected: false,
            input_text: "[user]\nhello".to_string(),
            output_text: "hi there".to_string(),
            classifier: None,
        }
    }

    // ── schema / migrations ───────────────────────────────────────────────────

    #[test]
    fn creates_empty_table() {
        let db = mem_db();
        let rows = db.query_recent(10).unwrap();
        assert_eq!(rows.len(), 0);
    }

    #[test]
    fn migration_is_idempotent() {
        let db = mem_db();
        db.insert_call(&record("m")).unwrap();
        let rows = db.query_recent(10).unwrap();
        assert_eq!(rows.len(), 1);
    }

    // ── insert_call / query_recent ────────────────────────────────────────────

    #[test]
    fn round_trip_all_fields() {
        let db = mem_db();
        let mut r = record("claude-haiku-4-5");
        r.loop_detected = true;
        r.input_text = "[user]\ncount to 3".to_string();
        r.output_text = "1\n2\n3".to_string();
        r.classifier = Some("retry_storm".to_string());
        db.insert_call(&r).unwrap();

        let rows = db.query_recent(1).unwrap();
        assert_eq!(rows.len(), 1);
        let row = &rows[0];
        assert_eq!(row.model, "claude-haiku-4-5");
        assert_eq!(row.prompt_tokens, 10);
        assert_eq!(row.output_tokens, 20);
        assert_eq!(row.latency_ms, 500);
        assert!((row.cost_usd - 0.001).abs() < 1e-9);
        assert!(row.loop_detected);
        assert_eq!(row.input_text, "[user]\ncount to 3");
        assert_eq!(row.output_text, "1\n2\n3");
        assert_eq!(row.classifier.as_deref(), Some("retry_storm"));
    }

    #[test]
    fn query_recent_newest_first() {
        let db = mem_db();
        db.insert_call(&record("a")).unwrap();
        db.insert_call(&record("b")).unwrap();
        db.insert_call(&record("c")).unwrap();
        let rows = db.query_recent(10).unwrap();
        assert_eq!(rows[0].model, "c");
        assert_eq!(rows[1].model, "b");
        assert_eq!(rows[2].model, "a");
    }

    #[test]
    fn query_recent_respects_limit() {
        let db = mem_db();
        for _ in 0..10 {
            db.insert_call(&record("m")).unwrap();
        }
        assert_eq!(db.query_recent(3).unwrap().len(), 3);
    }

    // ── query_stats ───────────────────────────────────────────────────────────

    #[test]
    fn stats_empty_db() {
        let s = mem_db().query_stats().unwrap();
        assert_eq!(s.total_calls, 0);
        assert_eq!(s.total_cost_usd, 0.0);
        assert_eq!(s.avg_latency_ms, 0.0);
        assert_eq!(s.total_detections, 0);
    }

    #[test]
    fn stats_aggregates_correctly() {
        let db = mem_db();
        db.insert_call(&record("m")).unwrap();
        db.insert_call(&record("m")).unwrap();
        let s = db.query_stats().unwrap();
        assert_eq!(s.total_calls, 2);
        assert!((s.total_cost_usd - 0.002).abs() < 1e-9);
        assert!((s.avg_latency_ms - 500.0).abs() < 1e-9);
    }

    #[test]
    fn loop_detected_flag_persisted() {
        let db = mem_db();
        let mut r = record("m");
        r.loop_detected = true;
        db.insert_call(&r).unwrap();
        assert!(db.query_recent(1).unwrap()[0].loop_detected);
    }

    // ── detections ────────────────────────────────────────────────────────────

    #[test]
    fn insert_and_query_detection() {
        let db = mem_db();
        let d = Detection {
            timestamp: "2026-05-14T00:00:00Z".to_string(),
            classifier: "retry_storm".to_string(),
            call_ids: "1,2,3".to_string(),
            detail: "3 similar prompts in 30s".to_string(),
            suggested_fix: "add backoff".to_string(),
            cost_usd: 0.005,
        };
        db.insert_detection(&d).unwrap();
        let rows = db.query_recent_detections(10).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].classifier, "retry_storm");
        assert_eq!(rows[0].call_ids, "1,2,3");
        assert_eq!(rows[0].detail, "3 similar prompts in 30s");
        assert_eq!(rows[0].suggested_fix, "add backoff");
        assert!((rows[0].cost_usd - 0.005).abs() < 1e-9);
    }

    #[test]
    fn stats_counts_detections() {
        let db = mem_db();
        let d = Detection {
            timestamp: "2026-05-14T00:00:00Z".to_string(),
            classifier: "cost_inflation".to_string(),
            call_ids: "1".to_string(),
            detail: "tier jumped".to_string(),
            suggested_fix: "fix routing".to_string(),
            cost_usd: 0.0,
        };
        db.insert_detection(&d).unwrap();
        assert_eq!(db.query_stats().unwrap().total_detections, 1);
    }

    #[test]
    fn update_call_classifier_sets_field() {
        let db = mem_db();
        let id = db.insert_call(&record("m")).unwrap();
        db.update_call_classifier(id, "cost_inflation").unwrap();
        let rows = db.query_recent(1).unwrap();
        assert_eq!(rows[0].classifier.as_deref(), Some("cost_inflation"));
    }
}
