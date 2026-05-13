use std::sync::{Arc, Mutex};

use rusqlite::{Connection, params};

// Written by the proxy on every call.
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
}

// Read back for the TUI.
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
}

#[derive(Default)]
pub struct Stats {
    pub total_calls: i64,
    pub total_cost_usd: f64,
    pub avg_latency_ms: f64,
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
                output_text   TEXT    NOT NULL DEFAULT ''
            );
            CREATE INDEX IF NOT EXISTS idx_calls_timestamp   ON calls(timestamp);
            CREATE INDEX IF NOT EXISTS idx_calls_prompt_hash ON calls(prompt_hash);",
        )?;
        // Non-destructive migrations for DBs created before these columns existed.
        conn.execute("ALTER TABLE calls ADD COLUMN loop_detected INTEGER NOT NULL DEFAULT 0", []).ok();
        conn.execute("ALTER TABLE calls ADD COLUMN input_text TEXT NOT NULL DEFAULT ''", []).ok();
        conn.execute("ALTER TABLE calls ADD COLUMN output_text TEXT NOT NULL DEFAULT ''", []).ok();
        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
        })
    }

    pub fn insert_call(&self, r: &CallRecord) -> anyhow::Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT INTO calls
             (timestamp, model, prompt_tokens, output_tokens, latency_ms, prompt_hash, cost_usd, loop_detected, input_text, output_text)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
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
            ],
        )?;
        Ok(())
    }

    pub fn query_recent(&self, limit: usize) -> anyhow::Result<Vec<CallRow>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, timestamp, model, prompt_tokens, output_tokens, latency_ms, cost_usd, loop_detected, input_text, output_text
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
        Ok(Stats {
            total_calls,
            total_cost_usd,
            avg_latency_ms,
        })
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
        // Opening the same path twice must not error (ALTER TABLE IF NOT EXISTS
        // would be ideal, but .ok() swallows the "duplicate column" error).
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
    }

    #[test]
    fn stats_aggregates_correctly() {
        let db = mem_db();
        db.insert_call(&record("m")).unwrap(); // cost 0.001, latency 500
        db.insert_call(&record("m")).unwrap(); // cost 0.001, latency 500
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
}
