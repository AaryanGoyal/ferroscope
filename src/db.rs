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
                loop_detected INTEGER NOT NULL DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_calls_timestamp   ON calls(timestamp);
            CREATE INDEX IF NOT EXISTS idx_calls_prompt_hash ON calls(prompt_hash);",
        )?;
        // Non-destructive migration for DBs created before loop_detected existed.
        conn.execute(
            "ALTER TABLE calls ADD COLUMN loop_detected INTEGER NOT NULL DEFAULT 0",
            [],
        )
        .ok();
        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
        })
    }

    pub fn insert_call(&self, r: &CallRecord) -> anyhow::Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT INTO calls
             (timestamp, model, prompt_tokens, output_tokens, latency_ms, prompt_hash, cost_usd, loop_detected)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            params![
                r.timestamp,
                r.model,
                r.prompt_tokens,
                r.output_tokens,
                r.latency_ms,
                r.prompt_hash,
                r.cost_usd,
                r.loop_detected as i64,
            ],
        )?;
        Ok(())
    }

    pub fn query_recent(&self, limit: usize) -> anyhow::Result<Vec<CallRow>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, timestamp, model, prompt_tokens, output_tokens, latency_ms, cost_usd, loop_detected
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
