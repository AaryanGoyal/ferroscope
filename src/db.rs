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
                cost_usd      REAL    NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_calls_timestamp   ON calls(timestamp);
            CREATE INDEX IF NOT EXISTS idx_calls_prompt_hash ON calls(prompt_hash);",
        )?;
        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
        })
    }

    pub fn insert_call(&self, r: &CallRecord) -> anyhow::Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT INTO calls
             (timestamp, model, prompt_tokens, output_tokens, latency_ms, prompt_hash, cost_usd)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![
                r.timestamp,
                r.model,
                r.prompt_tokens,
                r.output_tokens,
                r.latency_ms,
                r.prompt_hash,
                r.cost_usd,
            ],
        )?;
        Ok(())
    }
}
