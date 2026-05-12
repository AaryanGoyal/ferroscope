mod db;
mod loop_detector;
mod proxy;

use std::sync::Arc;

use axum::{Router, routing::post};
use tokio::sync::Mutex;
use tracing_subscriber::EnvFilter;

use db::Database;
use loop_detector::LoopDetector;

pub struct AppState {
    pub db: Database,
    pub loop_detector: Mutex<LoopDetector>,
    pub http_client: reqwest::Client,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::from_default_env()
                .add_directive("ferroscope=debug".parse()?),
        )
        .init();

    let db = Database::new("ferroscope.db")?;
    let state = Arc::new(AppState {
        db,
        loop_detector: Mutex::new(LoopDetector::new()),
        http_client: reqwest::Client::new(),
    });

    let app = Router::new()
        .route("/v1/messages", post(proxy::handle_messages))
        .with_state(state);

    let addr = "127.0.0.1:8080";
    tracing::info!("ferroscope listening on {addr}");
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}
