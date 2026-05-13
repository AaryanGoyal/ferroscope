mod db;
mod loop_detector;
mod proxy;
mod tui;

use std::sync::Arc;

use axum::{Router, routing::post};
use clap::Parser;
use tokio::sync::Mutex;
use tracing_subscriber::EnvFilter;

use db::Database;
use loop_detector::LoopDetector;

pub struct AppState {
    pub db: Database,
    pub loop_detector: Mutex<LoopDetector>,
    pub http_client: reqwest::Client,
    /// Full URL of the upstream LLM API (default: Anthropic). Overridable for
    /// testing or pointing at a custom/compatible endpoint.
    pub upstream_url: String,
}

#[derive(Parser)]
#[command(name = "ferroscope", version, about = "LLM observability proxy")]
struct Cli {
    /// Address to bind the proxy on.
    #[arg(long, default_value = "127.0.0.1:8080")]
    addr: String,

    /// SQLite database path.
    #[arg(long, default_value = "ferroscope.db")]
    db: String,

    /// Launch the ratatui TUI alongside the proxy (logs go to ferroscope.log).
    #[arg(long)]
    tui: bool,

    /// Override the upstream LLM API URL (useful for OpenAI-compatible endpoints).
    #[arg(long, default_value = "https://api.anthropic.com/v1/messages")]
    upstream: String,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    // When the TUI owns the terminal, redirect logs to a file so they don't
    // corrupt the display.
    if cli.tui {
        let log_file = std::fs::File::create("ferroscope.log")?;
        tracing_subscriber::fmt()
            .with_writer(log_file)
            .with_env_filter(
                EnvFilter::from_default_env()
                    .add_directive("ferroscope=info".parse()?),
            )
            .init();
    } else {
        tracing_subscriber::fmt()
            .with_env_filter(
                EnvFilter::from_default_env()
                    .add_directive("ferroscope=debug".parse()?),
            )
            .init();
    }

    let db = Database::new(&cli.db)?;

    if cli.tui {
        let tui_db = db.clone();
        std::thread::spawn(move || {
            if let Err(e) = tui::run(tui_db) {
                eprintln!("TUI error: {e}");
                std::process::exit(1);
            }
            // 'q' in the TUI exits the whole process.
            std::process::exit(0);
        });
    }

    let state = Arc::new(AppState {
        db,
        loop_detector: Mutex::new(LoopDetector::new()),
        http_client: reqwest::Client::new(),
        upstream_url: cli.upstream,
    });

    let app = Router::new()
        .route("/v1/messages", post(proxy::handle_messages))
        .with_state(state);

    tracing::info!("ferroscope listening on {}", cli.addr);
    let listener = tokio::net::TcpListener::bind(&cli.addr).await.map_err(|e| {
        if e.kind() == std::io::ErrorKind::AddrInUse {
            anyhow::anyhow!(
                "port {} is already in use — kill the existing process with:\n  kill $(lsof -ti :{})",
                cli.addr,
                cli.addr.split(':').last().unwrap_or("8080")
            )
        } else {
            anyhow::anyhow!(e)
        }
    })?;
    axum::serve(listener, app).await?;
    Ok(())
}
