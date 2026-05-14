mod classifiers;
mod db;
mod loop_detector;
mod proxy;
mod tui;

use std::sync::Arc;

use clap::{Parser, Subcommand};
use tokio::sync::Mutex;
use tracing_subscriber::EnvFilter;

use db::Database;
use loop_detector::LoopDetector;

pub struct AppState {
    pub db: Database,
    pub loop_detector: Mutex<LoopDetector>,
    pub http_client: reqwest::Client,
    pub anthropic_upstream: String,
    pub openai_upstream: String,
}

#[derive(Parser)]
#[command(name = "ferroscope", version, about = "LLM observability proxy")]
struct Cli {
    /// Address to bind the proxy on (standalone server mode only).
    #[arg(long, default_value = "127.0.0.1:8080", global = true)]
    addr: String,

    /// SQLite database path.
    #[arg(long, default_value = "ferroscope.db", global = true)]
    db: String,

    /// Launch the ratatui TUI alongside the proxy (logs go to ferroscope.log).
    #[arg(long, global = true)]
    tui: bool,

    /// Override the upstream Anthropic API URL.
    #[arg(long, default_value = "https://api.anthropic.com/v1/messages", global = true)]
    anthropic_upstream: String,

    /// Override the upstream OpenAI API URL.
    #[arg(long, default_value = "https://api.openai.com/v1/chat/completions", global = true)]
    openai_upstream: String,

    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Wrap a command: start proxy, inject env vars, run cmd, print summary.
    Run(RunArgs),
}

#[derive(Parser)]
struct RunArgs {
    /// Command and arguments to run (use -- to separate flags).
    #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
    cmd: Vec<String>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Some(Commands::Run(args)) => {
            run_subcommand(&cli.db, &cli.anthropic_upstream, &cli.openai_upstream, cli.tui, args).await
        }
        None => {
            serve(&cli.addr, &cli.db, &cli.anthropic_upstream, &cli.openai_upstream, cli.tui).await
        }
    }
}

async fn serve(addr: &str, db_path: &str, anthropic_upstream: &str, openai_upstream: &str, with_tui: bool) -> anyhow::Result<()> {
    if with_tui {
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

    let db = Database::new(db_path)?;

    if with_tui {
        let tui_db = db.clone();
        std::thread::spawn(move || {
            if let Err(e) = tui::run(tui_db) {
                eprintln!("TUI error: {e}");
                std::process::exit(1);
            }
            std::process::exit(0);
        });
    }

    start_server(addr, db, anthropic_upstream, openai_upstream).await
}

async fn start_server(addr: &str, db: Database, anthropic_upstream: &str, openai_upstream: &str) -> anyhow::Result<()> {
    let state = Arc::new(AppState {
        db,
        loop_detector: Mutex::new(LoopDetector::new()),
        http_client: reqwest::Client::new(),
        anthropic_upstream: anthropic_upstream.to_string(),
        openai_upstream: openai_upstream.to_string(),
    });

    let app = proxy::make_app(state);

    tracing::info!("ferroscope listening on {}", addr);
    let listener = tokio::net::TcpListener::bind(addr).await.map_err(|e| {
        if e.kind() == std::io::ErrorKind::AddrInUse {
            anyhow::anyhow!(
                "port {} is already in use — kill the existing process with:\n  kill $(lsof -ti :{})",
                addr,
                addr.split(':').last().unwrap_or("8080")
            )
        } else {
            anyhow::anyhow!(e)
        }
    })?;
    axum::serve(listener, app).await?;
    Ok(())
}

async fn run_subcommand(
    db_path: &str,
    anthropic_upstream: &str,
    openai_upstream: &str,
    with_tui: bool,
    args: RunArgs,
) -> anyhow::Result<()> {
    if args.cmd.is_empty() {
        anyhow::bail!("Usage: ferroscope run [--tui] [--] <command> [args...]");
    }

    // Init logging to stderr so it doesn't pollute child stdout.
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(
            EnvFilter::from_default_env()
                .add_directive("ferroscope=info".parse()?),
        )
        .init();

    // Bind port 0 → OS assigns a free port; keep listener alive to hold the port.
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
    let port = listener.local_addr()?.port();
    let base_url = format!("http://127.0.0.1:{port}");

    tracing::info!("ferroscope proxy on :{port}");

    let db = Database::new(db_path)?;
    let session_start = chrono::Utc::now().to_rfc3339();

    if with_tui {
        let tui_db = db.clone();
        std::thread::spawn(move || {
            if let Err(e) = tui::run(tui_db) {
                eprintln!("TUI error: {e}");
            }
        });
    }

    // Start proxy in background.
    let state = Arc::new(AppState {
        db: db.clone(),
        loop_detector: Mutex::new(LoopDetector::new()),
        http_client: reqwest::Client::new(),
        anthropic_upstream: anthropic_upstream.to_string(),
        openai_upstream: openai_upstream.to_string(),
    });
    let app = proxy::make_app(state);

    tokio::spawn(async move {
        if let Err(e) = axum::serve(listener, app).await {
            tracing::error!("proxy error: {e}");
        }
    });

    // Launch child process with env vars injected.
    let (program, child_args) = args.cmd.split_first().unwrap();
    let status = tokio::process::Command::new(program)
        .args(child_args)
        .env("ANTHROPIC_BASE_URL", &base_url)
        .env("OPENAI_BASE_URL", &base_url)
        .status()
        .await?;

    print_summary(&db, &session_start)?;

    std::process::exit(status.code().unwrap_or(1));
}

fn print_summary(db: &Database, since: &str) -> anyhow::Result<()> {
    let stats = db.query_stats_since(since)?;
    let detections = db.query_recent_detections(100)?;
    let session_detections: Vec<_> = detections.iter().filter(|d| d.timestamp.as_str() >= since).collect();

    println!("\n── ferroscope session summary ──────────────────────");
    println!("  calls:      {}", stats.total_calls);
    println!("  total cost: ${:.6}", stats.total_cost_usd);
    println!("  avg latency:{:.0} ms", stats.avg_latency_ms);
    println!("  detections: {}", session_detections.len());
    for d in &session_detections {
        println!("    [{}] {} — {}", d.classifier, d.call_ids, d.detail);
    }
    println!("────────────────────────────────────────────────────");
    Ok(())
}
