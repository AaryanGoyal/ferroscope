use std::io;
use std::time::Duration;

use crossterm::event::{self, Event, KeyCode, KeyEventKind};
use crossterm::execute;
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout};
use ratatui::style::{Color, Modifier, Style};
use ratatui::widgets::{Block, Borders, Cell, Paragraph, Row, Table, TableState};
use ratatui::Terminal;

use crate::db::{CallRow, Database, Stats};

// Restores the terminal when dropped — works even on panic.
struct TerminalHandle {
    terminal: Terminal<CrosstermBackend<io::Stdout>>,
}

impl TerminalHandle {
    fn new() -> anyhow::Result<Self> {
        enable_raw_mode()?;
        let mut stdout = io::stdout();
        execute!(stdout, EnterAlternateScreen)?;
        let backend = CrosstermBackend::new(io::stdout());
        let terminal = Terminal::new(backend)?;
        Ok(Self { terminal })
    }
}

impl Drop for TerminalHandle {
    fn drop(&mut self) {
        let _ = disable_raw_mode();
        let _ = execute!(self.terminal.backend_mut(), LeaveAlternateScreen);
    }
}

struct TuiState {
    rows: Vec<CallRow>,
    stats: Stats,
    table_state: TableState,
}

impl TuiState {
    fn new() -> Self {
        Self {
            rows: vec![],
            stats: Stats::default(),
            table_state: TableState::default(),
        }
    }

    fn refresh(&mut self, db: &Database) {
        self.rows = db.query_recent(100).unwrap_or_default();
        self.stats = db.query_stats().unwrap_or_default();
    }

    fn scroll_up(&mut self) {
        let i = self
            .table_state
            .selected()
            .map(|i| i.saturating_sub(1))
            .unwrap_or(0);
        self.table_state.select(Some(i));
    }

    fn scroll_down(&mut self) {
        if self.rows.is_empty() {
            return;
        }
        let i = self
            .table_state
            .selected()
            .map(|i| (i + 1).min(self.rows.len() - 1))
            .unwrap_or(0);
        self.table_state.select(Some(i));
    }
}

pub fn run(db: Database) -> anyhow::Result<()> {
    let mut handle = TerminalHandle::new()?;
    let mut state = TuiState::new();

    loop {
        state.refresh(&db);
        handle.terminal.draw(|f| render(f, &mut state))?;

        if event::poll(Duration::from_millis(500))? {
            if let Event::Key(key) = event::read()? {
                if key.kind != KeyEventKind::Press {
                    continue;
                }
                match key.code {
                    KeyCode::Char('q') | KeyCode::Esc => {
                        drop(handle); // restore terminal before exit
                        std::process::exit(0);
                    }
                    KeyCode::Up | KeyCode::Char('k') => state.scroll_up(),
                    KeyCode::Down | KeyCode::Char('j') => state.scroll_down(),
                    _ => {}
                }
            }
        }
    }
}

fn render(f: &mut ratatui::Frame, state: &mut TuiState) {
    let area = f.area();
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3), // stats header
            Constraint::Min(0),    // call table
            Constraint::Length(1), // key hint footer
        ])
        .split(area);

    // ── stats header ─────────────────────────────────────────────────────────
    let header_text = format!(
        " calls: {}   cost: ${:.4}   avg latency: {:.0}ms",
        state.stats.total_calls, state.stats.total_cost_usd, state.stats.avg_latency_ms,
    );
    f.render_widget(
        Paragraph::new(header_text)
            .block(
                Block::default()
                    .title(" ferroscope ")
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(Color::Cyan)),
            )
            .style(Style::default().fg(Color::Cyan)),
        chunks[0],
    );

    // ── call table ───────────────────────────────────────────────────────────
    let col_header = Row::new(["#", "timestamp", "model", "in", "out", "ms", "$"])
        .style(
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        )
        .height(1);

    let rows: Vec<Row> = state
        .rows
        .iter()
        .map(|r| {
            let flag = if r.loop_detected { "⚠" } else { " " };
            let style = if r.loop_detected {
                Style::default().fg(Color::Red)
            } else {
                Style::default()
            };
            let ts = r
                .timestamp
                .get(..19)
                .unwrap_or(&r.timestamp)
                .replace('T', " ");
            Row::new(vec![
                Cell::from(format!("{} {}", flag, r.id)),
                Cell::from(ts),
                Cell::from(r.model.clone()),
                Cell::from(r.prompt_tokens.to_string()),
                Cell::from(r.output_tokens.to_string()),
                Cell::from(r.latency_ms.to_string()),
                Cell::from(format!("${:.4}", r.cost_usd)),
            ])
            .style(style)
        })
        .collect();

    let widths = [
        Constraint::Length(6),  // id + flag
        Constraint::Length(20), // timestamp
        Constraint::Min(18),    // model (elastic)
        Constraint::Length(7),  // in tokens
        Constraint::Length(7),  // out tokens
        Constraint::Length(7),  // latency
        Constraint::Length(9),  // cost
    ];

    let table = Table::new(rows, widths)
        .header(col_header)
        .block(
            Block::default()
                .title(" calls (newest first) ")
                .borders(Borders::ALL),
        )
        .row_highlight_style(Style::default().add_modifier(Modifier::REVERSED))
        .highlight_symbol("▶ ");

    f.render_stateful_widget(table, chunks[1], &mut state.table_state);

    // ── footer ───────────────────────────────────────────────────────────────
    f.render_widget(
        Paragraph::new("  q quit   ↑/k ↓/j scroll   ⚠ = loop detected")
            .style(Style::default().fg(Color::DarkGray)),
        chunks[2],
    );
}
