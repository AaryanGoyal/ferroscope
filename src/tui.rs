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
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{
    Block, Borders, Cell, Paragraph, Row, Table, TableState, Wrap,
};
use ratatui::Terminal;

use crate::db::{CallRow, Database, DetectionRow, Stats};

#[derive(Clone, Copy, PartialEq)]
enum ViewMode {
    Calls,
    Detections,
}

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
    mode: ViewMode,
    rows: Vec<CallRow>,
    detections: Vec<DetectionRow>,
    stats: Stats,
    calls_table: TableState,
    detections_table: TableState,
}

impl TuiState {
    fn new() -> Self {
        Self {
            mode: ViewMode::Calls,
            rows: vec![],
            detections: vec![],
            stats: Stats::default(),
            calls_table: TableState::default(),
            detections_table: TableState::default(),
        }
    }

    fn refresh(&mut self, db: &Database) {
        self.rows = db.query_recent(100).unwrap_or_default();
        self.detections = db.query_recent_detections(100).unwrap_or_default();
        self.stats = db.query_stats().unwrap_or_default();

        let clamp = |state: &mut TableState, len: usize| {
            if let Some(i) = state.selected() {
                if len == 0 {
                    state.select(None);
                } else if i >= len {
                    state.select(Some(len - 1));
                }
            }
        };
        clamp(&mut self.calls_table, self.rows.len());
        clamp(&mut self.detections_table, self.detections.len());
    }

    fn scroll_up(&mut self) {
        match self.mode {
            ViewMode::Calls => {
                let i = self.calls_table.selected().map(|i| i.saturating_sub(1)).unwrap_or(0);
                self.calls_table.select(Some(i));
            }
            ViewMode::Detections => {
                let i = self.detections_table.selected().map(|i| i.saturating_sub(1)).unwrap_or(0);
                self.detections_table.select(Some(i));
            }
        }
    }

    fn scroll_down(&mut self) {
        match self.mode {
            ViewMode::Calls => {
                if self.rows.is_empty() { return; }
                let i = self.calls_table.selected()
                    .map(|i| (i + 1).min(self.rows.len() - 1))
                    .unwrap_or(0);
                self.calls_table.select(Some(i));
            }
            ViewMode::Detections => {
                if self.detections.is_empty() { return; }
                let i = self.detections_table.selected()
                    .map(|i| (i + 1).min(self.detections.len() - 1))
                    .unwrap_or(0);
                self.detections_table.select(Some(i));
            }
        }
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
                        drop(handle);
                        std::process::exit(0);
                    }
                    KeyCode::Tab => {
                        state.mode = match state.mode {
                            ViewMode::Calls => ViewMode::Detections,
                            ViewMode::Detections => ViewMode::Calls,
                        };
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
            Constraint::Length(3),      // stats header
            Constraint::Percentage(45), // table
            Constraint::Min(6),         // detail panel
            Constraint::Length(1),      // key hints
        ])
        .split(area);

    // ── stats header ─────────────────────────────────────────────────────────
    f.render_widget(
        Paragraph::new(format!(
            " calls: {}   cost: ${:.4}   avg latency: {:.0}ms   detections: {}",
            state.stats.total_calls,
            state.stats.total_cost_usd,
            state.stats.avg_latency_ms,
            state.stats.total_detections,
        ))
        .block(
            Block::default()
                .title(" ferroscope ")
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Cyan)),
        )
        .style(Style::default().fg(Color::Cyan)),
        chunks[0],
    );

    match state.mode {
        ViewMode::Calls => render_calls(f, state, chunks[1], chunks[2]),
        ViewMode::Detections => render_detections(f, state, chunks[1], chunks[2]),
    }

    // ── footer ───────────────────────────────────────────────────────────────
    let tab_hint = match state.mode {
        ViewMode::Calls => "Tab → detections",
        ViewMode::Detections => "Tab → calls",
    };
    f.render_widget(
        Paragraph::new(format!("  q quit   ↑/k ↓/j scroll   {tab_hint}   ⚠ = loop detected"))
            .style(Style::default().fg(Color::DarkGray)),
        chunks[3],
    );
}

fn render_calls(f: &mut ratatui::Frame, state: &mut TuiState, table_area: ratatui::layout::Rect, detail_area: ratatui::layout::Rect) {
    let col_header = Row::new(["#", "timestamp", "model", "in", "out", "ms", "$", "classifier"])
        .style(Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD))
        .height(1);

    let rows: Vec<Row> = state.rows.iter().map(|r| {
        let flag = if r.loop_detected { "⚠" } else { " " };
        let style = if r.classifier.is_some() {
            Style::default().fg(Color::Magenta)
        } else if r.loop_detected {
            Style::default().fg(Color::Red)
        } else {
            Style::default()
        };
        let ts = r.timestamp.get(..19).unwrap_or(&r.timestamp).replace('T', " ");
        let clf = r.classifier.as_deref().unwrap_or("-");
        Row::new(vec![
            Cell::from(format!("{flag} {}", r.id)),
            Cell::from(ts),
            Cell::from(r.model.clone()),
            Cell::from(r.prompt_tokens.to_string()),
            Cell::from(r.output_tokens.to_string()),
            Cell::from(r.latency_ms.to_string()),
            Cell::from(format!("${:.4}", r.cost_usd)),
            Cell::from(clf.to_string()),
        ])
        .style(style)
    }).collect();

    let widths = [
        Constraint::Length(6),
        Constraint::Length(20),
        Constraint::Min(16),
        Constraint::Length(6),
        Constraint::Length(6),
        Constraint::Length(6),
        Constraint::Length(9),
        Constraint::Length(14),
    ];

    f.render_stateful_widget(
        Table::new(rows, widths)
            .header(col_header)
            .block(Block::default().title(" calls (newest first) [Tab to switch] ").borders(Borders::ALL))
            .row_highlight_style(Style::default().add_modifier(Modifier::REVERSED))
            .highlight_symbol("▶ "),
        table_area,
        &mut state.calls_table,
    );

    // ── detail panel ─────────────────────────────────────────────────────────
    let detail_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(detail_area);

    let selected = state.calls_table.selected().and_then(|i| state.rows.get(i));
    let (input_text, output_text, detail_title) = match selected {
        None => (
            Text::from("(select a row with ↑/k ↓/j)"),
            Text::from(""),
            " detail ".to_string(),
        ),
        Some(row) => {
            let title = format!(" call #{} ", row.id);
            let inp = if row.input_text.is_empty() {
                Text::from(Span::styled("(empty)", Style::default().fg(Color::DarkGray)))
            } else {
                build_message_text(&row.input_text)
            };
            let out = if row.output_text.is_empty() {
                Text::from(Span::styled("(empty)", Style::default().fg(Color::DarkGray)))
            } else {
                Text::raw(row.output_text.clone())
            };
            (inp, out, title)
        }
    };

    f.render_widget(
        Paragraph::new(input_text)
            .block(Block::default().title(format!("{detail_title}input")).borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Blue)))
            .wrap(Wrap { trim: false }),
        detail_chunks[0],
    );
    f.render_widget(
        Paragraph::new(output_text)
            .block(Block::default().title(format!("{detail_title}output")).borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Green)))
            .wrap(Wrap { trim: false }),
        detail_chunks[1],
    );
}

fn render_detections(f: &mut ratatui::Frame, state: &mut TuiState, table_area: ratatui::layout::Rect, detail_area: ratatui::layout::Rect) {
    let col_header = Row::new(["#", "timestamp", "classifier", "calls", "$"])
        .style(Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD))
        .height(1);

    let rows: Vec<Row> = state.detections.iter().map(|d| {
        let ts = d.timestamp.get(..19).unwrap_or(&d.timestamp).replace('T', " ");
        let style = match d.classifier.as_str() {
            "retry_storm" => Style::default().fg(Color::Red),
            "cost_inflation" => Style::default().fg(Color::Yellow),
            "self_correction" => Style::default().fg(Color::Magenta),
            _ => Style::default(),
        };
        Row::new(vec![
            Cell::from(d.id.to_string()),
            Cell::from(ts),
            Cell::from(d.classifier.clone()),
            Cell::from(d.call_ids.clone()),
            Cell::from(format!("${:.4}", d.cost_usd)),
        ])
        .style(style)
    }).collect();

    let widths = [
        Constraint::Length(5),
        Constraint::Length(20),
        Constraint::Length(16),
        Constraint::Min(12),
        Constraint::Length(9),
    ];

    f.render_stateful_widget(
        Table::new(rows, widths)
            .header(col_header)
            .block(Block::default().title(" detections (newest first) [Tab to switch] ").borders(Borders::ALL))
            .row_highlight_style(Style::default().add_modifier(Modifier::REVERSED))
            .highlight_symbol("▶ "),
        table_area,
        &mut state.detections_table,
    );

    // ── detail panel (detail + suggested_fix) ─────────────────────────────────
    let detail_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(60), Constraint::Percentage(40)])
        .split(detail_area);

    let selected = state.detections_table.selected().and_then(|i| state.detections.get(i));
    let (detail_text, fix_text, panel_title) = match selected {
        None => (
            Text::from("(select a row with ↑/k ↓/j)"),
            Text::from(""),
            " detection detail ".to_string(),
        ),
        Some(d) => {
            let title = format!(" {} #{} ", d.classifier, d.id);
            let det = Text::raw(format!("{}\n\ncall ids: {}", d.detail, d.call_ids));
            let fix = Text::raw(d.suggested_fix.clone());
            (det, fix, title)
        }
    };

    f.render_widget(
        Paragraph::new(detail_text)
            .block(Block::default().title(format!("{panel_title}detail")).borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Red)))
            .wrap(Wrap { trim: false }),
        detail_chunks[0],
    );
    f.render_widget(
        Paragraph::new(fix_text)
            .block(Block::default().title(format!("{panel_title}suggested fix")).borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Green)))
            .wrap(Wrap { trim: false }),
        detail_chunks[1],
    );
}

fn build_message_text(raw: &str) -> Text<'static> {
    let mut lines: Vec<Line<'static>> = Vec::new();
    for line in raw.lines() {
        if line.starts_with('[') && line.contains(']') {
            lines.push(Line::from(Span::styled(
                line.to_owned(),
                Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD),
            )));
        } else {
            lines.push(Line::from(line.to_owned()));
        }
    }
    Text::from(lines)
}
