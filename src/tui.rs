use std::collections::HashSet;
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
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Cell, Paragraph, Row, Table, TableState};
use ratatui::Terminal;

use crate::db::{CallRow, Database, DetectionRow, Stats};
use crate::loop_detector::normalized_levenshtein;

// ── constants ─────────────────────────────────────────────────────────────────

const BAR_WIDTH: usize = 18;
const TICKER_SPEED: usize = 2; // chars advanced per 500 ms tick

// ── health score ──────────────────────────────────────────────────────────────

fn score_deduction(classifier: &str) -> i32 {
    match classifier {
        "retry_storm"    => 20,
        "cost_inflation" => 15,
        "self_correction" => 10,
        "ping_pong"      => 10,
        _                => 0,
    }
}

fn compute_health(detections: &[DetectionRow]) -> (i32, char) {
    let deductions: i32 = detections.iter().map(|d| score_deduction(&d.classifier)).sum();
    let score = (100 - deductions).max(0);
    let grade = match score {
        90..=100 => 'A',
        75..=89  => 'B',
        60..=74  => 'C',
        45..=59  => 'D',
        _        => 'F',
    };
    (score, grade)
}

fn grade_color(grade: char) -> Color {
    match grade {
        'A' => Color::Green,
        'B' => Color::Cyan,
        'C' => Color::Yellow,
        'D' => Color::Magenta,
        _   => Color::Red,
    }
}

// ── radar helpers ─────────────────────────────────────────────────────────────

fn model_tier(model: &str) -> u8 {
    if model.contains("haiku") || model.contains("gpt-4o-mini") { 1 }
    else if model.contains("sonnet") || model.contains("gpt-4o") { 2 }
    else if model.contains("opus") || model.contains("gpt-4-turbo") { 3 }
    else { 0 }
}

const CORRECTION_PHRASES: &[&str] = &[
    "actually,", "wait, let me", "let me reconsider", "i made an error",
    "i was wrong", "correction:", "sorry, i", "upon reflection",
    "i need to correct", "to clarify,",
];

fn contains_correction(text: &str) -> bool {
    let lower = text.to_lowercase();
    CORRECTION_PHRASES.iter().any(|p| lower.contains(p))
}

struct RadarRow {
    label: &'static str,
    ratio: f64,
    count_label: String,
    fired: bool,
}

fn compute_radar(calls_newest_first: &[CallRow], detections: &[DetectionRow]) -> [RadarRow; 4] {
    // Reverse so consecutive-pair checks run oldest → newest.
    let calls: Vec<&CallRow> = calls_newest_first.iter().rev().collect();
    let fired: HashSet<&str> = detections.iter().map(|d| d.classifier.as_str()).collect();

    // retry_storm — loop_detected count as proxy for similar-prompt cluster size.
    let loop_count = calls.iter().filter(|c| c.loop_detected).count();
    let rs_fired = fired.contains("retry_storm");
    let rs_ratio = if rs_fired { 1.0 } else { (loop_count as f64 / 3.0).min(1.0) };

    // cost_inflation — consecutive tier-escalation pairs.
    let mut escalations = 0usize;
    for i in 0..calls.len().saturating_sub(1) {
        let ta = model_tier(&calls[i].model);
        let tb = model_tier(&calls[i + 1].model);
        if ta > 0 && tb > ta {
            escalations += 1;
        }
    }
    let ci_fired = fired.contains("cost_inflation");
    let ci_ratio = if ci_fired { 1.0 } else { (escalations as f64 / 2.0).min(1.0) };

    // self_correction — correction phrase seen in any output.
    let correction_seen = calls.iter().any(|c| contains_correction(&c.output_text));
    let sc_fired = fired.contains("self_correction");
    let sc_ratio = if sc_fired { 1.0 } else if correction_seen { 0.5 } else { 0.0 };

    // ping_pong — A-B-A triplets in output fingerprints.
    let fps: Vec<String> = calls.iter()
        .map(|c| c.output_text.chars().take(300).collect())
        .collect();
    let mut triplets = 0usize;
    for i in 2..fps.len() {
        let sim_same = normalized_levenshtein(&fps[i], &fps[i - 2]);
        let sim_diff = normalized_levenshtein(&fps[i], &fps[i - 1]);
        if sim_same > 0.80 && sim_diff < 0.40 {
            triplets += 1;
        }
    }
    let pp_fired = fired.contains("ping_pong");
    let pp_ratio = if pp_fired { 1.0 } else { (triplets as f64).min(1.0) };

    [
        RadarRow {
            label: "retry_storm",
            ratio: rs_ratio,
            count_label: if rs_fired { "FIRED".into() } else { format!("{}/3", loop_count.min(3)) },
            fired: rs_fired,
        },
        RadarRow {
            label: "cost_inflation",
            ratio: ci_ratio,
            count_label: if ci_fired { "FIRED".into() } else { format!("{}/2", escalations.min(2)) },
            fired: ci_fired,
        },
        RadarRow {
            label: "self_correction",
            ratio: sc_ratio,
            count_label: if sc_fired { "FIRED".into() } else if correction_seen { "1/2".into() } else { "0/2".into() },
            fired: sc_fired,
        },
        RadarRow {
            label: "ping_pong",
            ratio: pp_ratio,
            count_label: if pp_fired { "FIRED".into() } else { format!("{}/3", triplets.min(3)) },
            fired: pp_fired,
        },
    ]
}

fn radar_line(r: &RadarRow) -> Line<'static> {
    let bar_color = if r.fired { Color::Red } else { Color::Cyan };
    let filled = ((r.ratio * BAR_WIDTH as f64).round() as usize).min(BAR_WIDTH);
    let empty = BAR_WIDTH - filled;
    let label_style = if r.fired {
        Style::default().fg(Color::Red).add_modifier(Modifier::BOLD)
    } else {
        Style::default().fg(Color::White)
    };
    let count_style = if r.fired {
        Style::default().fg(Color::Red).add_modifier(Modifier::BOLD)
    } else {
        Style::default().fg(Color::DarkGray)
    };
    Line::from(vec![
        Span::styled(format!("{:<16} ", r.label), label_style),
        Span::styled("█".repeat(filled), Style::default().fg(bar_color)),
        Span::styled("░".repeat(empty), Style::default().fg(Color::DarkGray)),
        Span::styled(format!(" {}", r.count_label), count_style),
    ])
}

// ── terminal handle ───────────────────────────────────────────────────────────

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

// ── tui state ─────────────────────────────────────────────────────────────────

struct TuiState {
    rows: Vec<CallRow>,
    detections: Vec<DetectionRow>,
    stats: Stats,
    calls_table: TableState,
    // ticker
    seen_ids: HashSet<i64>,
    ticker_chars: Vec<char>,
    ticker_offset: usize,
}

impl TuiState {
    fn new() -> Self {
        Self {
            rows: vec![],
            detections: vec![],
            stats: Stats::default(),
            calls_table: TableState::default(),
            seen_ids: HashSet::new(),
            ticker_chars: vec![],
            ticker_offset: 0,
        }
    }

    fn refresh(&mut self, db: &Database) {
        self.rows = db.query_recent(100).unwrap_or_default();
        self.detections = db.query_recent_detections(100).unwrap_or_default();
        self.stats = db.query_stats().unwrap_or_default();

        // Clamp selection within bounds.
        if let Some(i) = self.calls_table.selected() {
            let len = self.rows.len();
            if len == 0 {
                self.calls_table.select(None);
            } else if i >= len {
                self.calls_table.select(Some(len - 1));
            }
        }

        // Enqueue newly seen detections into the ticker (oldest first).
        let mut new_items: Vec<&DetectionRow> = self.detections.iter()
            .filter(|d| !self.seen_ids.contains(&d.id))
            .collect();
        new_items.reverse();
        for d in new_items {
            self.seen_ids.insert(d.id);
            let ts = d.timestamp.get(11..19).unwrap_or("??:??:??");
            let msg = format!(
                "   ⚠  {}  ·  calls {}  ·  ${:.4}  ·  {}   ",
                d.classifier, d.call_ids, d.cost_usd, ts
            );
            self.ticker_chars.extend(msg.chars());
        }
    }

    fn tick(&mut self) {
        if !self.ticker_chars.is_empty() {
            self.ticker_offset += TICKER_SPEED;
            if self.ticker_offset >= self.ticker_chars.len() {
                self.ticker_chars.clear();
                self.ticker_offset = 0;
            }
        }
    }

    fn scroll_up(&mut self) {
        let i = self.calls_table.selected().map(|i| i.saturating_sub(1)).unwrap_or(0);
        self.calls_table.select(Some(i));
    }

    fn scroll_down(&mut self) {
        if self.rows.is_empty() { return; }
        let i = self.calls_table.selected()
            .map(|i| (i + 1).min(self.rows.len() - 1))
            .unwrap_or(0);
        self.calls_table.select(Some(i));
    }
}

// ── main loop ─────────────────────────────────────────────────────────────────

pub fn run(db: Database) -> anyhow::Result<()> {
    let mut handle = TerminalHandle::new()?;
    let mut state = TuiState::new();

    loop {
        state.refresh(&db);
        state.tick();
        handle.terminal.draw(|f| render(f, &mut state))?;

        if event::poll(Duration::from_millis(500))? {
            if let Event::Key(key) = event::read()? {
                if key.kind != KeyEventKind::Press { continue; }
                match key.code {
                    KeyCode::Char('q') | KeyCode::Esc => {
                        drop(handle);
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

// ── rendering ─────────────────────────────────────────────────────────────────

fn render(f: &mut ratatui::Frame, state: &mut TuiState) {
    let area = f.area();
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3), // header bar
            Constraint::Min(6),    // calls + radar
            Constraint::Length(1), // ticker
            Constraint::Length(1), // key hints
        ])
        .split(area);

    render_header(f, state, chunks[0]);
    render_main(f, state, chunks[1]);
    render_ticker(f, state, chunks[2]);
    render_hints(f, chunks[3]);
}

fn render_header(f: &mut ratatui::Frame, state: &TuiState, area: ratatui::layout::Rect) {
    let (score, grade) = compute_health(&state.detections);
    let gc = grade_color(grade);
    let grade_style = Style::default().fg(gc).add_modifier(Modifier::BOLD);

    let line = Line::from(vec![
        Span::styled(" ferroscope  ", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
        Span::raw("health "),
        Span::styled(format!("{score:>3}"), grade_style),
        Span::styled(format!(" {grade}  "), grade_style),
        Span::styled("│", Style::default().fg(Color::DarkGray)),
        Span::raw(format!("  cost ${:.4}  ", state.stats.total_cost_usd)),
        Span::styled("│", Style::default().fg(Color::DarkGray)),
        Span::raw(format!("  latency {:.0}ms  ", state.stats.avg_latency_ms)),
        Span::styled("│", Style::default().fg(Color::DarkGray)),
        Span::raw(format!(
            "  {} calls  {} detections",
            state.stats.total_calls, state.stats.total_detections
        )),
    ]);

    f.render_widget(
        Paragraph::new(line)
            .block(Block::default().borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Cyan))),
        area,
    );
}

fn render_main(f: &mut ratatui::Frame, state: &mut TuiState, area: ratatui::layout::Rect) {
    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(60), Constraint::Percentage(40)])
        .split(area);

    render_calls(f, state, chunks[0]);
    render_radar(f, state, chunks[1]);
}

fn render_calls(f: &mut ratatui::Frame, state: &mut TuiState, area: ratatui::layout::Rect) {
    let header = Row::new(["#", "model", "in", "out", "ms", "$", "classifier"])
        .style(Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD));

    let rows: Vec<Row> = state.rows.iter().map(|r| {
        let flag = if r.loop_detected { "⚠" } else { " " };
        let style = match (&r.classifier, r.loop_detected) {
            (Some(_), true)  => Style::default().fg(Color::Red),
            (Some(_), false) => Style::default().fg(Color::Magenta),
            (None,    true)  => Style::default().fg(Color::Red),
            (None,    false) => Style::default(),
        };
        let clf = r.classifier.as_deref().unwrap_or("-");
        Row::new(vec![
            Cell::from(format!("{flag}{}", r.id)),
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
        Constraint::Length(5),
        Constraint::Min(14),
        Constraint::Length(5),
        Constraint::Length(5),
        Constraint::Length(6),
        Constraint::Length(8),
        Constraint::Min(12),
    ];

    f.render_stateful_widget(
        Table::new(rows, widths)
            .header(header)
            .block(Block::default()
                .title(" calls (newest first) ")
                .borders(Borders::ALL))
            .row_highlight_style(Style::default().add_modifier(Modifier::REVERSED))
            .highlight_symbol("▶ "),
        area,
        &mut state.calls_table,
    );
}

fn render_radar(f: &mut ratatui::Frame, state: &TuiState, area: ratatui::layout::Rect) {
    let radar = compute_radar(&state.rows, &state.detections);

    let any_fired = radar.iter().any(|r| r.fired);
    let border_color = if any_fired { Color::Red } else { Color::Magenta };

    let mut lines: Vec<Line> = vec![Line::raw("")]; // top padding inside border
    for row in &radar {
        lines.push(radar_line(row));
        lines.push(Line::raw("")); // row spacing
    }

    f.render_widget(
        Paragraph::new(lines)
            .block(Block::default()
                .title(" classifier radar ")
                .borders(Borders::ALL)
                .border_style(Style::default().fg(border_color))),
        area,
    );
}

fn render_ticker(f: &mut ratatui::Frame, state: &TuiState, area: ratatui::layout::Rect) {
    let width = area.width as usize;
    let (text, style) = if state.ticker_chars.is_empty() {
        (
            format!("{:<width$}", "  ◉  monitoring — no detections yet"),
            Style::default().fg(Color::DarkGray),
        )
    } else {
        let start = state.ticker_offset.min(state.ticker_chars.len());
        let visible: String = state.ticker_chars[start..].iter().take(width).collect();
        (
            format!("{:<width$}", visible),
            Style::default().fg(Color::Black).bg(Color::Yellow),
        )
    };

    f.render_widget(Paragraph::new(text).style(style), area);
}

fn render_hints(f: &mut ratatui::Frame, area: ratatui::layout::Rect) {
    f.render_widget(
        Paragraph::new("  q quit   ↑/k ↓/j scroll calls")
            .style(Style::default().fg(Color::DarkGray)),
        area,
    );
}
