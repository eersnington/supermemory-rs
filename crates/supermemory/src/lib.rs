//! Application configuration and startup.

use std::{
    io::{self, IsTerminal, Write},
    net::SocketAddr,
    path::{Path, PathBuf},
    time::Instant,
};

use clap::Parser;
use thiserror::Error;
use tracing_subscriber::EnvFilter;

/// Non-secret command-line configuration.
#[derive(Debug, Parser)]
#[command(version, about = "Self-hosted memory service")]
pub struct Config {
    /// Address on which the HTTP server listens.
    #[arg(long, env = "SUPERMEMORY_BIND", default_value = "127.0.0.1:6767")]
    pub bind: SocketAddr,

    /// Path to the `SQLite` database.
    #[arg(long, env = "SUPERMEMORY_DATABASE", default_value_os_t = default_database_path())]
    pub database: PathBuf,
}

fn default_database_path() -> PathBuf {
    std::env::var_os("HOME").map_or_else(
        || PathBuf::from("supermemory.db"),
        |home| PathBuf::from(home).join(".supermemory-rs/supermemory.db"),
    )
}

impl Config {
    /// Parses non-secret startup settings from command-line arguments.
    ///
    /// # Errors
    ///
    /// Returns a clap error when an argument is unknown or has an invalid value.
    pub fn try_parse_from<I, T>(args: I) -> Result<Self, clap::Error>
    where
        I: IntoIterator<Item = T>,
        T: Into<std::ffi::OsString> + Clone,
    {
        <Self as Parser>::try_parse_from(args)
    }
}

/// Parses configuration and runs the service until shutdown.
///
/// # Errors
///
/// Returns an error when startup configuration, logging, storage, or the HTTP server fails.
pub async fn run() -> Result<(), StartupError> {
    start(Config::parse()).await
}

async fn start(config: Config) -> Result<(), StartupError> {
    init_tracing()?;
    let boot = Instant::now();
    print_banner();
    let api_key = std::env::var("SUPERMEMORY_API_KEY")
        .ok()
        .filter(|key| !key.is_empty());

    if let Some(parent) = config.database.parent() {
        std::fs::create_dir_all(parent).map_err(|source| StartupError::CreateDataDirectory {
            path: parent.to_path_buf(),
            source,
        })?;
    }

    let database = config.database.clone();
    let database_started = Instant::now();
    print_step(
        "local SQLite storage",
        &config.database.display().to_string(),
    );
    let storage = tokio::task::spawn_blocking(move || storage::Storage::open(database))
        .await
        .map_err(StartupError::DatabaseExecutor)??;
    let storage = std::sync::Arc::new(std::sync::Mutex::new(storage));
    print_success("local SQLite storage", "ready", database_started.elapsed());
    print_step("http server", &format!("port {}", config.bind.port()));
    let address = config.bind;
    let database = config.database.clone();
    let has_api_key = api_key.is_some();
    server::serve_with_ready(address, api_key, storage, move || {
        print_success(
            "http server",
            &format!("listening on http://localhost:{}", address.port()),
            boot.elapsed(),
        );
        print_ready(address.port(), &database, has_api_key, boot.elapsed());
    })
    .await?;
    Ok(())
}

fn init_tracing() -> Result<(), StartupError> {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("warn"));
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .try_init()
        .map_err(StartupError::Tracing)
}

const RESET: &str = "\x1b[0m";
const BOLD: &str = "\x1b[1m";
const DIM: &str = "\x1b[2m";
const LOGO: [&str; 5] = [
    " ██████  ██    ██ ██████  ███████ ██████  ███    ███ ███████ ███    ███  ██████  ██████  ██    ██",
    "██       ██    ██ ██   ██ ██      ██   ██ ████  ████ ██      ████  ████ ██    ██ ██   ██  ██  ██ ",
    " █████   ██    ██ ██████  █████   ██████  ██ ████ ██ █████   ██ ████ ██ ██    ██ ██████    ████  ",
    "      ██ ██    ██ ██      ██      ██   ██ ██  ██  ██ ██      ██  ██  ██ ██    ██ ██   ██    ██   ",
    "██████    ██████  ██      ███████ ██   ██ ██      ██ ███████ ██      ██  ██████  ██   ██    ██   ",
];
const RS_LOGO: [&str; 5] = [
    "      ██████  ███████",
    "     ██   ██  ██     ",
    "████ ██████   ███████",
    "     ██   ██       ██",
    "     ██   ██  ███████",
];

fn print_banner() {
    if !io::stdout().is_terminal() {
        return;
    }
    let colors = [45, 51, 87, 123, 159];
    let rs_colors = [208, 214, 215, 216, 223];
    println!();
    for (((line, rs), color), rs_color) in LOGO.iter().zip(RS_LOGO).zip(colors).zip(rs_colors) {
        println!("  \x1b[38;5;{color}m{line}{RESET}\x1b[38;5;{rs_color}m{rs}{RESET}");
    }
    println!("\n  {DIM}local · self-hosted · running on this machine{RESET}\n");
}

fn print_step(label: &str, detail: &str) {
    if io::stdout().is_terminal() {
        println!("  \x1b[38;5;45m◆{RESET} {BOLD}{label}{RESET}  {DIM}{detail}{RESET}");
    }
}

fn print_success(label: &str, detail: &str, elapsed: std::time::Duration) {
    if io::stdout().is_terminal() {
        println!(
            "  \x1b[38;5;120m✓{RESET} {BOLD}{label}{RESET}  {DIM}{detail} · {}{RESET}",
            format_duration(elapsed)
        );
    }
}

fn print_ready(port: u16, database: &Path, has_api_key: bool, elapsed: std::time::Duration) {
    if !io::stdout().is_terminal() {
        return;
    }
    let mut rows = vec![
        ("url", format!("http://localhost:{port}")),
        ("database", format!("local SQLite ({})", database.display())),
        ("search", "SQLite FTS5".to_owned()),
        ("workflow", "durable local worker".to_owned()),
        ("boot", format_duration(elapsed)),
    ];
    if has_api_key {
        rows.insert(4, ("auth", "loopback + bearer API key".to_owned()));
    }
    let label_width = rows.iter().map(|(label, _)| label.len()).max().unwrap_or(0);
    let content_width = rows
        .iter()
        .map(|(_, value)| label_width + value.len() + 4)
        .chain([19])
        .max()
        .unwrap_or(19);
    let horizontal = "─".repeat(content_width + 4);
    println!("\n\x1b[38;5;81m╭{horizontal}╮{RESET}");
    println!(
        "\x1b[38;5;81m│{RESET}  \x1b[38;5;45m→{RESET} {BOLD}\x1b[38;5;45msupermemory ready{RESET}{}\x1b[38;5;81m│{RESET}",
        " ".repeat(content_width.saturating_sub(18))
    );
    println!(
        "\x1b[38;5;81m│{RESET}{}\x1b[38;5;81m│{RESET}",
        " ".repeat(content_width + 4)
    );
    for (label, value) in rows {
        let row = format!("  {label:>label_width$}  {BOLD}{value}{RESET}");
        let visible = label_width + value.len() + 4;
        println!(
            "\x1b[38;5;81m│{RESET}{row}{}  \x1b[38;5;81m│{RESET}",
            " ".repeat(content_width.saturating_sub(visible))
        );
    }
    println!("\x1b[38;5;81m╰{horizontal}╯{RESET}\n");
    let _ = io::stdout().flush();
}

fn format_duration(duration: std::time::Duration) -> String {
    let milliseconds = duration.as_millis();
    if milliseconds < 1_000 {
        format!("{milliseconds}ms")
    } else if milliseconds < 10_000 {
        format!("{:.1}s", duration.as_secs_f64())
    } else {
        format!("{}s", duration.as_secs())
    }
}

/// Failure while initializing or running the application.
#[derive(Debug, Error)]
pub enum StartupError {
    #[error("failed to create data directory {path}: {source}")]
    CreateDataDirectory {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to initialize application logging: {0}")]
    Tracing(#[source] Box<dyn std::error::Error + Send + Sync>),
    #[error(transparent)]
    Storage(#[from] storage::StorageError),
    #[error("database executor stopped during startup: {0}")]
    DatabaseExecutor(#[source] tokio::task::JoinError),
    #[error(transparent)]
    Server(#[from] server::ServerError),
}
