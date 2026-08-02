//! Application configuration and startup.

pub mod credentials;
#[cfg(feature = "legacy-import")]
pub mod legacy;
pub mod model_config;

use std::{
    fs::OpenOptions,
    io::{self, IsTerminal, Write},
    net::SocketAddr,
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    time::Instant,
};

use clap::Parser;
#[cfg(feature = "legacy-import")]
use sha2::{Digest, Sha256};
use sysinfo::{Pid, ProcessRefreshKind, ProcessesToUpdate, System};
use thiserror::Error;
use tracing_subscriber::EnvFilter;
use unicode_width::UnicodeWidthStr;

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

    /// Existing local BGE model directory.
    #[arg(long, env = "SUPERMEMORY_MODEL", default_value_os_t = default_model_path())]
    pub model: PathBuf,

    /// Existing native ONNX Runtime dynamic library.
    #[arg(long, env = "SUPERMEMORY_ORT_LIBRARY", default_value_os_t = default_ort_library_path())]
    pub ort_library: PathBuf,

    /// Show live process memory usage in an interactive terminal.
    #[arg(long, env = "SUPERMEMORY_MONITOR")]
    pub monitor: bool,
}

fn default_database_path() -> PathBuf {
    std::env::var_os("HOME").map_or_else(
        || PathBuf::from("supermemory.db"),
        |home| PathBuf::from(home).join(".supermemory-rs/supermemory.db"),
    )
}

fn default_model_path() -> PathBuf {
    std::env::var_os("HOME").map_or_else(
        || PathBuf::from("models/Xenova/bge-base-en-v1.5"),
        |home| PathBuf::from(home).join(".supermemory/models/Xenova/bge-base-en-v1.5"),
    )
}

fn default_ort_library_path() -> PathBuf {
    std::env::var_os("HOME").map_or_else(
        || PathBuf::from("libonnxruntime.dylib"),
        |home| {
            PathBuf::from(home).join(".supermemory/runtime/ort-native/onnxruntime-node/bin/napi-v6/darwin/arm64/libonnxruntime.1.23.2.dylib")
        },
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
    if let Some(parent) = config.database.parent() {
        std::fs::create_dir_all(parent).map_err(|source| StartupError::CreateDataDirectory {
            path: parent.to_path_buf(),
            source,
        })?;
    }
    let data_dir = config.database.parent().unwrap_or_else(|| Path::new("."));
    let api_key = std::env::var("SUPERMEMORY_API_KEY")
        .ok()
        .filter(|key| !key.is_empty())
        .map_or_else(|| load_or_create_api_key(data_dir), Ok)?;
    let legacy_data_dir = std::env::var_os("HOME").map_or_else(
        || PathBuf::from(".supermemory"),
        |home| PathBuf::from(home).join(".supermemory"),
    );
    let mut provider_values = credentials::load_or_import(data_dir, &legacy_data_dir)?;
    let model_config = model_config::load_or_create(data_dir)?;
    let provider_models = memory_engine::ProviderModels::from(model_config.clone());
    let runtime_limits = runtime_limits(&model_config.performance);
    if provider_config(&provider_values, &provider_models).is_none()
        && io::stdin().is_terminal()
        && io::stdout().is_terminal()
    {
        prompt_for_provider(data_dir, &mut provider_values)?;
    }
    let provider_config = provider_config(&provider_values, &provider_models);
    let provider = provider_config
        .map(memory_engine::MemoryProvider::new)
        .transpose()?
        .map(std::sync::Arc::new);

    let database = config.database.clone();
    let database_started = Instant::now();
    print_step(
        "local SQLite storage",
        &config.database.display().to_string(),
    );
    let storage = tokio::task::spawn_blocking(move || storage::Storage::open(database))
        .await
        .map_err(StartupError::DatabaseExecutor)??;
    #[cfg(feature = "legacy-import")]
    let mut storage = storage;
    #[cfg(feature = "legacy-import")]
    migrate_legacy_snapshot(&mut storage, &legacy_data_dir)?;
    let organization_id = storage.local_organization_id().to_owned();
    print_success("local SQLite storage", "ready", database_started.elapsed());
    let model_path = config.model.clone();
    let ort_library = config.ort_library.clone();
    let model_started = Instant::now();
    print_step("local embeddings", &config.model.display().to_string());
    let embeddings = tokio::task::spawn_blocking(move || {
        memory_engine::EmbeddingModel::load(&model_path, &ort_library)
    })
    .await
    .map_err(StartupError::ModelExecutor)??;
    let embeddings = Arc::new(embeddings);
    print_success(
        "local embeddings",
        "BGE 768d ready",
        model_started.elapsed(),
    );
    if let Some(provider) = provider.as_ref() {
        print_success(
            "memory provider",
            provider.kind().as_str(),
            model_started.elapsed(),
        );
    }
    print_step("http server", &format!("port {}", config.bind.port()));
    let address = config.bind;
    let database = config.database.clone();
    let monitor = MemoryMonitor::new(config.monitor && io::stdout().is_terminal());
    let ready_monitor = monitor.clone();
    let result = server::serve_with_services_limits_ready(
        address,
        Some(api_key),
        storage,
        Some(embeddings),
        provider,
        runtime_limits,
        move || {
            print_success(
                "http server",
                &format!("listening on http://localhost:{}", address.port()),
                boot.elapsed(),
            );
            print_ready(address.port(), &database, &organization_id, boot.elapsed());
            ready_monitor.begin();
        },
    )
    .await;
    monitor.stop().await;
    result?;
    Ok(())
}

fn runtime_limits(performance: &model_config::Performance) -> server::RuntimeLimits {
    server::RuntimeLimits {
        provider_concurrency: environment_limit(
            "SUPERMEMORY_PROVIDER_CONCURRENCY",
            performance.provider_concurrency,
            1,
            16,
        ),
        embedding_queue_capacity: environment_limit(
            "SUPERMEMORY_EMBEDDING_QUEUE_CAPACITY",
            performance.embedding_queue_capacity,
            1,
            1_024,
        ),
        embedding_max_items: environment_limit(
            "SUPERMEMORY_EMBEDDING_MAX_ITEMS",
            performance.embedding_max_items,
            1,
            128,
        ),
        embedding_max_padded_tokens: environment_limit(
            "SUPERMEMORY_EMBEDDING_MAX_PADDED_TOKENS",
            performance.embedding_max_padded_tokens,
            512,
            32_768,
        ),
    }
}

fn environment_limit(name: &str, default: usize, minimum: usize, maximum: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .filter(|value| (minimum..=maximum).contains(value))
        .unwrap_or(default)
}

fn provider_config(
    stored: &std::collections::BTreeMap<String, String>,
    models: &memory_engine::ProviderModels,
) -> Option<memory_engine::ProviderConfig> {
    memory_engine::ProviderConfig::from_values(
        |key| std::env::var(key).ok().or_else(|| stored.get(key).cloned()),
        models,
    )
}

fn prompt_for_provider(
    data_dir: &Path,
    values: &mut std::collections::BTreeMap<String, String>,
) -> Result<(), StartupError> {
    println!("  supermemory-rs needs an LLM API key for memory extraction.");
    println!("  Pick a provider (you can add more later by setting environment variables):");
    println!("    1) OpenAI      (OPENAI_API_KEY)");
    println!("    2) Anthropic   (ANTHROPIC_API_KEY)");
    println!("    3) Gemini      (GEMINI_API_KEY)");
    println!("    4) Skip for now");

    loop {
        print!("  Choice [1-4]: ");
        io::stdout().flush().map_err(StartupError::ProviderPrompt)?;
        let mut input = String::new();
        let bytes = io::stdin()
            .read_line(&mut input)
            .map_err(StartupError::ProviderPrompt)?;
        if bytes == 0 {
            return Err(StartupError::ProviderPromptClosed);
        }
        let Some((variable, label)) = provider_selection(&input) else {
            println!("  Enter 1, 2, 3, or 4.");
            continue;
        };
        let Some(variable) = variable else {
            println!("  Skipped provider setup. Documents will not produce extracted memories.");
            return Ok(());
        };
        let key = rpassword::prompt_password(format!("  Paste your {label} API key: "))
            .map_err(StartupError::ProviderPrompt)?;
        if key.trim().is_empty() {
            println!("  The API key cannot be empty. Choose a provider and try again.");
            continue;
        }
        values.insert(variable.to_owned(), key.trim().to_owned());
        let path = data_dir.join("env.enc");
        credentials::encrypt_file(&path, data_dir, values)?;
        println!(
            "  ✓ Saved {variable} → {} (encrypted · mode 600)",
            path.display()
        );
        return Ok(());
    }
}

fn provider_selection(input: &str) -> Option<(Option<&'static str>, &'static str)> {
    match input.trim() {
        "1" => Some((Some("OPENAI_API_KEY"), "OpenAI")),
        "2" => Some((Some("ANTHROPIC_API_KEY"), "Anthropic")),
        "3" => Some((Some("GEMINI_API_KEY"), "Gemini")),
        "4" => Some((None, "")),
        _ => None,
    }
}

#[cfg(feature = "legacy-import")]
fn migrate_legacy_snapshot(
    storage: &mut storage::Storage,
    legacy_data_dir: &Path,
) -> Result<(), StartupError> {
    let source = legacy_data_dir.join("data");
    if !source.exists() {
        return Ok(());
    }
    let source_bytes =
        std::fs::read(&source).map_err(|source_error| StartupError::ReadLegacySnapshot {
            path: source.clone(),
            source: source_error,
        })?;
    let source_hash = format!("{:x}", Sha256::digest(&source_bytes));
    drop(source_bytes);
    if storage.has_legacy_import(&source_hash)? {
        return Ok(());
    }
    let output = std::env::temp_dir().join(format!(
        "supermemory-legacy-export-{}.jsonl",
        storage::generate_id()?
    ));
    legacy::export_snapshot(legacy_data_dir, &output)?;
    let import_result = storage.import_legacy_export(&output);
    let cleanup_result = std::fs::remove_file(&output);
    let report = import_result?;
    cleanup_result.map_err(|source| StartupError::RemoveLegacyExport {
        path: output,
        source,
    })?;
    storage.record_legacy_import(&source_hash, &report)?;
    Ok(())
}

fn load_or_create_api_key(data_dir: &Path) -> Result<String, StartupError> {
    let path = data_dir.join("api-key");
    match std::fs::read_to_string(&path) {
        Ok(value) if !value.trim().is_empty() => return Ok(value.trim().to_owned()),
        Ok(_) => return Err(StartupError::EmptyApiKeyFile { path }),
        Err(source) if source.kind() != io::ErrorKind::NotFound => {
            return Err(StartupError::ReadApiKey { path, source });
        }
        Err(_) => {}
    }

    let key = format!("sm_{}{}", storage::generate_id()?, storage::generate_id()?);
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    match options.open(&path) {
        Ok(mut file) => {
            writeln!(file, "{key}").map_err(|source| StartupError::WriteApiKey {
                path: path.clone(),
                source,
            })?;
            Ok(key)
        }
        Err(source) if source.kind() == io::ErrorKind::AlreadyExists => {
            std::fs::read_to_string(&path)
                .map(|value| value.trim().to_owned())
                .map_err(|source| StartupError::ReadApiKey { path, source })
        }
        Err(source) => Err(StartupError::WriteApiKey { path, source }),
    }
}

fn init_tracing() -> Result<(), StartupError> {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("warn"));
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_writer(|| MonitorAwareStderr)
        .try_init()
        .map_err(StartupError::Tracing)
}

static TERMINAL_OUTPUT_LOCK: Mutex<()> = Mutex::new(());
static MEMORY_MONITOR_ACTIVE: AtomicBool = AtomicBool::new(false);

#[derive(Clone)]
struct MemoryMonitor {
    enabled: bool,
    task: Arc<Mutex<Option<tokio::task::JoinHandle<()>>>>,
}

impl MemoryMonitor {
    fn new(enabled: bool) -> Self {
        Self {
            enabled,
            task: Arc::new(Mutex::new(None)),
        }
    }

    fn begin(&self) {
        if self.enabled
            && let Ok(mut task) = self.task.lock()
            && task.is_none()
        {
            *task = Some(tokio::spawn(monitor_memory()));
        }
    }

    async fn stop(self) {
        let task = self.task.lock().ok().and_then(|mut task| task.take());
        if let Some(task) = task {
            task.abort();
            let _ = task.await;
            clear_memory_monitor();
        }
    }
}

struct MonitorAwareStderr;

impl Write for MonitorAwareStderr {
    fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
        let _guard = TERMINAL_OUTPUT_LOCK.lock().ok();
        let mut stderr = io::stderr().lock();
        if MEMORY_MONITOR_ACTIVE.load(Ordering::Acquire) {
            stderr.write_all(b"\r\x1b[2K")?;
        }
        stderr.write(buffer)
    }

    fn flush(&mut self) -> io::Result<()> {
        io::stderr().flush()
    }
}

async fn monitor_memory() {
    let pid = Pid::from_u32(std::process::id());
    let mut system = System::new();
    let mut peak = 0;
    let mut interval = tokio::time::interval(std::time::Duration::from_secs(1));
    MEMORY_MONITOR_ACTIVE.store(true, Ordering::Release);
    loop {
        interval.tick().await;
        system.refresh_processes_specifics(
            ProcessesToUpdate::Some(&[pid]),
            false,
            ProcessRefreshKind::nothing().with_memory(),
        );
        let Some(process) = system.process(pid) else {
            continue;
        };
        let rss = process.memory();
        peak = peak.max(rss);
        if let Ok(_guard) = TERMINAL_OUTPUT_LOCK.lock() {
            let mut stdout = io::stdout().lock();
            let _ = write!(stdout, "\r\x1b[2K  {}", memory_status(rss, peak));
            let _ = stdout.flush();
        }
    }
}

fn clear_memory_monitor() {
    MEMORY_MONITOR_ACTIVE.store(false, Ordering::Release);
    if let Ok(_guard) = TERMINAL_OUTPUT_LOCK.lock() {
        let mut stdout = io::stdout().lock();
        let _ = stdout.write_all(b"\r\x1b[2K");
        let _ = stdout.flush();
    }
}

fn memory_status(rss: u64, peak: u64) -> String {
    const MIB: u64 = 1024 * 1024;
    format!(
        "memory  {} MiB RSS · {} MiB peak",
        rss.div_ceil(MIB),
        peak.div_ceil(MIB)
    )
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
    "     ██████    ██████ ",
    "     ██   ██  ██      ",
    "████ ██████    █████  ",
    "     ██   ██        ██",
    "     ██   ██  ██████  ",
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

fn ready_rows(
    port: u16,
    database: &Path,
    organization_id: &str,
    elapsed: std::time::Duration,
) -> Vec<(&'static str, String)> {
    vec![
        ("url", format!("http://localhost:{port}")),
        ("database", format!("local SQLite ({})", database.display())),
        ("search", "local BGE vectors ranked in SQLite".to_owned()),
        ("embeddings", "BGE base 768d · local q8".to_owned()),
        ("workers", "1 document · 10 memory".to_owned()),
        ("jobs", "durable · revision-guarded".to_owned()),
        ("org id", organization_id.to_owned()),
        ("boot", format_duration(elapsed)),
    ]
}

fn print_ready(port: u16, database: &Path, organization_id: &str, elapsed: std::time::Duration) {
    if !io::stdout().is_terminal() {
        return;
    }
    let rows = ready_rows(port, database, organization_id, elapsed);
    let label_width = rows
        .iter()
        .map(|(label, _)| display_width(label))
        .max()
        .unwrap_or(0);
    let line_width = rows
        .iter()
        .map(|(_, value)| label_width + display_width(value) + 2)
        .chain([19])
        .max()
        .unwrap_or(19);
    let horizontal = "─".repeat(line_width + 4);
    println!("\n\x1b[38;5;81m╭{horizontal}╮{RESET}");
    let title = "→ supermemory ready";
    println!(
        "\x1b[38;5;81m│{RESET}  \x1b[38;5;45m→{RESET} {BOLD}\x1b[38;5;45msupermemory ready{RESET}{}  \x1b[38;5;81m│{RESET}",
        " ".repeat(line_width.saturating_sub(title.chars().count()))
    );
    println!(
        "\x1b[38;5;81m│{RESET}{}\x1b[38;5;81m│{RESET}",
        " ".repeat(line_width + 4)
    );
    for (label, value) in rows {
        let visible = label_width + display_width(&value) + 2;
        println!(
            "\x1b[38;5;81m│{RESET}  {DIM}{label:>label_width$}{RESET}  {BOLD}{value}{RESET}{}  \x1b[38;5;81m│{RESET}",
            " ".repeat(line_width.saturating_sub(visible))
        );
    }
    println!("\x1b[38;5;81m╰{horizontal}╯{RESET}\n");
    println!("  {DIM}localhost requests do not need an API key.{RESET}\n");
    let _ = io::stdout().flush();
}

fn display_width(value: &str) -> usize {
    UnicodeWidthStr::width(value)
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
    #[error("local API key file {path} is empty; remove it to generate a replacement")]
    EmptyApiKeyFile { path: PathBuf },
    #[error("failed to read local API key from {path}; existing data was not modified: {source}")]
    ReadApiKey {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to securely create local API key at {path}: {source}")]
    WriteApiKey {
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
    #[error("embedding model executor stopped during startup: {0}")]
    ModelExecutor(#[source] tokio::task::JoinError),
    #[error(transparent)]
    Embedding(#[from] memory_engine::EmbeddingError),
    #[error(transparent)]
    Credentials(#[from] credentials::CredentialError),
    #[error(transparent)]
    Provider(#[from] memory_engine::ProviderError),
    #[error(transparent)]
    ModelConfig(#[from] model_config::ModelConfigError),
    #[error("failed to read provider setup input; no credentials were written: {0}")]
    ProviderPrompt(#[source] std::io::Error),
    #[error("provider setup input closed before a choice was made; no credentials were written")]
    ProviderPromptClosed,
    #[cfg(feature = "legacy-import")]
    #[error(transparent)]
    LegacyExport(#[from] legacy::LegacyExportError),
    #[cfg(feature = "legacy-import")]
    #[error("failed to read legacy snapshot {path}; source data was not changed: {source}")]
    ReadLegacySnapshot {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[cfg(feature = "legacy-import")]
    #[error("failed to remove temporary legacy export {path}: {source}")]
    RemoveLegacyExport {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error(transparent)]
    Server(#[from] server::ServerError),
}

#[cfg(test)]
mod tests {
    use std::{path::Path, time::Duration};

    use super::{display_width, memory_status, provider_selection, ready_rows};

    #[test]
    fn display_width_counts_middle_dot_as_one_column() {
        assert_eq!(display_width("BGE base 768d · local q8"), 24);
    }

    #[test]
    fn memory_status_uses_binary_units() {
        assert_eq!(
            memory_status(214 * 1024 * 1024, 445 * 1024 * 1024),
            "memory  214 MiB RSS · 445 MiB peak"
        );
    }

    #[test]
    fn ready_output_describes_current_workers_without_exposing_a_key() {
        let rows = ready_rows(
            6767,
            Path::new("/tmp/supermemory.db"),
            "org_1",
            Duration::ZERO,
        );
        assert!(rows.contains(&("search", "local BGE vectors ranked in SQLite".to_owned())));
        assert!(rows.contains(&("workers", "1 document · 10 memory".to_owned())));
        assert!(rows.iter().all(|(label, _)| *label != "api key"));
        assert!(rows.iter().all(|(_, value)| !value.contains("FTS5")));
    }

    #[test]
    fn provider_selection_maps_supported_interactive_choices() {
        assert_eq!(
            provider_selection("1\n"),
            Some((Some("OPENAI_API_KEY"), "OpenAI"))
        );
        assert_eq!(
            provider_selection("2"),
            Some((Some("ANTHROPIC_API_KEY"), "Anthropic"))
        );
        assert_eq!(
            provider_selection("3"),
            Some((Some("GEMINI_API_KEY"), "Gemini"))
        );
        assert_eq!(provider_selection("4"), Some((None, "")));
        assert_eq!(provider_selection("invalid"), None);
    }
}
