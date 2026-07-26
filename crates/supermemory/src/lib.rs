//! Application configuration and startup.

pub mod credentials;
pub mod model_config;

use std::{
    fs::OpenOptions,
    io::{self, IsTerminal, Write},
    net::SocketAddr,
    path::{Path, PathBuf},
    time::Instant,
};

use clap::Parser;
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

    /// Directory used for local configuration and credentials.
    #[arg(long, env = "SUPERMEMORY_DATA", default_value_os_t = default_data_path())]
    pub data: PathBuf,

    /// Existing local BGE model directory.
    #[arg(long, env = "SUPERMEMORY_MODEL", default_value_os_t = default_model_path())]
    pub model: PathBuf,

    /// Existing native ONNX Runtime dynamic library.
    #[arg(long, env = "SUPERMEMORY_ORT_LIBRARY", default_value_os_t = default_ort_library_path())]
    pub ort_library: PathBuf,
}

fn default_data_path() -> PathBuf {
    std::env::var_os("HOME").map_or_else(
        || PathBuf::from(".supermemory-rs"),
        |home| PathBuf::from(home).join(".supermemory-rs"),
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
    std::fs::create_dir_all(&config.data).map_err(|source| StartupError::CreateDataDirectory {
        path: config.data.clone(),
        source,
    })?;
    let data_dir = config.data.as_path();
    let api_key = std::env::var("SUPERMEMORY_API_KEY")
        .ok()
        .filter(|key| !key.is_empty())
        .map_or_else(|| load_or_create_api_key(data_dir), Ok)?;
    let legacy_data_dir = std::env::var_os("HOME").map_or_else(
        || PathBuf::from(".supermemory"),
        |home| PathBuf::from(home).join(".supermemory"),
    );
    let mut provider_values = credentials::load_or_import(data_dir, &legacy_data_dir)?;
    let provider_models =
        memory_engine::ProviderModels::from(model_config::load_or_create(data_dir)?);
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

    let database_path = data_dir.join("supermemory-turso.db");
    let database_started = Instant::now();
    print_step(
        "embedded Turso storage",
        &database_path.display().to_string(),
    );
    let storage = start_embedded_turso(database_path.clone()).await?;
    let organization_id = storage.local_organization_id().to_owned();
    let storage = std::sync::Arc::new(std::sync::Mutex::new(storage));
    print_success(
        "embedded Turso storage",
        "ready",
        database_started.elapsed(),
    );
    let model_path = config.model.clone();
    let ort_library = config.ort_library.clone();
    let model_started = Instant::now();
    print_step("local embeddings", &config.model.display().to_string());
    let embeddings = tokio::task::spawn_blocking(move || {
        memory_engine::EmbeddingModel::load(&model_path, &ort_library)
    })
    .await
    .map_err(StartupError::ModelExecutor)??;
    let embeddings = std::sync::Arc::new(embeddings);
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
    let displayed_database = database_path;
    let displayed_api_key = api_key.clone();
    server::serve_with_services_ready(
        address,
        Some(api_key),
        storage,
        Some(embeddings),
        provider,
        move || {
            print_success(
                "http server",
                &format!("listening on http://localhost:{}", address.port()),
                boot.elapsed(),
            );
            print_ready(
                address.port(),
                &displayed_database,
                &displayed_api_key,
                &organization_id,
                boot.elapsed(),
            );
        },
    )
    .await?;
    Ok(())
}

async fn start_embedded_turso(database: PathBuf) -> Result<storage::Storage, StartupError> {
    let listener = std::net::TcpListener::bind("127.0.0.1:0")
        .map_err(|error| StartupError::EmbeddedTurso(error.to_string()))?;
    let address = listener
        .local_addr()
        .map_err(|error| StartupError::EmbeddedTurso(error.to_string()))?;
    drop(listener);

    let database_text = database.to_string_lossy().into_owned();
    let options = turso_pg::DatabaseOpts::new()
        .with_views(true)
        .with_custom_types(true)
        .with_encryption(true)
        .with_index_method(true)
        .with_autovacuum(true)
        .with_attach(true)
        .with_generated_columns(true);
    let (_, database) = turso_pg::open_database(
        &database_text,
        None,
        turso_pg::OpenFlags::default(),
        options,
    )
    .map_err(|error| StartupError::EmbeddedTurso(error.to_string()))?;
    let connection = turso_pg::Connection::new(
        database
            .connect()
            .map_err(|error| StartupError::EmbeddedTurso(error.to_string()))?,
    );
    let server = turso_pg_server::TursoPgServer::new(
        address.to_string(),
        database_text,
        connection,
        std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0)),
    );
    std::thread::Builder::new()
        .name("supermemory-tursopg".to_owned())
        .spawn(move || {
            if let Err(error) = server.run() {
                tracing::error!(%error, "embedded Turso server stopped");
            }
        })
        .map_err(|error| StartupError::EmbeddedTurso(error.to_string()))?;

    let database_url = format!("postgresql://{address}/postgres");
    tokio::task::spawn_blocking(move || {
        let mut last_error = None;
        for _ in 0..100 {
            match storage::Storage::open(&database_url) {
                Ok(storage) => return Ok(storage),
                Err(error) => last_error = Some(error),
            }
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
        Err(last_error.expect("embedded Turso connection was attempted"))
    })
    .await
    .map_err(StartupError::DatabaseExecutor)?
    .map_err(StartupError::Storage)
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

fn print_ready(
    port: u16,
    database: &Path,
    api_key: &str,
    organization_id: &str,
    elapsed: std::time::Duration,
) {
    if !io::stdout().is_terminal() {
        return;
    }
    let rows = vec![
        ("url", format!("http://localhost:{port}")),
        (
            "database",
            format!("embedded Turso ({})", database.display()),
        ),
        (
            "search",
            "Turso HNSW candidates + exact BGE reranking".to_owned(),
        ),
        ("embeddings", "BGE base 768d · local q8".to_owned()),
        ("workflow", "revision-guarded durable worker".to_owned()),
        ("api key", api_key.to_owned()),
        ("org id", organization_id.to_owned()),
        ("boot", format_duration(elapsed)),
    ];
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
    println!(
        "  {DIM}the api key above is auto-applied for unauthenticated localhost requests.{RESET}\n"
    );
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
    #[error("failed to start embedded Turso: {0}")]
    EmbeddedTurso(String),
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
    #[error(transparent)]
    Server(#[from] server::ServerError),
}

#[cfg(test)]
mod tests {
    use super::{display_width, provider_selection};

    #[test]
    fn display_width_counts_middle_dot_as_one_column() {
        assert_eq!(display_width("BGE base 768d · local q8"), 24);
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
