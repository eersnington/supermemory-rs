//! Application configuration and startup.

use std::{net::SocketAddr, path::PathBuf};

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
    #[arg(long, env = "SUPERMEMORY_DATABASE", default_value = "supermemory.db")]
    pub database: PathBuf,
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
    let api_key = std::env::var("SUPERMEMORY_API_KEY").map_err(|_| StartupError::MissingApiKey)?;
    if api_key.is_empty() {
        return Err(StartupError::MissingApiKey);
    }

    let database = config.database.clone();
    let storage = tokio::task::spawn_blocking(move || storage::Storage::open(database))
        .await
        .map_err(StartupError::DatabaseExecutor)??;
    let storage = std::sync::Arc::new(std::sync::Mutex::new(storage));
    tracing::info!(address = %config.bind, database = %config.database.display(), "starting server");
    server::serve(config.bind, api_key, storage).await?;
    Ok(())
}

fn init_tracing() -> Result<(), StartupError> {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .try_init()
        .map_err(StartupError::Tracing)
}

/// Failure while initializing or running the application.
#[derive(Debug, Error)]
pub enum StartupError {
    #[error(
        "SUPERMEMORY_API_KEY is missing or empty; set it to a private bearer key before starting the server"
    )]
    MissingApiKey,
    #[error("failed to initialize application logging: {0}")]
    Tracing(#[source] Box<dyn std::error::Error + Send + Sync>),
    #[error(transparent)]
    Storage(#[from] storage::StorageError),
    #[error("database executor stopped during startup: {0}")]
    DatabaseExecutor(#[source] tokio::task::JoinError),
    #[error(transparent)]
    Server(#[from] server::ServerError),
}
