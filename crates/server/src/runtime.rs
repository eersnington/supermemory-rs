//! Process-scoped ownership of database connections and embedding inference.

use std::sync::{Arc, Mutex};
use tokio::sync::Semaphore;

use crate::writer::StorageWriter;

const SEARCH_CONNECTIONS: usize = 5;
const DEFAULT_EMBEDDING_QUEUE_CAPACITY: usize = 64;
const DEFAULT_EMBEDDING_MAX_ITEMS: usize = 32;
const DEFAULT_EMBEDDING_MAX_PADDED_TOKENS: usize = 8_192;

/// Runtime bounds for provider and embedding work.
#[derive(Debug, Clone, Copy)]
pub struct RuntimeLimits {
    pub provider_concurrency: usize,
    pub embedding_queue_capacity: usize,
    pub embedding_max_items: usize,
    pub embedding_max_padded_tokens: usize,
}

impl Default for RuntimeLimits {
    fn default() -> Self {
        Self {
            provider_concurrency: 8,
            embedding_queue_capacity: DEFAULT_EMBEDDING_QUEUE_CAPACITY,
            embedding_max_items: DEFAULT_EMBEDDING_MAX_ITEMS,
            embedding_max_padded_tokens: DEFAULT_EMBEDDING_MAX_PADDED_TOKENS,
        }
    }
}

/// A process/runtime-scoped owner of mutable services. Construction prepares
/// immutable startup metadata and read connections before moving the only write
/// connection into `StorageWriter`.
#[derive(Clone)]
pub struct ServerRuntime {
    writer: StorageWriter,
    embeddings: Option<memory_engine::EmbeddingExecutor>,
    local_org_id: String,
    api_key_identities: Vec<storage::ApiKeyIdentity>,
    readers: StorageReaders,
}

/// Bounded read-only `SQLite` access shared by HTTP search and background context reads.
#[derive(Clone)]
pub(crate) struct StorageReaders {
    writer: StorageWriter,
    connections: Arc<Mutex<Vec<storage::Storage>>>,
    permits: Arc<Semaphore>,
}

impl StorageReaders {
    pub(crate) async fn execute<T, E>(
        &self,
        operation: impl FnOnce(&storage::Storage) -> Result<T, E> + Send + 'static,
    ) -> Result<Result<T, E>, StorageReaderError>
    where
        T: Send + 'static,
        E: Send + 'static,
    {
        let permit = Arc::clone(&self.permits)
            .acquire_owned()
            .await
            .map_err(|_| StorageReaderError::Closed)?;
        let connection = self
            .connections
            .lock()
            .map_err(|_| StorageReaderError::Unavailable)?
            .pop();
        if let Some(connection) = connection {
            let connections = Arc::clone(&self.connections);
            tokio::task::spawn_blocking(move || {
                let _permit = permit;
                let result = operation(&connection);
                connections
                    .lock()
                    .map_err(|_| StorageReaderError::Unavailable)?
                    .push(connection);
                Ok(result)
            })
            .await
            .map_err(StorageReaderError::Executor)?
        } else {
            let _permit = permit;
            self.writer
                .execute(move |storage| operation(storage))
                .await
                .map_err(|_| StorageReaderError::Closed)
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub(crate) enum StorageReaderError {
    #[error("SQLite reader pool is unavailable")]
    Unavailable,
    #[error("SQLite reader task stopped before completing the operation: {0}")]
    Executor(#[source] tokio::task::JoinError),
    #[error("SQLite reader pool has stopped")]
    Closed,
}

impl ServerRuntime {
    #[must_use]
    pub fn new(
        storage: storage::Storage,
        model: Option<Arc<memory_engine::EmbeddingModel>>,
    ) -> Self {
        Self::with_limits(storage, model, RuntimeLimits::default())
    }
    #[must_use]
    pub fn with_limits(
        storage: storage::Storage,
        model: Option<Arc<memory_engine::EmbeddingModel>>,
        limits: RuntimeLimits,
    ) -> Self {
        let local_org_id = storage.local_organization_id().to_owned();
        let api_key_identities = storage.api_key_identities().unwrap_or_else(|error| {
            tracing::error!(%error, "failed to load API key identities");
            Vec::new()
        });
        let mut readers = Vec::with_capacity(SEARCH_CONNECTIONS);
        for _ in 0..SEARCH_CONNECTIONS {
            match storage.fork() {
                Ok(reader) => readers.push(reader),
                Err(storage::StorageError::CannotForkInMemory) => break,
                Err(error) => {
                    tracing::error!(%error, "failed to open SQLite search connection");
                    break;
                }
            }
        }
        let writer = StorageWriter::start(storage, 128);
        let readers = StorageReaders {
            writer: writer.clone(),
            permits: Arc::new(Semaphore::new(readers.len().max(1))),
            connections: Arc::new(Mutex::new(readers)),
        };
        let (queue_capacity, max_items, max_padded_tokens) = embedding_limits(limits);
        tracing::info!(
            queue_capacity,
            max_items,
            max_padded_tokens,
            "embedding executor limits configured"
        );
        Self {
            writer,
            embeddings: model.map(|model| {
                memory_engine::EmbeddingExecutor::with_limits(
                    model,
                    queue_capacity,
                    max_items,
                    max_padded_tokens,
                )
            }),
            local_org_id,
            api_key_identities,
            readers,
        }
    }
    pub(crate) fn writer(&self) -> StorageWriter {
        self.writer.clone()
    }
    pub(crate) fn embeddings(&self) -> Option<memory_engine::EmbeddingExecutor> {
        self.embeddings.clone()
    }
    pub(crate) fn local_org_id(&self) -> &str {
        &self.local_org_id
    }
    pub(crate) fn api_key_identities(&self) -> &[storage::ApiKeyIdentity] {
        &self.api_key_identities
    }
    pub(crate) fn readers(&self) -> StorageReaders {
        self.readers.clone()
    }
}

fn embedding_limits(limits: RuntimeLimits) -> (usize, usize, usize) {
    (
        bounded_environment_usize(
            "SUPERMEMORY_EMBEDDING_QUEUE_CAPACITY",
            limits.embedding_queue_capacity,
            1,
            1_024,
        ),
        bounded_environment_usize(
            "SUPERMEMORY_EMBEDDING_MAX_ITEMS",
            limits.embedding_max_items,
            1,
            128,
        ),
        bounded_environment_usize(
            "SUPERMEMORY_EMBEDDING_MAX_PADDED_TOKENS",
            limits.embedding_max_padded_tokens,
            512,
            32_768,
        ),
    )
}

fn bounded_environment_usize(name: &str, default: usize, minimum: usize, maximum: usize) -> usize {
    match std::env::var(name) {
        Ok(value) => match value.parse::<usize>() {
            Ok(value) if (minimum..=maximum).contains(&value) => value,
            _ => {
                tracing::warn!(%name, %value, minimum, maximum, "ignoring invalid performance setting");
                default
            }
        },
        Err(_) => default,
    }
}
