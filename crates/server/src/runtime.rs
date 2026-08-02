//! Process-scoped ownership of database connections and embedding inference.

use std::sync::{Arc, Mutex};

use crate::writer::StorageWriter;

const SEARCH_CONNECTIONS: usize = 5;

/// A process/runtime-scoped owner of mutable services. Construction prepares
/// immutable startup metadata and read connections before moving the only write
/// connection into `StorageWriter`.
#[derive(Clone)]
pub struct ServerRuntime {
    writer: StorageWriter,
    embeddings: Option<memory_engine::EmbeddingExecutor>,
    local_org_id: String,
    api_key_identities: Vec<storage::ApiKeyIdentity>,
    search_connections: Arc<Mutex<Vec<storage::Storage>>>,
}

impl ServerRuntime {
    #[must_use]
    pub fn new(
        storage: storage::Storage,
        model: Option<Arc<memory_engine::EmbeddingModel>>,
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
        Self {
            writer: StorageWriter::start(storage, 128),
            embeddings: model
                .map(|model| memory_engine::EmbeddingExecutor::with_limits(model, 64, 32, 8_192)),
            local_org_id,
            api_key_identities,
            search_connections: Arc::new(Mutex::new(readers)),
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
    pub(crate) fn search_connections(&self) -> Arc<Mutex<Vec<storage::Storage>>> {
        Arc::clone(&self.search_connections)
    }
}
