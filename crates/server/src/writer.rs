//! Bounded, single-task ownership of all application `SQLite` mutations.
//!
//! `StorageWriter` is deliberately constructed from a `Storage`, not shared
//! storage. The task below is therefore the only application owner of the write
//! connection for its whole lifetime.

use std::{sync::Arc, time::Instant};

use tokio::sync::{Notify, mpsc, oneshot};

use storage::{
    ClaimedJob, ClaimedMemoryJob, DocumentState, Storage, StorageError, UpsertDocument,
    UpsertResult,
};

type WriteOperation = Box<dyn FnOnce(&mut Storage) + Send + 'static>;

enum Command {
    Run {
        operation: WriteOperation,
        submitted_at: Instant,
    },
}

/// Bounded owner for serialized `SQLite` writes and durable-work notifications.
#[derive(Clone)]
pub(crate) struct StorageWriter {
    commands: mpsc::Sender<Command>,
    document_available: Arc<Notify>,
    memory_available: Arc<Notify>,
}

impl StorageWriter {
    #[must_use]
    pub(crate) fn start(storage: Storage, capacity: usize) -> Self {
        let (commands, mut receiver) = mpsc::channel(capacity);
        let document_available = Arc::new(Notify::new());
        let memory_available = Arc::new(Notify::new());
        tokio::spawn(async move {
            // This binding owns the sole application write connection. It is not
            // wrapped in a mutex because this task is its serialization boundary.
            let mut storage = storage;
            while let Some(Command::Run {
                operation,
                submitted_at,
            }) = receiver.recv().await
            {
                let started = Instant::now();
                operation(&mut storage);
                tracing::debug!(
                    stage = "sqlite_write",
                    queue_wait_ms = started.duration_since(submitted_at).as_millis(),
                    elapsed_ms = started.elapsed().as_millis(),
                    "writer command completed"
                );
            }
        });
        Self {
            commands,
            document_available,
            memory_available,
        }
    }

    pub(crate) fn document_available(&self) -> Arc<Notify> {
        Arc::clone(&self.document_available)
    }
    pub(crate) fn memory_available(&self) -> Arc<Notify> {
        Arc::clone(&self.memory_available)
    }
    pub(crate) fn notify_documents(&self) {
        self.document_available.notify_one();
    }
    pub(crate) fn notify_memories(&self) {
        self.memory_available.notify_one();
    }

    pub(crate) async fn upsert_document(
        &self,
        organization: String,
        document: UpsertDocument,
    ) -> Result<Result<UpsertResult, StorageError>, WriterError> {
        let result = self
            .execute(move |storage| storage.upsert_document_for(&organization, document))
            .await?;
        if matches!(&result, Ok(result) if result.enqueued) {
            self.notify_documents();
        }
        Ok(result)
    }
    pub(crate) async fn forget_memory(
        &self,
        organization: String,
        id: Option<String>,
        content: Option<String>,
        container_tag: String,
        reason: Option<String>,
    ) -> Result<Result<String, StorageError>, WriterError> {
        self.execute(move |storage| {
            storage.forget_memory_for(
                &organization,
                id.as_deref(),
                content.as_deref(),
                &container_tag,
                reason.as_deref(),
            )
        })
        .await
    }
    pub(crate) async fn claim_job(
        &self,
    ) -> Result<Result<Option<ClaimedJob>, StorageError>, WriterError> {
        self.execute(Storage::claim_job).await
    }
    pub(crate) async fn mark_job_stage(
        &self,
        job: ClaimedJob,
        stage: DocumentState,
    ) -> Result<Result<(), StorageError>, WriterError> {
        self.execute(move |storage| storage.mark_job_stage(&job, stage))
            .await
    }
    pub(crate) async fn fail_job(
        &self,
        job: ClaimedJob,
        message: String,
    ) -> Result<Result<(), StorageError>, WriterError> {
        self.execute(move |storage| storage.fail_job(&job, &message))
            .await
    }
    pub(crate) async fn delete_empty_document(
        &self,
        job: ClaimedJob,
    ) -> Result<Result<(), StorageError>, WriterError> {
        self.execute(move |storage| storage.delete_empty_document(&job))
            .await
    }
    pub(crate) async fn claim_memory_job(
        &self,
    ) -> Result<Result<Option<ClaimedMemoryJob>, StorageError>, WriterError> {
        self.execute(Storage::claim_memory_job).await
    }

    pub(crate) async fn execute<T, E>(
        &self,
        operation: impl FnOnce(&mut Storage) -> Result<T, E> + Send + 'static,
    ) -> Result<Result<T, E>, WriterError>
    where
        T: Send + 'static,
        E: Send + 'static,
    {
        let (reply, response) = oneshot::channel();
        self.commands
            .send(Command::Run {
                operation: Box::new(move |storage| {
                    let _ = reply.send(operation(storage));
                }),
                submitted_at: Instant::now(),
            })
            .await
            .map_err(|_| WriterError::Closed)?;
        response.await.map_err(|_| WriterError::Closed)
    }
}

/// The writer task stopped before accepting or completing an operation.
#[derive(Debug, thiserror::Error)]
pub(crate) enum WriterError {
    #[error("SQLite writer has stopped")]
    Closed,
}
