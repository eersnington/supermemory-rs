//! Durable, bounded multi-document indexing lifecycle.

use std::{sync::Arc, time::Duration};

use tokio::{task::JoinSet, time::interval};

use super::{WorkerError, health::ServiceHealth, writer::StorageWriter};

const PREPARATION_CONCURRENCY: usize = 4;
const DURABILITY_RESCAN: Duration = Duration::from_secs(5);

/// Owns document claim/preparation/inference/publication scheduling. The writer
/// remains the single serialized publication boundary.
#[derive(Clone)]
pub(super) struct IndexingCoordinator {
    writer: StorageWriter,
    embeddings: Option<memory_engine::EmbeddingExecutor>,
    provider: Option<Arc<memory_engine::MemoryProvider>>,
}

impl IndexingCoordinator {
    pub(super) fn new(
        writer: StorageWriter,
        embeddings: Option<memory_engine::EmbeddingExecutor>,
        provider: Option<Arc<memory_engine::MemoryProvider>>,
    ) -> Self {
        Self {
            writer,
            embeddings,
            provider,
        }
    }

    /// Processes a bounded cohort. Preparation and inference for cohort members
    /// overlap, while every durable transition and publication remains serialized.
    pub(super) async fn process_available(&self) -> Result<bool, WorkerError> {
        let mut jobs = Vec::with_capacity(PREPARATION_CONCURRENCY);
        for _ in 0..PREPARATION_CONCURRENCY {
            let job = self
                .writer
                .claim_job()
                .await
                .map_err(|_| WorkerError::StorageUnavailable)?
                .map_err(WorkerError::Storage)?;
            let Some(job) = job else { break };
            jobs.push(job);
        }
        if jobs.is_empty() {
            return Ok(false);
        }
        let mut work = JoinSet::new();
        for job in jobs {
            let coordinator = self.clone();
            work.spawn(async move { coordinator.process(job).await });
        }
        while let Some(result) = work.join_next().await {
            result.map_err(WorkerError::Executor)??;
        }
        Ok(true)
    }

    /// Notification-first lifetime with a slow rescan for notifications lost
    /// across crashes or an external process writing the durable queue.
    pub(super) async fn run(self, health: Arc<ServiceHealth>) {
        let notify = self.writer.document_available();
        let mut rescan = interval(DURABILITY_RESCAN);
        loop {
            if health.is_degraded() {
                break;
            }
            match self.process_available().await {
                Ok(true) => {}
                Ok(false) => {
                    tokio::select! { () = notify.notified() => {}, _ = rescan.tick() => {} }
                }
                Err(error) if error.is_fatal() => {
                    health.degrade();
                    tracing::error!(%error, "indexing coordinator stopped; service degraded");
                    break;
                }
                Err(error) => {
                    tracing::warn!(%error, "indexing cohort failed");
                    tokio::select! { () = notify.notified() => {}, _ = rescan.tick() => {} }
                }
            }
        }
    }

    async fn process(&self, job: storage::ClaimedJob) -> Result<(), WorkerError> {
        self.stage(job.clone(), storage::DocumentState::Chunking)
            .await?;
        let content = job.content.clone();
        let chunks = tokio::task::spawn_blocking(move || memory_engine::chunk_text(&content, None))
            .await
            .map_err(WorkerError::Executor)?;
        let chunks = match chunks {
            Ok(chunks) => chunks,
            Err(error) => {
                self.fail(job, error.to_string()).await?;
                return Err(WorkerError::Chunking(error));
            }
        };
        if chunks.is_empty() {
            self.writer
                .delete_empty_document(job)
                .await
                .map_err(|_| WorkerError::StorageUnavailable)?
                .map_err(WorkerError::Storage)?;
            return Ok(());
        }
        let Some(embeddings) = &self.embeddings else {
            self.writer
                .execute(move |storage| {
                    storage
                        .complete_job(&job, &chunks)
                        .map_err(WorkerError::Storage)
                })
                .await
                .map_err(|_| WorkerError::StorageUnavailable)??;
            return Ok(());
        };
        self.stage(job.clone(), storage::DocumentState::Embedding)
            .await?;
        let embedded = match embeddings
            .embed_owned(memory_engine::EmbeddingPriority::Document, chunks)
            .await
        {
            Ok(vectors) => vectors,
            Err(error) => {
                self.fail(job, error.to_string()).await?;
                return Err(WorkerError::EmbeddingExecutor(error));
            }
        };
        self.stage(job.clone(), storage::DocumentState::Indexing)
            .await?;
        let extraction = self.provider.is_some();
        self.writer
            .execute(move |storage| {
                let embedded = embedded
                    .iter()
                    .map(|chunk| storage::EmbeddedChunk {
                        content: &chunk.text,
                        vector: chunk.vector.as_slice(),
                    })
                    .collect::<Vec<_>>();
                if extraction {
                    storage.complete_embedded_job_with_memory_extraction(
                        &job,
                        &embedded,
                        memory_engine::BGE_MODEL_ID,
                        memory_engine::BGE_DIMENSIONS,
                    )
                } else {
                    storage.complete_embedded_job(
                        &job,
                        &embedded,
                        memory_engine::BGE_MODEL_ID,
                        memory_engine::BGE_DIMENSIONS,
                    )
                }
                .map_err(WorkerError::Storage)
            })
            .await
            .map_err(|_| WorkerError::StorageUnavailable)??;
        if extraction {
            self.writer.notify_memories();
        }
        Ok(())
    }
    async fn stage(
        &self,
        job: storage::ClaimedJob,
        stage: storage::DocumentState,
    ) -> Result<(), WorkerError> {
        self.writer
            .mark_job_stage(job, stage)
            .await
            .map_err(|_| WorkerError::StorageUnavailable)?
            .map_err(WorkerError::Storage)
    }
    async fn fail(&self, job: storage::ClaimedJob, message: String) -> Result<(), WorkerError> {
        self.writer
            .fail_job(job, message)
            .await
            .map_err(|_| WorkerError::StorageUnavailable)?
            .map_err(WorkerError::Storage)
    }
}
