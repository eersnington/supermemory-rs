//! Durable memory-extraction and reconciliation lifecycle.

use std::{sync::Arc, time::Duration};

use serde_json::{Map, Value};

use super::{WorkerError, health::ServiceHealth, writer::StorageWriter};

/// Coordinates claim, provider extraction, cached recovery, embedding, and atomic
/// reconciliation for one durable extraction job.
#[derive(Clone)]
pub(super) struct MemoryExtractionCoordinator {
    writer: StorageWriter,
    embeddings: memory_engine::EmbeddingExecutor,
    provider: Arc<memory_engine::MemoryProvider>,
}

impl MemoryExtractionCoordinator {
    pub(super) fn new(
        writer: StorageWriter,
        embeddings: memory_engine::EmbeddingExecutor,
        provider: Arc<memory_engine::MemoryProvider>,
    ) -> Self {
        Self {
            writer,
            embeddings,
            provider,
        }
    }

    /// Claims at most one job; the coordinator's notification loop supplies
    /// backpressure instead of speculative empty claims from many workers.
    pub(super) async fn process_available(&self) -> Result<bool, WorkerError> {
        let job = self
            .writer
            .claim_memory_job()
            .await
            .map_err(|_| WorkerError::StorageUnavailable)?
            .map_err(WorkerError::Storage)?;
        let Some(job) = job else { return Ok(false) };
        let candidates = self.extract(&job).await?;
        if candidates.is_empty() {
            return self.complete(job).await;
        }
        let values = candidates
            .iter()
            .map(|candidate| candidate.memory.clone())
            .collect();
        let vectors = match self
            .embeddings
            .embed(memory_engine::EmbeddingPriority::Memory, values)
            .await
        {
            Ok(vectors) => vectors,
            Err(error) => {
                self.retry(
                    job,
                    "embedding",
                    Some(Duration::from_secs(1)),
                    error.to_string(),
                )
                .await?;
                return Err(WorkerError::EmbeddingExecutor(error));
            }
        };
        match self.reconcile(&job, candidates, vectors).await {
            Ok(()) => Ok(true),
            Err(WorkerError::Storage(storage::StorageError::StaleRevision { .. })) => {
                self.complete(job).await
            }
            Err(error) if error.is_fatal() => Err(error),
            Err(error) => {
                self.retry(
                    job,
                    "reconciliation",
                    Some(Duration::from_secs(1)),
                    error.to_string(),
                )
                .await?;
                Err(error)
            }
        }
    }

    /// Runs notification-first with a slow durable rescan for externally queued
    /// or recovered jobs. There is intentionally no short polling interval.
    pub(super) async fn run(self, health: Arc<ServiceHealth>) {
        let notify = self.writer.memory_available();
        let mut rescan = tokio::time::interval(Duration::from_secs(5));
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
                    tracing::error!(%error, "memory extraction coordinator stopped; service degraded");
                    break;
                }
                Err(error) => {
                    tracing::warn!(%error, "memory extraction job failed");
                    tokio::select! { () = notify.notified() => {}, _ = rescan.tick() => {} }
                }
            }
        }
    }

    async fn extract(
        &self,
        job: &storage::ClaimedMemoryJob,
    ) -> Result<Vec<memory_engine::MemoryCandidate>, WorkerError> {
        if let Some(cached) = job.extraction_result.as_deref() {
            return serde_json::from_str(cached).map_err(WorkerError::CachedExtraction);
        }
        let organization = job.organization_id.clone();
        let container = job.container_tag.clone();
        let existing = self
            .writer
            .execute(move |storage| {
                storage
                    .existing_memories_for_extraction(&organization, &container)
                    .map_err(WorkerError::Storage)
            })
            .await
            .map_err(|_| WorkerError::StorageUnavailable)??;
        let context = existing
            .into_iter()
            .map(|memory| (memory.id, memory.content))
            .collect::<Vec<_>>();
        let candidates = match self
            .provider
            .extract_once(&job.content, job.document_date.as_deref(), &context)
            .await
        {
            Ok(candidates) => candidates,
            Err(error) => {
                let failure = error.failure();
                self.retry(
                    job.clone(),
                    failure_name(failure),
                    retry_delay(failure, job.attempts),
                    error.to_string(),
                )
                .await?;
                return Err(WorkerError::Provider(error));
            }
        };
        let cached = serde_json::to_string(&candidates).map_err(WorkerError::CachedExtraction)?;
        let job = job.clone();
        self.writer
            .execute(move |storage| {
                storage
                    .cache_memory_extraction(&job, &cached)
                    .map_err(WorkerError::Storage)
            })
            .await
            .map_err(|_| WorkerError::StorageUnavailable)??;
        Ok(candidates)
    }

    async fn reconcile(
        &self,
        job: &storage::ClaimedMemoryJob,
        candidates: Vec<memory_engine::MemoryCandidate>,
        vectors: Vec<memory_engine::EmbeddingVector>,
    ) -> Result<(), WorkerError> {
        let organization = job.organization_id.clone();
        let document = job.document_id.clone();
        let revision = job.revision;
        let completion = job.id.clone();
        let container = job.container_tag.clone();
        let proposals = candidates
            .into_iter()
            .zip(vectors)
            .map(|(candidate, vector)| {
                let mut metadata = Map::new();
                metadata.insert("buckets".to_owned(), serde_json::json!(candidate.buckets));
                if let Some(temporal) = candidate.temporal_context {
                    metadata.insert(
                        "temporalContext".to_owned(),
                        serde_json::to_value(temporal).unwrap_or(Value::Null),
                    );
                }
                storage::MemoryProposal {
                    temporary_id: candidate.tmp_id,
                    content: candidate.memory,
                    is_inferred: candidate.is_inferred,
                    is_static: candidate.add_to_static_profile,
                    metadata,
                    parents: candidate
                        .parent_relations
                        .into_iter()
                        .map(|parent| storage::MemoryParent {
                            memory_id: parent.memory_id,
                            relation: match parent.relation {
                                memory_engine::RelationKind::Updates => "updates",
                                memory_engine::RelationKind::Extends => "extends",
                                memory_engine::RelationKind::Derives => "derives",
                            }
                            .to_owned(),
                        })
                        .collect(),
                    forget_after: candidate.forget_after,
                    forget_reason: candidate.forget_reason,
                    vector: vector.as_slice().to_vec(),
                }
            })
            .collect::<Vec<_>>();
        self.writer
            .execute(move |storage| {
                storage
                    .reconcile_memories_for(
                        &organization,
                        &document,
                        Some(revision),
                        Some(&completion),
                        &container,
                        &proposals,
                        memory_engine::BGE_MODEL_ID,
                        memory_engine::BGE_DIMENSIONS,
                    )
                    .map(|_| ())
                    .map_err(WorkerError::Storage)
            })
            .await
            .map_err(|_| WorkerError::StorageUnavailable)??;
        Ok(())
    }

    async fn complete(&self, job: storage::ClaimedMemoryJob) -> Result<bool, WorkerError> {
        self.writer
            .execute(move |storage| {
                storage
                    .complete_memory_job(&job)
                    .map_err(WorkerError::Storage)
            })
            .await
            .map_err(|_| WorkerError::StorageUnavailable)??;
        Ok(true)
    }

    async fn retry(
        &self,
        job: storage::ClaimedMemoryJob,
        kind: &'static str,
        delay: Option<Duration>,
        message: String,
    ) -> Result<(), WorkerError> {
        self.writer
            .execute(move |storage| {
                storage
                    .retry_memory_job(&job, kind, &message, delay)
                    .map_err(WorkerError::Storage)
            })
            .await
            .map_err(|_| WorkerError::StorageUnavailable)??;
        Ok(())
    }
}

fn retry_delay(failure: memory_engine::ExtractionFailure, attempt: i64) -> Option<Duration> {
    match failure {
        memory_engine::ExtractionFailure::Authentication
        | memory_engine::ExtractionFailure::Configuration => None,
        memory_engine::ExtractionFailure::InvalidOutput if attempt >= 2 => None,
        memory_engine::ExtractionFailure::InvalidOutput => Some(Duration::from_secs(1)),
        memory_engine::ExtractionFailure::RateLimited
        | memory_engine::ExtractionFailure::Transport
        | memory_engine::ExtractionFailure::Provider => {
            Some(Duration::from_secs(1_u64 << attempt.clamp(0, 6)))
        }
    }
}
fn failure_name(failure: memory_engine::ExtractionFailure) -> &'static str {
    match failure {
        memory_engine::ExtractionFailure::RateLimited => "rate_limited",
        memory_engine::ExtractionFailure::Transport => "transport",
        memory_engine::ExtractionFailure::InvalidOutput => "invalid_output",
        memory_engine::ExtractionFailure::Authentication => "authentication",
        memory_engine::ExtractionFailure::Configuration => "configuration",
        memory_engine::ExtractionFailure::Provider => "provider",
    }
}
