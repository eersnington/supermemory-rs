//! Durable memory-extraction and reconciliation lifecycle.

use std::{sync::Arc, time::Duration};

use serde_json::{Map, Value};
use tokio::{task::JoinSet, time::interval};

use super::{WorkerError, health::ServiceHealth, runtime::StorageReaders, writer::StorageWriter};

const DURABILITY_RESCAN: Duration = Duration::from_secs(5);

/// Coordinates claim, provider extraction, cached recovery, embedding, and atomic
/// reconciliation for one durable extraction job.
#[derive(Clone)]
pub(super) struct MemoryExtractionCoordinator {
    writer: StorageWriter,
    readers: StorageReaders,
    embeddings: memory_engine::EmbeddingExecutor,
    provider: Arc<memory_engine::MemoryProvider>,
    maximum_concurrency: usize,
}

impl MemoryExtractionCoordinator {
    pub(super) fn new(
        writer: StorageWriter,
        readers: StorageReaders,
        embeddings: memory_engine::EmbeddingExecutor,
        provider: Arc<memory_engine::MemoryProvider>,
        maximum_concurrency: usize,
    ) -> Self {
        Self {
            writer,
            readers,
            embeddings,
            provider,
            maximum_concurrency,
        }
    }

    async fn process(&self, job: storage::ClaimedMemoryJob) -> Result<(), WorkerError> {
        let candidates = self.extract(&job).await?;
        if candidates.is_empty() {
            self.complete(job).await?;
            return Ok(());
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
            Ok(()) => Ok(()),
            Err(WorkerError::Storage(storage::StorageError::StaleRevision { .. })) => {
                self.complete(job).await.map(|_| ())
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
        let mut rescan = interval(DURABILITY_RESCAN);
        let mut jobs = JoinSet::new();
        let maximum_concurrency = self.maximum_concurrency;
        let mut concurrency = maximum_concurrency;
        tracing::info!(
            maximum_concurrency,
            "memory extraction provider concurrency configured"
        );
        loop {
            if health.is_degraded() {
                break;
            }
            while jobs.len() < concurrency {
                let job = match self
                    .writer
                    .claim_memory_job()
                    .await
                    .map_err(|_| WorkerError::StorageUnavailable)
                    .and_then(|result| result.map_err(WorkerError::Storage))
                {
                    Ok(Some(job)) => job,
                    Ok(None) => break,
                    Err(error) if error.is_fatal() => {
                        health.degrade();
                        tracing::error!(%error, "memory extraction coordinator stopped; service degraded");
                        return;
                    }
                    Err(error) => {
                        tracing::warn!(%error, "could not claim memory extraction job");
                        break;
                    }
                };
                let coordinator = self.clone();
                jobs.spawn(async move { coordinator.process(job).await });
            }

            if jobs.is_empty() {
                tokio::select! {
                    () = notify.notified() => {},
                    _ = rescan.tick() => {
                        // Restore capacity gradually after transient provider throttling.
                        concurrency = (concurrency + 1).min(maximum_concurrency);
                    }
                }
                continue;
            }

            tokio::select! {
                result = jobs.join_next() => {
                    let Some(result) = result else { continue };
                    match result {
                        Ok(Ok(())) => {},
                        Ok(Err(error)) if error.is_fatal() => {
                            health.degrade();
                            tracing::error!(%error, "memory extraction coordinator stopped; service degraded");
                            return;
                        }
                        Ok(Err(error)) => {
                            if matches!(&error, WorkerError::Provider(provider) if provider.failure() == memory_engine::ExtractionFailure::RateLimited) {
                                concurrency = (concurrency / 2).max(1);
                                tracing::warn!(%error, concurrency, "provider rate limited extraction; reducing concurrency");
                            } else {
                                tracing::warn!(%error, "memory extraction job failed");
                            }
                        }
                        Err(error) => {
                            tracing::error!(%error, "memory extraction task stopped unexpectedly");
                            health.degrade();
                            return;
                        }
                    }
                }
                () = notify.notified() => {},
                _ = rescan.tick() => {
                    concurrency = (concurrency + 1).min(maximum_concurrency);
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
            .readers
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
        let provider_started = std::time::Instant::now();
        let outcome = match self
            .provider
            .extract_once_with_usage(&job.content, job.document_date.as_deref(), &context)
            .await
        {
            Ok(outcome) => outcome,
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
        tracing::info!(
            document_id = %job.document_id,
            context_memories = context.len(),
            candidates = outcome.memories.len(),
            input_tokens = outcome.usage.input_tokens,
            output_tokens = outcome.usage.output_tokens,
            provider_ms = provider_started.elapsed().as_millis(),
            "memory extraction completed"
        );
        let candidates = outcome.memories;
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
