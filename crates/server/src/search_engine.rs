//! Application search orchestration.
//!
//! HTTP adapters validate wire input and hand this module a [`SearchQuery`]. This
//! module owns evidence retrieval, calibrated fusion, diversity selection, and
//! deferred hydration.

use std::{collections::HashMap, num::NonZeroUsize, time::Instant};

use super::{
    ApiError, AppState, SearchMode, SearchResult, memory_search_result, run_search,
    search_projection,
};

const RRF_K: f64 = 60.0;

/// Validated application-level search input.
#[expect(
    clippy::struct_excessive_bools,
    reason = "the validated V4 include and aggregate switches retain the wire contract"
)]
pub(super) struct SearchQuery {
    pub text: String,
    pub mode: SearchMode,
    pub limit: NonZeroUsize,
    pub threshold: f32,
    pub options: storage::SearchOptions,
    pub include_forgotten: bool,
    pub include_documents: bool,
    pub include_summaries: bool,
    pub include_related: bool,
    /// When set, return at most one best chunk per document.
    pub aggregate: bool,
}

/// Fully ranked and budgeted search results.
pub(super) struct SearchOutcome {
    pub results: Vec<SearchResult>,
}

/// Validated semantic lookup requested from the profile endpoint.
pub(super) struct ProfileSearchQuery {
    pub organization_id: String,
    pub text: String,
    pub container_tag: String,
    pub threshold: f32,
}

/// Profile facts selected, hydrated, projected, and budgeted by the search engine.
pub(super) struct ProfileSearchOutcome {
    pub results: Vec<SearchResult>,
    pub timing: f64,
}

#[derive(Clone)]
enum RankedCandidate {
    Memory(storage::MemoryCandidate),
    Chunk(storage::ChunkCandidate),
}

impl RankedCandidate {
    fn id(&self) -> String {
        match self {
            Self::Memory(candidate) => format!("m:{}", candidate.id),
            Self::Chunk(candidate) => format!("c:{}", candidate.stable_id),
        }
    }

    fn document_id(&self) -> Option<&str> {
        match self {
            Self::Memory(_) => None,
            Self::Chunk(candidate) => Some(&candidate.document_id),
        }
    }

    fn score(&self) -> f64 {
        match self {
            Self::Memory(candidate) => candidate.semantic_score.unwrap_or_default(),
            Self::Chunk(candidate) => candidate.semantic_score.unwrap_or_default(),
        }
    }

    fn with_score(mut self, score: f64) -> Self {
        match &mut self {
            Self::Memory(candidate) => candidate.semantic_score = Some(score),
            Self::Chunk(candidate) => candidate.semantic_score = Some(score),
        }
        self
    }
}

/// The application module responsible for search orchestration.
#[derive(Clone)]
pub(super) struct SearchEngine {
    state: AppState,
}

impl SearchEngine {
    /// Creates a search engine over the application's readers and embeddings.
    pub(super) fn new(state: AppState) -> Self {
        Self { state }
    }

    /// Schedules query inference through the runtime-owned executor.
    pub(super) async fn embed_query(
        &self,
        text: String,
    ) -> Result<Option<memory_engine::EmbeddingVector>, ApiError> {
        let Some(executor) = self.state.embedding_executor.as_ref() else {
            return Ok(None);
        };
        executor
            .embed(memory_engine::EmbeddingPriority::Query, vec![text])
            .await
            .map_err(ApiError::EmbeddingExecutor)?
            .into_iter()
            .next()
            .map(Some)
            .ok_or(ApiError::EmptyEmbedding)
    }

    /// Runs profile semantic search through the runtime-owned executor. It first
    /// ranks narrow candidates, then hydrates only that selection before projecting
    /// complete facts into the bounded response context.
    pub(super) async fn search_profile(
        &self,
        query: ProfileSearchQuery,
    ) -> Result<ProfileSearchOutcome, ApiError> {
        let started = Instant::now();
        let Some(vector) = self.embed_query(query.text).await? else {
            return Ok(ProfileSearchOutcome {
                results: Vec::new(),
                timing: 0.0,
            });
        };
        let candidates = run_search(&self.state, move |storage| {
            storage
                .search_memory_candidates(
                    &query.organization_id,
                    vector.as_slice(),
                    memory_engine::BGE_MODEL_ID,
                    &query.container_tag,
                    15,
                    query.threshold,
                    storage::MemoryVisibility::default(),
                )
                .map_err(ApiError::Storage)
        })
        .await?;
        let hits = run_search(&self.state, move |storage| {
            storage
                .hydrate_memories(&candidates, storage::MemoryHydration::default())
                .map_err(ApiError::Storage)
        })
        .await?;
        let results = hits
            .into_iter()
            .map(|hit| SearchResult::Memory(memory_search_result(hit, 1.0, false, false, false)))
            .collect();
        Ok(ProfileSearchOutcome {
            results: search_projection::fit_search_context(results),
            timing: started.elapsed().as_secs_f64() * 1_000.0,
        })
    }

    /// Runs the V3 document-search pipeline: narrow ranking, explicit hydration,
    /// document grouping, thresholding, and final selection.
    pub(super) async fn search_v3(
        &self,
        organization: &str,
        request: super::V3SearchRequest,
    ) -> Result<super::V3SearchResponse, ApiError> {
        let started = Instant::now();
        let query = request.q.trim().to_owned();
        let query_vector = self.embed_query(query.clone()).await?;
        let candidate_limit = request.limit.saturating_mul(5).min(500);
        let tags = request.container_tag.map_or_else(
            || {
                request
                    .container_tags
                    .unwrap_or_else(|| vec!["sm_project_default".to_owned()])
            },
            |tag| vec![tag],
        );
        let filters = request
            .filters
            .as_ref()
            .map(super::parse_filter)
            .transpose()?;
        let options = storage::SearchOptions {
            organization_id: Some(organization.to_owned()),
            container_tags: tags,
            document_id: request.doc_id,
            filepath: request.filepath,
            filters,
        };
        let threshold = request.chunk_threshold;
        let hits = run_search(&self.state, move |storage| {
            let candidates = query_vector
                .as_ref()
                .map_or_else(
                    || storage.search_chunk_lexical_candidates(&query, candidate_limit, &options),
                    |vector| {
                        storage.search_chunk_candidates(
                            vector.as_slice(),
                            memory_engine::BGE_MODEL_ID,
                            candidate_limit,
                            threshold,
                            &options,
                        )
                    },
                )
                .map_err(ApiError::Storage)?;
            storage
                .hydrate_chunks(
                    &candidates,
                    storage::ChunkHydration {
                        full_document: request.include_full_docs,
                    },
                )
                .map_err(ApiError::Storage)
        })
        .await?;
        let results = group_v3_results(
            hits,
            request.limit,
            request.document_threshold,
            request.include_summary,
            request.include_full_docs,
        );
        let total = results.iter().map(|result| result.chunks.len()).sum();
        Ok(super::V3SearchResponse {
            results,
            timing: started.elapsed().as_secs_f64() * 1_000.0,
            total,
        })
    }

    /// Retrieves independent lexical and semantic evidence, fuses ranks with RRF,
    /// applies exact deduplication/diversity, and only then hydrates selected rows.
    #[expect(
        clippy::too_many_lines,
        reason = "the retrieval pipeline keeps its ordered stages visible together"
    )]
    pub(super) async fn search(
        &self,
        organization: &str,
        query: SearchQuery,
    ) -> Result<SearchOutcome, ApiError> {
        let started = Instant::now();
        let vector = self.embed_query(query.text.clone()).await?;
        let memory_mode = query.mode != SearchMode::Documents;
        let document_mode = query.mode != SearchMode::Memories;
        let candidate_limit = query.limit.get().saturating_mul(5).clamp(20, 500);
        let organization = organization.to_owned();
        let text = query.text.clone();
        let options = query.options.clone();
        let container_tag = options
            .container_tags
            .first()
            .cloned()
            .unwrap_or_else(|| "sm_project_default".to_owned());
        let visibility = storage::MemoryVisibility {
            include_forgotten: query.include_forgotten,
            ..storage::MemoryVisibility::default()
        };
        let threshold = query.threshold;
        let (semantic_memories, lexical_memories, semantic_chunks, lexical_chunks) =
            run_search(&self.state, move |storage| {
                let semantic_memories = if memory_mode {
                    vector
                        .as_ref()
                        .map(|vector| {
                            storage.search_memory_candidates(
                                &organization,
                                vector.as_slice(),
                                memory_engine::BGE_MODEL_ID,
                                &container_tag,
                                candidate_limit,
                                threshold,
                                visibility,
                            )
                        })
                        .transpose()
                        .map_err(ApiError::Storage)?
                        .unwrap_or_default()
                } else {
                    Vec::new()
                };
                let lexical_memories = if memory_mode {
                    storage
                        .search_memory_lexical_candidates(
                            &organization,
                            &text,
                            &container_tag,
                            candidate_limit,
                            visibility,
                        )
                        .map_err(ApiError::Storage)?
                } else {
                    Vec::new()
                };
                let semantic_chunks = if document_mode {
                    vector
                        .as_ref()
                        .map(|vector| {
                            storage.search_chunk_candidates(
                                vector.as_slice(),
                                memory_engine::BGE_MODEL_ID,
                                candidate_limit,
                                threshold,
                                &options,
                            )
                        })
                        .transpose()
                        .map_err(ApiError::Storage)?
                        .unwrap_or_default()
                } else {
                    Vec::new()
                };
                let lexical_chunks = if document_mode {
                    storage
                        .search_chunk_lexical_candidates(&text, candidate_limit, &options)
                        .map_err(ApiError::Storage)?
                } else {
                    Vec::new()
                };
                Ok((
                    semantic_memories,
                    lexical_memories,
                    semantic_chunks,
                    lexical_chunks,
                ))
            })
            .await?;

        let mut ranked = fuse_candidates(
            semantic_memories,
            lexical_memories,
            semantic_chunks,
            lexical_chunks,
        );
        ranked.sort_by(|left, right| right.score().total_cmp(&left.score()));
        let selected = select_diverse(ranked, query.limit.get(), query.aggregate);
        let memory_selection = selected
            .iter()
            .filter_map(|candidate| match candidate {
                RankedCandidate::Memory(candidate) => Some(candidate.clone()),
                RankedCandidate::Chunk(_) => None,
            })
            .collect::<Vec<_>>();
        let chunk_selection = selected
            .iter()
            .filter_map(|candidate| match candidate {
                RankedCandidate::Memory(_) => None,
                RankedCandidate::Chunk(candidate) => Some(candidate.clone()),
            })
            .collect::<Vec<_>>();
        let include_documents = query.include_documents;
        let include_related = query.include_related;
        let (memory_hits, chunk_hits) = run_search(&self.state, move |storage| {
            Ok((
                storage
                    .hydrate_memories(
                        &memory_selection,
                        storage::MemoryHydration {
                            relations: include_related,
                            documents: include_documents,
                        },
                    )
                    .map_err(ApiError::Storage)?,
                storage
                    .hydrate_chunks(&chunk_selection, storage::ChunkHydration::default())
                    .map_err(ApiError::Storage)?,
            ))
        })
        .await?;

        // Hydration happens per type; reconstruct the fused order rather than sorting
        // by a raw modality score.
        let mut results_by_id = HashMap::new();
        for hit in memory_hits {
            let fused_score = hit.similarity;
            results_by_id.insert(
                format!("m:{}", hit.record.id),
                SearchResult::Memory(memory_search_result(
                    hit,
                    fused_score,
                    query.include_documents,
                    query.include_summaries,
                    query.include_related,
                )),
            );
        }
        for hit in chunk_hits {
            results_by_id.insert(
                format!("c:{}", hit.id),
                SearchResult::Chunk(super::chunk_search_result(hit)),
            );
        }
        let results = selected
            .into_iter()
            .filter_map(|candidate| results_by_id.remove(&candidate.id()))
            .collect();
        tracing::info!(
            stage = "search",
            elapsed_ms = started.elapsed().as_millis(),
            "search completed"
        );
        Ok(SearchOutcome {
            results: search_projection::fit_search_context(results),
        })
    }
}

fn group_v3_results(
    hits: Vec<storage::SearchHit>,
    limit: usize,
    document_threshold: f32,
    include_summary: bool,
    include_full_docs: bool,
) -> Vec<super::V3DocumentResult> {
    let mut results = Vec::new();
    for hit in hits {
        let chunk = super::V3ChunkResult {
            content: hit.chunk,
            is_relevant: true,
            score: hit.score,
            position: hit.position,
        };
        if let Some(existing) = results
            .iter_mut()
            .find(|result: &&mut super::V3DocumentResult| result.document_id == hit.document_id)
        {
            existing.score = existing.score.max(hit.score);
            existing.chunks.push(chunk);
            continue;
        }
        let summary = include_summary
            .then(|| super::metadata_string(&hit.metadata, "summary"))
            .flatten();
        let content = include_full_docs.then_some(hit.document_content);
        results.push(super::V3DocumentResult {
            chunks: vec![chunk],
            created_at: hit.created_at,
            document_id: hit.document_id,
            metadata: Some(hit.metadata.clone()),
            score: hit.score,
            summary,
            content,
            title: super::metadata_string(&hit.metadata, "title"),
            updated_at: hit.updated_at,
            document_type: super::metadata_string(&hit.metadata, "type"),
        });
    }
    results.retain(|result| result.score >= f64::from(document_threshold));
    results.sort_by(|left, right| right.score.total_cmp(&left.score));
    results.truncate(limit);
    results
}

fn fuse_candidates(
    semantic_memories: Vec<storage::MemoryCandidate>,
    lexical_memories: Vec<storage::MemoryCandidate>,
    semantic_chunks: Vec<storage::ChunkCandidate>,
    lexical_chunks: Vec<storage::ChunkCandidate>,
) -> Vec<RankedCandidate> {
    let mut fused: HashMap<String, (RankedCandidate, f64)> = HashMap::new();
    for (rank, candidate) in semantic_memories.into_iter().enumerate() {
        add_evidence(&mut fused, RankedCandidate::Memory(candidate), rank + 1);
    }
    for candidate in lexical_memories {
        add_evidence(
            &mut fused,
            RankedCandidate::Memory(candidate.clone()),
            candidate.lexical_rank.unwrap_or(1),
        );
    }
    for (rank, candidate) in semantic_chunks.into_iter().enumerate() {
        add_evidence(&mut fused, RankedCandidate::Chunk(candidate), rank + 1);
    }
    for candidate in lexical_chunks {
        add_evidence(
            &mut fused,
            RankedCandidate::Chunk(candidate.clone()),
            candidate.lexical_rank.unwrap_or(1),
        );
    }
    fused
        .into_values()
        .map(|(candidate, score)| candidate.with_score(score))
        .collect()
}

fn add_evidence(
    fused: &mut HashMap<String, (RankedCandidate, f64)>,
    candidate: RankedCandidate,
    rank: usize,
) {
    let id = candidate.id();
    #[expect(
        clippy::cast_precision_loss,
        reason = "candidate ranks are bounded to 500 before RRF conversion"
    )]
    let score = 1.0 / (RRF_K + rank as f64);
    fused
        .entry(id)
        .and_modify(|(_, total)| *total += score)
        .or_insert((candidate, score));
}

fn select_diverse(
    candidates: Vec<RankedCandidate>,
    limit: usize,
    aggregate: bool,
) -> Vec<RankedCandidate> {
    let mut selected = Vec::with_capacity(limit);
    let mut documents = std::collections::HashSet::new();
    for candidate in candidates {
        if selected.len() == limit {
            break;
        }
        // This is a deterministic MMR-style diversity constraint: chunks from an
        // already represented document have maximum redundancy. Aggregate makes it
        // strict; normal search admits them only after other documents are covered.
        let duplicate_document = candidate
            .document_id()
            .is_some_and(|id| documents.contains(id));
        if aggregate && duplicate_document {
            continue;
        }
        if let Some(id) = candidate.document_id() {
            documents.insert(id.to_owned());
        }
        selected.push(candidate);
    }
    selected
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rrf_rewards_distinct_lexical_and_semantic_evidence() {
        let results = fuse_candidates(
            vec![storage::MemoryCandidate {
                id: "both".to_owned(),
                semantic_score: Some(0.9),
                lexical_rank: None,
            }],
            vec![storage::MemoryCandidate {
                id: "both".to_owned(),
                semantic_score: None,
                lexical_rank: Some(1),
            }],
            Vec::new(),
            Vec::new(),
        );
        assert_eq!(results.len(), 1);
        assert!(results[0].score() > 0.03);
    }
}
