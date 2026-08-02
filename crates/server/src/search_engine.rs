//! Application search orchestration.
//!
//! HTTP adapters construct a validated [`SearchQuery`] and project the returned
//! [`SearchOutcome`]. Retrieval, ranking, and budgeting belong here so the
//! transport layer does not own search behavior.

use std::num::NonZeroUsize;

use super::{
    ApiError, AppState, SearchMode, SearchResult, embed_query, memory_search_result, run_search,
    search_projection,
};

/// Validated application-level search input.
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
    pub aggregate: bool,
}

/// Fully ranked and budgeted search results.
pub(super) struct SearchOutcome {
    pub results: Vec<SearchResult>,
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

    /// Embeds, retrieves, ranks, and projects a validated search query.
    pub(super) async fn search(
        &self,
        organization: &str,
        query: SearchQuery,
    ) -> Result<SearchOutcome, ApiError> {
        let query_vector = embed_query(self.state.embeddings.as_ref(), query.text.clone()).await?;
        let memory_mode = query.mode != SearchMode::Documents;
        let document_mode = query.mode != SearchMode::Memories;
        let candidate_limit = query
            .limit
            .get()
            .saturating_mul(if query.aggregate { 5 } else { 3 });
        let organization = organization.to_owned();
        let options = query.options;
        let threshold = query.threshold;
        let (memory_hits, chunk_hits) = run_search(&self.state, move |storage| {
            if let Some(vector) = query_vector.as_ref() {
                let memories = if memory_mode {
                    storage
                        .search_memories_for(
                            &organization,
                            vector.as_slice(),
                            memory_engine::BGE_MODEL_ID,
                            options
                                .container_tags
                                .first()
                                .map_or("sm_project_default", String::as_str),
                            candidate_limit,
                            threshold,
                            query.include_forgotten,
                            storage::MemoryHydration {
                                relations: query.include_related,
                                documents: query.include_documents,
                            },
                        )
                        .map_err(ApiError::Storage)?
                } else {
                    Vec::new()
                };
                let chunks = if document_mode {
                    storage
                        .search_semantic(
                            vector.as_slice(),
                            memory_engine::BGE_MODEL_ID,
                            candidate_limit,
                            threshold,
                            &options,
                        )
                        .map_err(ApiError::Storage)?
                } else {
                    Vec::new()
                };
                Ok((memories, chunks))
            } else {
                let chunks = if document_mode {
                    storage
                        .search_for(&organization, &query.text, candidate_limit)
                        .map_err(ApiError::Storage)?
                } else {
                    Vec::new()
                };
                Ok((Vec::new(), chunks))
            }
        })
        .await?;

        let mut results: Vec<_> = memory_hits
            .into_iter()
            .map(|hit| {
                SearchResult::Memory(memory_search_result(
                    hit,
                    1.0,
                    query.include_documents,
                    query.include_summaries,
                    query.include_related,
                ))
            })
            .chain(
                chunk_hits
                    .into_iter()
                    .map(super::chunk_search_result)
                    .map(SearchResult::Chunk),
            )
            .collect();
        results.retain(|result| result.similarity() >= f64::from(threshold));
        results.sort_by(|left, right| right.similarity().total_cmp(&left.similarity()));
        results.truncate(query.limit.get());

        Ok(SearchOutcome {
            results: search_projection::fit_search_context(results),
        })
    }
}
