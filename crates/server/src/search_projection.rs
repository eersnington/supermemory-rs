//! Response projection and byte budgeting for V4 search results.
//!
//! Budgeting may omit optional fields or whole results, but it never changes
//! retrieved fact text. This keeps every returned fact identical to storage.

use serde::Serialize;

use super::SearchResult;

const SEARCH_CONTEXT_BUDGET_BYTES: usize = 4_000;
// Reserve enough budget for at least one later complete result.
const MAX_RESULT_CONTEXT_BYTES: usize = SEARCH_CONTEXT_BUDGET_BYTES / 2;

impl SearchResult {
    fn reduce_optional_fields(&mut self) {
        match self {
            Self::Memory(result) => {
                result.context.parents.clear();
                result.context.children.clear();
                result.context.related.clear();
                result.documents.clear();
                result.chunks.clear();
                result.metadata = None;
            }
            Self::Chunk(result) => {
                result.context.parents.clear();
                result.context.children.clear();
                result.context.related.clear();
                result.documents.clear();
                result.chunks.clear();
                result.metadata = None;
            }
        }
    }
}

/// Admits complete results within the response-context byte budget.
///
/// Optional fields are removed before a result is rejected. Required fact text
/// is never truncated: a fact that cannot fit is omitted, allowing later facts
/// to be considered instead.
pub(super) fn fit_search_context(results: Vec<SearchResult>) -> Vec<SearchResult> {
    let mut remaining = SEARCH_CONTEXT_BUDGET_BYTES;
    results
        .into_iter()
        .enumerate()
        .filter_map(|(rank, mut result)| {
            let mut context_bytes = serialized_len(&result).unwrap_or(usize::MAX);
            if context_bytes > remaining {
                result.reduce_optional_fields();
                context_bytes = serialized_len(&result).unwrap_or(usize::MAX);
            }
            if context_bytes > remaining || context_bytes > MAX_RESULT_CONTEXT_BYTES {
                tracing::debug!(
                    rank,
                    context_bytes,
                    remaining,
                    "search result omitted by context budget"
                );
                return None;
            }
            remaining -= context_bytes;
            tracing::debug!(
                rank,
                context_bytes,
                remaining,
                "search result admitted to context budget"
            );
            Some(result)
        })
        .collect()
}

fn serialized_len(value: &impl Serialize) -> Result<usize, serde_json::Error> {
    let mut counter = ByteCounter::default();
    serde_json::to_writer(&mut counter, value)?;
    Ok(counter.0)
}

#[derive(Default)]
struct ByteCounter(usize);

impl std::io::Write for ByteCounter {
    fn write(&mut self, buffer: &[u8]) -> std::io::Result<usize> {
        self.0 += buffer.len();
        Ok(buffer.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use serde_json::Map;

    use super::{super::*, fit_search_context};

    fn memory(text: String) -> SearchResult {
        SearchResult::Memory(MemorySearchResult {
            id: "memory".to_owned(),
            memory: text,
            metadata: Some(Map::new()),
            updated_at: "2026-08-02T00:00:00Z".to_owned(),
            similarity: 0.9,
            version: 1,
            root_memory_id: None,
            context: EmptyContext {
                parents: Vec::new(),
                children: Vec::new(),
                related: Vec::new(),
            },
            documents: Vec::new(),
            chunks: Vec::new(),
        })
    }

    #[test]
    fn budgeting_never_truncates_fact_text() {
        let fact = "a".repeat(1_800);
        let results = fit_search_context(vec![memory(fact.clone())]);

        assert_eq!(results.len(), 1);
        assert!(matches!(
            &results[0],
            SearchResult::Memory(result) if result.memory == fact
        ));
    }

    #[test]
    fn budgeting_does_not_leave_a_prefix_of_a_rejected_fact() {
        let fact = "complete fact that must not become a prefix ".repeat(100);
        let results = fit_search_context(vec![memory(fact.clone()), memory("later".to_owned())]);

        assert!(results.iter().all(|result| matches!(
            result,
            SearchResult::Memory(result) if result.memory == fact || result.memory == "later"
        )));
        assert!(!results.iter().any(|result| matches!(
            result,
            SearchResult::Memory(result) if result.memory.starts_with("complete fact") && result.memory != fact
        )));
    }

    #[test]
    fn budgeting_considers_later_facts_after_an_oversized_fact() {
        let results =
            fit_search_context(vec![memory("a".repeat(2_100)), memory("kept".to_owned())]);

        assert!(matches!(
            &results[0],
            SearchResult::Memory(result) if result.memory == "kept"
        ));
    }
}
