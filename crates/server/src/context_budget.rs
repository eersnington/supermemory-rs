//! Pure response-size budgeting for V4 search results.

use serde::Serialize;

use super::SearchResult;

const SEARCH_CONTEXT_BUDGET_BYTES: usize = 4_000;

impl SearchResult {
    fn fit_context_budget(&mut self, remaining: &mut usize) -> Option<usize> {
        let mut size = serialized_len(self).unwrap_or(usize::MAX);
        if size > *remaining {
            match self {
                Self::Memory(result) => {
                    result.context.parents.clear();
                    result.context.children.clear();
                    result.context.related.clear();
                    result.documents.clear();
                    result.chunks.clear();
                }
                Self::Chunk(result) => {
                    result.context.parents.clear();
                    result.context.children.clear();
                    result.context.related.clear();
                    result.documents.clear();
                    result.chunks.clear();
                }
            }
            size = serialized_len(self).unwrap_or(usize::MAX);
        }
        if size > *remaining {
            match self {
                Self::Memory(result) => result.metadata = None,
                Self::Chunk(result) => result.metadata = None,
            }
            size = serialized_len(self).unwrap_or(usize::MAX);
        }
        if size > *remaining {
            let text = match self {
                Self::Memory(result) => &mut result.memory,
                Self::Chunk(result) => &mut result.chunk,
            };
            let overhead = size.saturating_sub(json_string_len(text));
            truncate_json_string(text, remaining.saturating_sub(overhead));
            size = serialized_len(self).unwrap_or(usize::MAX);
        }
        if size > *remaining {
            return None;
        }
        *remaining -= size;
        Some(size)
    }
}

pub(super) fn fit_search_context(results: Vec<SearchResult>) -> Vec<SearchResult> {
    let mut remaining = SEARCH_CONTEXT_BUDGET_BYTES;
    results
        .into_iter()
        .enumerate()
        .filter_map(|(rank, mut result)| {
            let context_bytes = result.fit_context_budget(&mut remaining)?;
            tracing::debug!(rank, context_bytes, "search result context size");
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

fn json_string_len(value: &str) -> usize {
    2 + value.chars().map(json_char_len).sum::<usize>()
}

fn json_char_len(character: char) -> usize {
    match character {
        '"' | '\\' | '\u{0008}' | '\t' | '\n' | '\u{000C}' | '\r' => 2,
        '\u{0000}'..='\u{001F}' => 6,
        character => character.len_utf8(),
    }
}

fn truncate_json_string(value: &mut String, max_serialized_bytes: usize) {
    let mut serialized_bytes = 2;
    for (index, character) in value.char_indices() {
        serialized_bytes += json_char_len(character);
        if serialized_bytes > max_serialized_bytes {
            value.truncate(index);
            return;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{json_string_len, truncate_json_string};

    #[test]
    fn truncation_accounts_for_json_escaping() {
        let mut value = "\"\n".repeat(1_000);
        truncate_json_string(&mut value, 20);
        assert!(json_string_len(&value) <= 20);
        assert!(!value.is_empty());
    }
}
