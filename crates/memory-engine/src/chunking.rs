//! v0.0.5-compatible text normalization and chunking.

use thiserror::Error;

/// Default chunk limit used by v0.0.5, measured in UTF-16 code units.
pub const DEFAULT_CHUNK_SIZE: usize = 1_075;
/// Largest accepted custom chunk limit, measured in UTF-16 code units.
pub const MAX_CHUNK_SIZE: usize = 28_672;
const OVERLAP: usize = 200;
const SHORT_CHUNK: usize = 322;

/// Removes code points discarded by the extraction-to-chunking boundary.
#[must_use]
pub fn normalize_extracted_text(text: &str) -> String {
    text.chars()
        .filter(|character| !matches!(*character, '\0' | '\u{fffd}'))
        .collect()
}

/// Splits extracted text with the recovered v0.0.5 defaults.
///
/// # Errors
/// Returns an error when the requested size is outside the accepted range.
pub fn chunk_text(text: &str, requested_size: Option<isize>) -> Result<Vec<String>, ChunkingError> {
    let limit = match requested_size {
        None | Some(-1 | 0) => DEFAULT_CHUNK_SIZE,
        Some(size) if size < -1 => return Err(ChunkingError::TooSmall),
        Some(size) => {
            let size = usize::try_from(size).map_err(|_| ChunkingError::TooSmall)?;
            if size > MAX_CHUNK_SIZE {
                return Err(ChunkingError::TooLarge);
            }
            size
        }
    };
    let normalized = normalize_extracted_text(text);
    if normalized.trim().is_empty() {
        return Ok(Vec::new());
    }
    if looks_like_markdown(&normalized) {
        Ok(chunk_markdown(&normalized, limit))
    } else {
        Ok(chunk_plain(&normalized, limit))
    }
}

fn looks_like_markdown(text: &str) -> bool {
    let mut score = 0;
    let mut fenced = false;
    for line in text.lines().take(100) {
        if line.starts_with("```") {
            if !fenced {
                score += 2;
            }
            fenced = !fenced;
            continue;
        }
        if fenced {
            continue;
        }
        if is_heading(line) {
            score += 2;
        }
        if is_unordered_item(line) || is_ordered_item(line) {
            score += 1;
        }
        if line.contains("](") && line.contains('[') {
            score += 1;
        }
        if line.starts_with("> ") || (line.starts_with('|') && line.ends_with('|')) {
            score += 1;
        }
    }
    score >= 3
}

fn is_heading(line: &str) -> bool {
    let hashes = line.bytes().take_while(|byte| *byte == b'#').count();
    (1..=6).contains(&hashes) && line.as_bytes().get(hashes) == Some(&b' ')
}

fn is_unordered_item(line: &str) -> bool {
    matches!(line.as_bytes(), [b'-' | b'*' | b'+', b' ', ..])
}

fn is_ordered_item(line: &str) -> bool {
    let digits = line.bytes().take_while(u8::is_ascii_digit).count();
    digits > 0 && line.as_bytes().get(digits..digits + 2) == Some(b". ")
}

fn chunk_markdown(text: &str, limit: usize) -> Vec<String> {
    let sections = markdown_sections(text);
    if sections.is_empty() {
        return vec![text.to_owned()];
    }
    let mut chunks = Vec::new();
    let mut pending = String::new();
    for section in sections {
        if utf16_len(&section) > limit {
            push_trimmed(&mut chunks, std::mem::take(&mut pending));
            let heading = section.lines().next().filter(|line| is_heading(line));
            let mut split = chunk_plain(&section, limit);
            if let Some(heading) = heading {
                for chunk in split.iter_mut().skip(1) {
                    let candidate = format!("{heading}\n\n{chunk}");
                    if utf16_len(&candidate) <= limit && !chunk.starts_with(heading) {
                        *chunk = candidate;
                    }
                }
            }
            chunks.extend(split);
            continue;
        }
        let added = if pending.is_empty() { 0 } else { 2 } + utf16_len(&section);
        if !pending.is_empty() && utf16_len(&pending) + added > limit {
            push_trimmed(&mut chunks, std::mem::take(&mut pending));
        }
        if !pending.is_empty() {
            pending.push_str("\n\n");
        }
        pending.push_str(&section);
    }
    push_trimmed(&mut chunks, pending);
    chunks
}

fn markdown_sections(text: &str) -> Vec<String> {
    let mut sections = Vec::new();
    let mut current = Vec::new();
    let mut fenced = false;
    for line in text.split('\n') {
        if line.starts_with("```") {
            fenced = !fenced;
            current.push(line);
        } else if !fenced && is_heading(line) {
            if !current.is_empty() {
                push_trimmed(&mut sections, current.join("\n"));
            }
            current = vec![line];
        } else {
            current.push(line);
        }
    }
    if !current.is_empty() {
        push_trimmed(&mut sections, current.join("\n"));
    }
    sections
}

fn chunk_plain(text: &str, limit: usize) -> Vec<String> {
    let sentences = split_sentences(text);
    chunk_units(&sentences, " ", limit)
}

fn split_sentences(text: &str) -> Vec<String> {
    let mut result = Vec::new();
    let mut start = 0;
    let mut chars = text.char_indices().peekable();
    while let Some((index, character)) = chars.next() {
        if matches!(character, '.' | '!' | '?')
            && chars.peek().is_none_or(|(_, next)| next.is_whitespace())
        {
            let end = index + character.len_utf8();
            push_trimmed(&mut result, text[start..end].to_owned());
            start = chars.peek().map_or(text.len(), |(next, _)| *next);
        }
    }
    if start < text.len() {
        push_trimmed(&mut result, text[start..].to_owned());
    }
    result
}

fn chunk_units(units: &[String], separator: &str, limit: usize) -> Vec<String> {
    let mut chunks = Vec::new();
    let mut pending: Vec<&str> = Vec::new();
    for unit in units
        .iter()
        .map(String::as_str)
        .filter(|unit| !unit.trim().is_empty())
    {
        if utf16_len(unit) > limit {
            flush_units(&mut chunks, &mut pending, separator, limit);
            let words: Vec<String> = unit.split_whitespace().map(str::to_owned).collect();
            if words.len() <= 1 {
                chunks.extend(fixed_width(unit, limit));
            } else {
                chunks.extend(chunk_units(&words, " ", limit));
            }
            continue;
        }
        let candidate_len = joined_len(&pending, separator)
            + usize::from(!pending.is_empty()) * utf16_len(separator)
            + utf16_len(unit);
        if candidate_len > limit {
            let overlap = pending.clone();
            flush_units(&mut chunks, &mut pending, separator, limit);
            pending = overlap;
            retain_overlap(&mut pending, separator, limit);
            if joined_len(&pending, separator)
                + usize::from(!pending.is_empty()) * utf16_len(separator)
                + utf16_len(unit)
                > limit
            {
                pending.clear();
            }
        }
        pending.push(unit);
    }
    flush_units(&mut chunks, &mut pending, separator, limit);
    chunks
}

fn flush_units(chunks: &mut Vec<String>, pending: &mut Vec<&str>, separator: &str, limit: usize) {
    if pending.is_empty() {
        return;
    }
    let chunk = pending.join(separator).trim().to_owned();
    pending.clear();
    if utf16_len(&chunk) < SHORT_CHUNK {
        if let Some(previous) = chunks.last_mut() {
            if utf16_len(previous) + utf16_len(separator) + utf16_len(&chunk) <= limit {
                previous.push_str(separator);
                previous.push_str(&chunk);
                return;
            }
        }
    }
    if !chunk.is_empty() {
        chunks.push(chunk);
    }
}

fn retain_overlap(pending: &mut Vec<&str>, separator: &str, limit: usize) {
    let budget = OVERLAP.min(limit / 2);
    while joined_len(pending, separator) > budget {
        pending.remove(0);
    }
}

fn joined_len(units: &[&str], separator: &str) -> usize {
    units.iter().map(|unit| utf16_len(unit)).sum::<usize>()
        + units.len().saturating_sub(1) * utf16_len(separator)
}

fn fixed_width(text: &str, limit: usize) -> Vec<String> {
    let mut chunks = Vec::new();
    let mut pending = String::new();
    for character in text.chars() {
        let width = character.len_utf16();
        if !pending.is_empty() && utf16_len(&pending) + width > limit {
            push_trimmed(&mut chunks, std::mem::take(&mut pending));
        }
        pending.push(character);
    }
    push_trimmed(&mut chunks, pending);
    chunks
}

fn push_trimmed(target: &mut Vec<String>, value: impl Into<String>) {
    let value = value.into();
    let value = value.trim();
    if !value.is_empty() {
        target.push(value.to_owned());
    }
}

fn utf16_len(text: &str) -> usize {
    text.encode_utf16().count()
}

/// Invalid chunk-size configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum ChunkingError {
    #[error("chunk size must be at least -1 characters")]
    TooSmall,
    #[error("chunk size must not exceed 28672 characters (embedding model limit is 8192 tokens)")]
    TooLarge,
}
