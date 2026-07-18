use memory_engine::{DEFAULT_CHUNK_SIZE, MAX_CHUNK_SIZE, chunk_text, normalize_extracted_text};

#[test]
fn extraction_normalization_removes_only_nul_and_replacement_characters() {
    assert_eq!(
        normalize_extracted_text(" a\0\u{fffd}\u{1} b "),
        " a\u{1} b "
    );
}

#[test]
fn default_and_configuration_limits_match_v005() {
    assert_eq!(DEFAULT_CHUNK_SIZE, 1_075);
    assert_eq!(MAX_CHUNK_SIZE, 28_672);
    assert!(chunk_text("content", Some(-2)).is_err());
    assert!(chunk_text("content", Some(28_673)).is_err());
}

#[test]
fn plain_text_chunks_overlap_on_whole_words() {
    let text = (0..80)
        .map(|index| format!("word{index}"))
        .collect::<Vec<_>>()
        .join(" ");
    let chunks = chunk_text(&text, Some(120)).expect("chunks");
    assert!(chunks.len() > 1);
    assert!(
        chunks
            .iter()
            .all(|chunk| chunk.encode_utf16().count() <= 120)
    );
    assert!(chunks[1].contains("word"));
}

#[test]
fn markdown_headings_form_sections_without_splitting_fences() {
    let text = "# One\n\nalpha\n\n```rust\n# not a heading\n```\n\n## Two\n\nbeta";
    let chunks = chunk_text(text, Some(45)).expect("chunks");
    assert!(chunks.iter().any(|chunk| chunk.contains("# not a heading")));
    assert!(chunks.iter().any(|chunk| chunk.contains("## Two")));
}

#[test]
fn utf16_code_units_control_chunk_limits() {
    let chunks = chunk_text(&"😀".repeat(10), Some(6)).expect("chunks");
    assert_eq!(chunks, vec!["😀😀😀", "😀😀😀", "😀😀😀", "😀"]);
}
