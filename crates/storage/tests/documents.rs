use std::path::PathBuf;

use rusqlite::Connection;
use serde_json::{Map, json};
use storage::{Storage, UpsertDocument, UpsertResult, content_hash, generate_id, sanitize_content};

fn path(label: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "supermemory-{label}-{}.db",
        generate_id().expect("id")
    ))
}

fn input(content: &str) -> UpsertDocument {
    UpsertDocument {
        content: content.to_owned(),
        custom_id: None,
        container_tags: vec!["b".into(), "a".into()],
        entity_context: None,
        metadata: Map::new(),
        task_type: "memory".into(),
        filepath: None,
        filter_by_metadata: Map::new(),
        dreaming: "dynamic".into(),
    }
}

fn set_states(path: &PathBuf, id: &str, document: &str, job: &str) {
    let connection = Connection::open(path).expect("fixture database");
    connection
        .execute("UPDATE documents SET status=?2 WHERE id=?1", [id, document])
        .expect("document state");
    connection
        .execute("UPDATE jobs SET status=?2 WHERE document_id=?1", [id, job])
        .expect("job state");
}

#[test]
fn ids_have_exact_base58_shape_and_are_not_repeated() {
    const BASE58: &[u8] = b"123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
    let first = generate_id().expect("random ID");
    let second = generate_id().expect("random ID");
    assert_eq!(first.len(), 22);
    assert!(first.bytes().all(|byte| BASE58.contains(&byte)));
    assert_ne!(first, second);
}

#[test]
fn sanitization_and_hash_match_expected_rules() {
    assert_eq!(sanitize_content(" \0\u{1}\t hello\n\u{b}\u{7f} "), "hello");
    assert_eq!(
        content_hash("abc"),
        "a9993e364706816aba3e25717850c26c9cd0d89d"
    );
}

#[test]
fn local_organization_is_persisted_across_reopen() {
    let path = path("organization");
    let mut first = Storage::open(&path).expect("first open");
    let mut value = input("one");
    value.custom_id = Some("persisted".into());
    let created = first.upsert_document(value).expect("create");
    drop(first);
    let reopened = Storage::open(&path).expect("second open");
    assert_eq!(
        reopened
            .find_document("persisted")
            .expect("lookup")
            .expect("document")
            .id,
        created.id
    );
    drop(reopened);
    std::fs::remove_file(path).expect("remove database");
}

#[test]
fn custom_id_requires_exact_tag_array_and_active_returns_existing() {
    let mut storage = Storage::in_memory().expect("storage");
    let mut first = input("one");
    first.custom_id = Some("same".into());
    let created = storage.upsert_document(first.clone()).expect("create");
    first.content = "two".into();
    assert_eq!(
        storage.upsert_document(first.clone()).expect("active"),
        UpsertResult {
            id: created.id.clone(),
            status: storage::DocumentState::Queued,
            enqueued: false
        }
    );
    first.container_tags.reverse();
    assert_ne!(
        storage.upsert_document(first).expect("different tags").id,
        created.id
    );
}

#[test]
fn done_branches_compare_filtered_ordered_metadata_and_merge_shallowly() {
    let path = path("done");
    let mut storage = Storage::open(&path).expect("storage");
    let mut first = input("one");
    first.custom_id = Some("custom".into());
    first.metadata.insert("a".into(), json!(1));
    let created = storage.upsert_document(first.clone()).expect("create");
    set_states(&path, &created.id, "done", "done");
    first.metadata.insert("sm_internal".into(), json!(true));
    assert_eq!(
        storage
            .upsert_document(first.clone())
            .expect("equivalent")
            .status,
        storage::DocumentState::Done
    );
    first.metadata.insert("b".into(), json!(2));
    assert!(
        !storage
            .upsert_document(first.clone())
            .expect("metadata update")
            .enqueued
    );
    first.content = "changed".into();
    let updated = storage.upsert_document(first).expect("content update");
    assert_eq!(updated.id, created.id);
    assert!(updated.enqueued);
    drop(storage);
    std::fs::remove_file(path).expect("remove database");
}

#[test]
fn failed_reuses_id_and_requeues() {
    let path = path("failed");
    let mut storage = Storage::open(&path).expect("storage");
    let mut value = input("one");
    value.custom_id = Some("custom".into());
    let created = storage.upsert_document(value.clone()).expect("create");
    set_states(&path, &created.id, "failed", "failed");
    let retried = storage.upsert_document(value).expect("retry");
    assert_eq!(retried.id, created.id);
    assert!(retried.enqueued);
    drop(storage);
    std::fs::remove_file(path).expect("remove database");
}

#[test]
fn content_duplicate_normalizes_tags_but_keeps_duplicates_significant() {
    let path = path("duplicate");
    let mut storage = Storage::open(&path).expect("storage");
    let created = storage.upsert_document(input("same")).expect("create");
    set_states(&path, &created.id, "done", "done");
    let mut reordered = input("same");
    reordered.container_tags = vec![" ".into(), "a".into(), "b".into()];
    assert_eq!(
        storage.upsert_document(reordered).expect("duplicate").id,
        created.id
    );
    let mut duplicate = input("same");
    duplicate.container_tags.push("a".into());
    assert_ne!(
        storage.upsert_document(duplicate).expect("new").id,
        created.id
    );
    drop(storage);
    std::fs::remove_file(path).expect("remove database");
}

#[test]
fn internal_id_lookup_precedes_matching_custom_id() {
    let mut storage = Storage::in_memory().expect("storage");
    let first = storage.upsert_document(input("internal")).expect("first");
    let mut second = input("custom");
    second.custom_id = Some(first.id.clone());
    second.container_tags = vec!["different".into()];
    storage.upsert_document(second).expect("second");
    assert_eq!(
        storage
            .find_document(&first.id)
            .expect("lookup")
            .expect("document")
            .content,
        "internal"
    );
}

#[test]
fn listing_chunks_and_deletion_respect_document_identity() {
    let mut storage = Storage::in_memory().expect("storage");
    let org_id = storage.local_organization_id().to_owned();
    let mut value = input("one two three");
    value.custom_id = Some("lifecycle".into());
    let created = storage.upsert_document(value).expect("create");
    let job = storage.claim_job().expect("claim").expect("queued job");
    storage
        .complete_job(&job, &["one two".into(), "three".into()])
        .expect("publish chunks");

    let documents = storage
        .list_documents_for(&org_id, 10, 0, Some("a"))
        .expect("list");
    assert_eq!(
        documents
            .iter()
            .map(|document| &document.id)
            .collect::<Vec<_>>(),
        vec![&created.id]
    );
    let chunks = storage
        .document_chunks_for(&org_id, "lifecycle")
        .expect("chunks")
        .expect("document");
    assert_eq!(
        chunks
            .iter()
            .map(|chunk| chunk.content.as_str())
            .collect::<Vec<_>>(),
        ["one two", "three"]
    );

    assert!(
        storage
            .delete_document_for(&org_id, "lifecycle")
            .expect("delete")
    );
    assert!(
        storage
            .document_chunks_for(&org_id, "lifecycle")
            .expect("lookup")
            .is_none()
    );
}
