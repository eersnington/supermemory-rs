use serde_json::Map;
use storage::{EmbeddedChunk, Storage, StorageError, UpsertDocument, generate_id};

fn input(content: &str) -> UpsertDocument {
    UpsertDocument {
        content: content.to_owned(),
        custom_id: None,
        container_tags: vec!["sm_project_default".into()],
        entity_context: None,
        metadata: Map::new(),
        task_type: "memory".into(),
        filepath: None,
        filter_by_metadata: Map::new(),
        dreaming: "dynamic".into(),
    }
}

#[test]
fn claimed_job_becomes_searchable_only_after_atomic_completion() {
    let mut storage = Storage::in_memory().expect("storage");
    let document = storage
        .upsert_document(input("A distinctive kingfisher observation"))
        .expect("document");
    let job = storage.claim_job().expect("claim").expect("queued job");

    assert!(storage.search("kingfisher", 10).expect("search").is_empty());

    storage
        .complete_job(&job, &[job.content.clone()])
        .expect("complete");
    let hits = storage.search("kingfisher", 10).expect("search");
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].document_id, document.id);
    assert!(storage.claim_job().expect("claim again").is_none());
}

#[test]
fn reprocessing_replaces_old_searchable_content() {
    let mut storage = Storage::in_memory().expect("storage");
    let mut value = input("first albatross content");
    value.custom_id = Some("replace-me".into());
    storage.upsert_document(value.clone()).expect("create");
    let first = storage.claim_job().expect("claim").expect("first job");
    storage
        .complete_job(&first, &[first.content.clone()])
        .expect("complete first");

    value.content = "second narwhal content".into();
    storage.upsert_document(value).expect("update");
    let second = storage.claim_job().expect("claim").expect("second job");
    storage
        .complete_job(&second, &[second.content.clone()])
        .expect("complete second");

    assert!(
        storage
            .search("albatross", 10)
            .expect("old search")
            .is_empty()
    );
    assert_eq!(storage.search("narwhal", 10).expect("new search").len(), 1);
}

#[test]
fn reopening_recovers_an_interrupted_claim() {
    let path = std::env::temp_dir().join(format!(
        "supermemory-recovery-{}.db",
        generate_id().expect("id")
    ));
    let mut storage = Storage::open(&path).expect("storage");
    storage
        .upsert_document(input("recoverable content"))
        .expect("document");
    let claimed = storage.claim_job().expect("claim").expect("queued job");
    drop(storage);

    let mut reopened = Storage::open(&path).expect("reopen");
    let recovered = reopened
        .claim_job()
        .expect("claim recovered")
        .expect("recovered job");
    assert_eq!(recovered.id, claimed.id);

    drop(reopened);
    std::fs::remove_file(path).expect("remove database");
}

#[test]
fn empty_extraction_cleanup_removes_the_document_and_job() {
    let mut storage = Storage::in_memory().expect("storage");
    let document = storage
        .upsert_document(input("\u{fffd}\0"))
        .expect("document");
    let job = storage.claim_job().expect("claim").expect("queued job");
    storage
        .delete_empty_document(&job)
        .expect("delete empty document");
    assert!(
        storage
            .find_document(&document.id)
            .expect("lookup")
            .is_none()
    );
}

#[test]
fn normalized_vectors_are_published_and_ranked_exactly() {
    let mut storage = Storage::in_memory().expect("storage");
    let first = storage
        .upsert_document(input("first semantic chunk"))
        .expect("first document");
    let first_job = storage.claim_job().expect("claim").expect("first job");
    storage
        .complete_embedded_job(
            &first_job,
            &[EmbeddedChunk {
                content: &first_job.content,
                vector: &[1.0, 0.0],
            }],
            "fixture-model",
            2,
        )
        .expect("publish first");

    let second = storage
        .upsert_document(input("second semantic chunk"))
        .expect("second document");
    let second_job = storage.claim_job().expect("claim").expect("second job");
    storage
        .complete_embedded_job(
            &second_job,
            &[EmbeddedChunk {
                content: &second_job.content,
                vector: &[0.8, 0.6],
            }],
            "fixture-model",
            2,
        )
        .expect("publish second");

    let hits = storage
        .search_semantic(
            &[1.0, 0.0],
            "fixture-model",
            10,
            0.0,
            &storage::SearchOptions::default(),
        )
        .expect("semantic search");
    assert_eq!(hits.len(), 2);
    assert_eq!(hits[0].document_id, first.id);
    assert_eq!(hits[1].document_id, second.id);
    assert!((hits[0].score - 1.0).abs() < f64::EPSILON);
}

#[test]
fn malformed_vectors_are_rejected_before_publication() {
    let mut storage = Storage::in_memory().expect("storage");
    storage
        .upsert_document(input("invalid embedding"))
        .expect("document");
    let job = storage.claim_job().expect("claim").expect("job");
    let error = storage
        .complete_embedded_job(
            &job,
            &[EmbeddedChunk {
                content: &job.content,
                vector: &[0.5, 0.5],
            }],
            "fixture-model",
            2,
        )
        .expect_err("non-normalized vector must fail");
    assert!(matches!(error, StorageError::InvalidVectorNorm { .. }));
    assert!(storage.search("embedding", 10).expect("search").is_empty());
}
