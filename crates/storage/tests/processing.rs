use serde_json::{Map, json};
use storage::{
    EmbeddedChunk, FilterCondition, FilterExpression, FilterKind, MemoryProposal, NumericOperator,
    SearchOptions, Storage, StorageError, UpsertDocument, generate_id,
};

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
        .complete_job(&job, std::slice::from_ref(&job.content))
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
        .complete_job(&first, std::slice::from_ref(&first.content))
        .expect("complete first");

    value.content = "second narwhal content".into();
    storage.upsert_document(value).expect("update");
    let second = storage.claim_job().expect("claim").expect("second job");
    storage
        .complete_job(&second, std::slice::from_ref(&second.content))
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
fn semantic_filters_run_before_ranking_and_limiting() {
    let mut storage = Storage::in_memory().expect("storage");
    for (content, tags, filepath, topic, vector) in [
        (
            "matching filtered chunk",
            vec!["first".into(), "wanted".into()],
            "folder/match.txt",
            "target",
            [1.0, 0.0],
        ),
        (
            "higher scoring but filtered chunk",
            vec!["first".into()],
            "other/nope.txt",
            "other",
            [1.0, 0.0],
        ),
    ] {
        let mut value = input(content);
        value.container_tags = tags;
        value.filepath = Some(filepath.into());
        value.metadata.insert("topic".into(), json!(topic));
        storage.upsert_document(value).expect("document");
        let job = storage.claim_job().expect("claim").expect("job");
        storage
            .complete_embedded_job(
                &job,
                &[EmbeddedChunk {
                    content: &job.content,
                    vector: &vector,
                }],
                "fixture-model",
                2,
            )
            .expect("publish");
    }

    let hits = storage
        .search_semantic(
            &[1.0, 0.0],
            "fixture-model",
            1,
            0.0,
            &SearchOptions {
                container_tags: vec!["wanted".into()],
                filepath: Some("folder/".into()),
                filters: Some(FilterExpression::Condition(FilterCondition {
                    key: "topic".into(),
                    value: "target".into(),
                    kind: FilterKind::Metadata,
                    numeric_operator: NumericOperator::Equal,
                    negate: false,
                    ignore_case: false,
                })),
                ..SearchOptions::default()
            },
        )
        .expect("filtered semantic search");
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].chunk, "matching filtered chunk");
}

#[test]
fn active_processing_jobs_cover_document_and_memory_publication() {
    let mut storage = Storage::in_memory().expect("storage");
    storage
        .upsert_document(input("queue drain source"))
        .expect("document");
    assert!(
        storage
            .has_active_processing_jobs()
            .expect("active document")
    );

    let document_job = storage.claim_job().expect("claim").expect("document job");
    let mut vector = vec![0.0; 768];
    vector[0] = 1.0;
    storage
        .complete_embedded_job_with_memory_extraction(
            &document_job,
            &[EmbeddedChunk {
                content: &document_job.content,
                vector: &vector,
            }],
            "fixture-model",
            768,
        )
        .expect("schedule extraction");
    assert!(storage.has_active_processing_jobs().expect("active memory"));

    let memory_job = storage
        .claim_memory_job()
        .expect("claim memory")
        .expect("memory job");
    storage
        .complete_memory_job(&memory_job)
        .expect("complete memory");
    assert!(!storage.has_active_processing_jobs().expect("drained queue"));
}

#[test]
fn interrupted_memory_job_reuses_cached_extraction_after_restart() {
    let path = std::env::temp_dir().join(format!(
        "supermemory-memory-recovery-{}.db",
        generate_id().expect("id")
    ));
    let mut storage = Storage::open(&path).expect("storage");
    storage
        .upsert_document(input("A durable extraction source"))
        .expect("document");
    let document_job = storage.claim_job().expect("claim").expect("document job");
    let mut vector = vec![0.0; 768];
    vector[0] = 1.0;
    storage
        .complete_embedded_job_with_memory_extraction(
            &document_job,
            &[EmbeddedChunk {
                content: &document_job.content,
                vector: &vector,
            }],
            "fixture-model",
            768,
        )
        .expect("schedule extraction");
    let memory_job = storage
        .claim_memory_job()
        .expect("claim memory")
        .expect("memory job");
    storage
        .cache_memory_extraction(&memory_job, "[]")
        .expect("cache extraction");
    drop(storage);

    let mut reopened = Storage::open(&path).expect("reopen");
    let recovered = reopened
        .claim_memory_job()
        .expect("claim recovered")
        .expect("recovered memory job");
    assert_eq!(recovered.extraction_result.as_deref(), Some("[]"));
    assert_eq!(recovered.attempts, 2);
    reopened
        .complete_memory_job(&recovered)
        .expect("complete recovered job");

    drop(reopened);
    std::fs::remove_file(path).expect("remove database");
}

#[test]
fn memory_extraction_does_not_block_a_new_document_revision() {
    let mut storage = Storage::in_memory().expect("storage");
    let mut value = input("first extraction source");
    value.custom_id = Some("revision-during-extraction".into());
    storage.upsert_document(value.clone()).expect("create");
    let document_job = storage.claim_job().expect("claim").expect("document job");
    let mut vector = vec![0.0; 768];
    vector[0] = 1.0;
    storage
        .complete_embedded_job_with_memory_extraction(
            &document_job,
            &[EmbeddedChunk {
                content: &document_job.content,
                vector: &vector,
            }],
            "fixture-model",
            768,
        )
        .expect("schedule extraction");

    let memory_job = storage
        .claim_memory_job()
        .expect("claim memory")
        .expect("memory job");
    let unchanged = storage.upsert_document(value.clone()).expect("same upsert");
    assert!(!unchanged.enqueued);
    assert_eq!(unchanged.status, "indexing");

    value.metadata.insert("date".into(), json!("2026-07-30"));
    let update = storage.upsert_document(value).expect("metadata update");
    assert!(update.enqueued);
    assert_eq!(update.status, "queued");
    assert!(matches!(
        storage.cache_memory_extraction(&memory_job, "[]"),
        Err(StorageError::StaleRevision { .. })
    ));
    assert_eq!(
        storage
            .claim_job()
            .expect("claim replacement")
            .expect("job")
            .content,
        "first extraction source"
    );
}

#[test]
fn permanent_memory_failure_is_terminal() {
    let mut storage = Storage::in_memory().expect("storage");
    let document = storage
        .upsert_document(input("A permanently failed extraction"))
        .expect("document");
    let document_job = storage.claim_job().expect("claim").expect("document job");
    let mut vector = vec![0.0; 768];
    vector[0] = 1.0;
    storage
        .complete_embedded_job_with_memory_extraction(
            &document_job,
            &[EmbeddedChunk {
                content: &document_job.content,
                vector: &vector,
            }],
            "fixture-model",
            768,
        )
        .expect("schedule extraction");
    let memory_job = storage
        .claim_memory_job()
        .expect("claim memory")
        .expect("memory job");
    storage
        .retry_memory_job(&memory_job, "authentication", "denied", None)
        .expect("terminal failure");

    assert_eq!(
        storage
            .find_document(&document.id)
            .expect("document")
            .expect("stored document")
            .status,
        "failed"
    );
    assert!(storage.claim_memory_job().expect("claim again").is_none());
}

#[test]
fn stale_revision_rejects_memory_publication() {
    let mut storage = Storage::in_memory().expect("storage");
    let document = storage
        .upsert_document(input("A stale memory source"))
        .expect("document");
    let proposal = MemoryProposal {
        temporary_id: "tmp_1".to_owned(),
        content: "This must not be published.".to_owned(),
        is_inferred: false,
        is_static: false,
        metadata: Map::new(),
        parents: Vec::new(),
        forget_after: None,
        forget_reason: None,
        vector: vec![1.0, 0.0],
    };
    let organization = storage.local_organization_id().to_owned();
    let error = storage
        .reconcile_memories_for(
            &organization,
            &document.id,
            Some(2),
            None,
            "sm_project_default",
            &[proposal],
            "fixture-model",
            2,
        )
        .expect_err("stale revision");

    assert!(matches!(error, StorageError::StaleRevision { .. }));
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
