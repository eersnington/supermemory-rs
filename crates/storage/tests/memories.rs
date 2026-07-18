use serde_json::{Map, json};
use storage::{MemoryParent, MemoryProposal, Storage, UpsertDocument};

fn document() -> UpsertDocument {
    UpsertDocument {
        content: "A source document".to_owned(),
        custom_id: None,
        container_tags: vec!["sm_project_default".to_owned()],
        entity_context: None,
        metadata: Map::new(),
        task_type: "memory".to_owned(),
        filepath: None,
        filter_by_metadata: Map::new(),
        dreaming: "dynamic".to_owned(),
    }
}

#[test]
fn reconciliation_builds_version_lineage_and_invalidates_update_parent() {
    let mut storage = Storage::in_memory().expect("storage");
    let document = storage.upsert_document(document()).expect("document");
    let proposals = vec![
        MemoryProposal {
            temporary_id: "tmp_1".to_owned(),
            content: "The project uses SQLite.".to_owned(),
            is_inferred: false,
            is_static: true,
            metadata: Map::new(),
            parents: Vec::new(),
            forget_after: None,
            forget_reason: None,
            vector: vec![1.0, 0.0],
        },
        MemoryProposal {
            temporary_id: "tmp_2".to_owned(),
            content: "The project uses SQLite in WAL mode.".to_owned(),
            is_inferred: false,
            is_static: true,
            metadata: Map::from_iter([("topic".to_owned(), json!("database"))]),
            parents: vec![MemoryParent {
                memory_id: "tmp_1".to_owned(),
                relation: "updates".to_owned(),
            }],
            forget_after: None,
            forget_reason: None,
            vector: vec![0.8, 0.6],
        },
    ];
    let memories = storage
        .reconcile_memories(
            &document.id,
            "sm_project_default",
            &proposals,
            "fixture-model",
            2,
        )
        .expect("reconcile");
    assert_eq!(memories.len(), 2);
    assert!(!memories[0].is_latest);
    assert!(!memories[0].is_static);
    assert_eq!(
        memories[1].parent_memory_id.as_deref(),
        Some(memories[0].id.as_str())
    );
    assert_eq!(
        memories[1].root_memory_id.as_deref(),
        Some(memories[0].id.as_str())
    );
    assert_eq!(memories[1].version, 2);
    let profile = storage
        .static_profile("sm_project_default")
        .expect("profile");
    assert_eq!(profile.len(), 1);
    assert_eq!(profile[0].id, memories[1].id);
}

#[test]
fn exact_duplicate_reuses_memory_and_adds_source_idempotently() {
    let mut storage = Storage::in_memory().expect("storage");
    let document = storage.upsert_document(document()).expect("document");
    let proposal = MemoryProposal {
        temporary_id: "tmp_1".to_owned(),
        content: "A stable fact.".to_owned(),
        is_inferred: false,
        is_static: false,
        metadata: Map::new(),
        parents: Vec::new(),
        forget_after: None,
        forget_reason: None,
        vector: vec![1.0, 0.0],
    };
    let first = storage
        .reconcile_memories(
            &document.id,
            "sm_project_default",
            std::slice::from_ref(&proposal),
            "fixture-model",
            2,
        )
        .expect("first");
    let second = storage
        .reconcile_memories(
            &document.id,
            "sm_project_default",
            &[proposal],
            "fixture-model",
            2,
        )
        .expect("second");
    assert_eq!(first[0].id, second[0].id);
}

#[test]
fn forgetting_removes_memory_from_active_profile_without_deleting_history() {
    let mut storage = Storage::in_memory().expect("storage");
    let document = storage.upsert_document(document()).expect("document");
    let proposal = MemoryProposal {
        temporary_id: "tmp_1".to_owned(),
        content: "The user prefers dark mode.".to_owned(),
        is_inferred: false,
        is_static: true,
        metadata: Map::from_iter([("buckets".to_owned(), json!(["preferences"]))]),
        parents: Vec::new(),
        forget_after: None,
        forget_reason: None,
        vector: vec![1.0, 0.0],
    };
    let memory = storage
        .reconcile_memories(
            &document.id,
            "sm_project_default",
            &[proposal],
            "fixture-model",
            2,
        )
        .expect("memory")
        .remove(0);
    assert_eq!(
        storage
            .bucket_profile("sm_project_default", "preferences")
            .expect("bucket")
            .len(),
        1
    );
    storage
        .forget_memory(
            Some(&memory.id),
            None,
            "sm_project_default",
            Some("user_requested"),
        )
        .expect("forget");
    assert!(
        storage
            .static_profile("sm_project_default")
            .expect("profile")
            .is_empty()
    );
    let forgotten = storage
        .search_memories(
            &[1.0, 0.0],
            "fixture-model",
            "sm_project_default",
            10,
            0.0,
            true,
        )
        .expect("forgotten search");
    assert_eq!(forgotten[0].record.id, memory.id);
    assert!(forgotten[0].record.is_forgotten);
}
