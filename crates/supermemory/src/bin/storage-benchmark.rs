//! Deterministic provider-free Turso/pgvector regression benchmark.

use std::time::{Duration, Instant};

use serde_json::{Map, Value, json};
use storage::{MemoryParent, MemoryProposal, Storage, UpsertDocument, generate_id};

const MEMORY_COUNT: usize = 100;
const SEARCH_RUNS: usize = 20;
const SEARCH_RUNS_F64: f64 = 20.0;
const MODEL: &str = "deterministic-768d";
const TAG: &str = "storage-benchmark";

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let database = std::env::temp_dir().join(format!(
        "supermemory-storage-benchmark-{}.db",
        generate_id()?
    ));
    let storage = supermemory::start_embedded_turso_for_benchmark(database.clone()).await?;
    let report = tokio::task::spawn_blocking(move || run_benchmark(storage)).await??;
    println!("{}", serde_json::to_string_pretty(&report)?);
    let _ = std::fs::remove_file(database);
    Ok(())
}

fn run_benchmark(mut storage: Storage) -> Result<Value, storage::StorageError> {
    let document = storage.upsert_document(UpsertDocument {
        content: "provider-free deterministic vector corpus".to_owned(),
        custom_id: Some("storage-benchmark".to_owned()),
        container_tags: vec![TAG.to_owned()],
        entity_context: None,
        metadata: Map::new(),
        task_type: "memory".to_owned(),
        filepath: None,
        filter_by_metadata: Map::new(),
        dreaming: "dynamic".to_owned(),
    })?;
    let proposals = (0..MEMORY_COUNT)
        .map(|index| MemoryProposal {
            temporary_id: format!("tmp_{index}"),
            content: format!("deterministic memory {index:03}"),
            is_inferred: false,
            is_static: false,
            metadata: Map::from_iter([("ordinal".to_owned(), json!(index))]),
            parents: Vec::<MemoryParent>::new(),
            forget_after: None,
            forget_reason: None,
            vector: unit_vector(index),
        })
        .collect::<Vec<_>>();

    let insertion_started = Instant::now();
    let inserted = storage.reconcile_memories(
        &document.id,
        TAG,
        &proposals,
        MODEL,
        memory_engine::BGE_DIMENSIONS,
    )?;
    let insertion = insertion_started.elapsed();
    let expected_first = inserted
        .first()
        .expect("benchmark corpus is non-empty")
        .id
        .clone();
    let query = unit_vector(0);

    let mut sql = Duration::ZERO;
    let mut hydration = Duration::ZERO;
    let search_started = Instant::now();
    for _ in 0..SEARCH_RUNS {
        let (hits, vector_sql, hydration_time) = storage.profile_memory_search(
            storage.local_organization_id(),
            &query,
            MODEL,
            TAG,
            10,
            MEMORY_COUNT,
            0.0,
            false,
        )?;
        assert_eq!(
            hits.first().map(|hit| &hit.record.id),
            Some(&expected_first)
        );
        for (index, hit) in hits.iter().enumerate() {
            assert_eq!(
                hit.record.memory,
                format!("deterministic memory {index:03}")
            );
        }
        sql += vector_sql;
        hydration += hydration_time;
    }
    let search_total = search_started.elapsed();
    let response_overhead = search_total.saturating_sub(sql + hydration);
    Ok(json!({
        "memoryCount": MEMORY_COUNT,
        "searchRuns": SEARCH_RUNS,
        "dimensions": memory_engine::BGE_DIMENSIONS,
        "insertionMs": insertion.as_secs_f64() * 1000.0,
        "searchMeanMs": search_total.as_secs_f64() * 1000.0 / SEARCH_RUNS_F64,
        "vectorSqlMeanMs": sql.as_secs_f64() * 1000.0 / SEARCH_RUNS_F64,
        "hydrationMeanMs": hydration.as_secs_f64() * 1000.0 / SEARCH_RUNS_F64,
        "responseOverheadMeanMs": response_overhead.as_secs_f64() * 1000.0 / SEARCH_RUNS_F64,
        "topResultVerified": true,
    }))
}

fn unit_vector(index: usize) -> Vec<f32> {
    let angle = f32::from(u16::try_from(index).expect("benchmark index fits u16")) * 0.01;
    let mut vector = vec![0.0; memory_engine::BGE_DIMENSIONS];
    vector[0] = angle.cos();
    vector[1] = angle.sin();
    vector
}
