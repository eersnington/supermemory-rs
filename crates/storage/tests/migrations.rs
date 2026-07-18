use std::path::PathBuf;

use rusqlite::Connection;
use storage::{Storage, StorageError, generate_id};

fn path(label: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "supermemory-{label}-{}.db",
        generate_id().expect("id")
    ))
}

#[test]
fn malformed_existing_table_is_rejected() {
    let path = path("malformed-existing");
    Connection::open(&path)
        .expect("open")
        .execute("CREATE TABLE documents (wrong TEXT)", [])
        .expect("fixture");
    assert!(Storage::open(&path).is_err());
    std::fs::remove_file(path).expect("remove database");
}

#[test]
fn schema_marked_current_but_malformed_is_rejected() {
    let path = path("malformed-current");
    Connection::open(&path).expect("open").execute_batch("CREATE TABLE schema_migrations (version INTEGER PRIMARY KEY NOT NULL, applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP) STRICT; INSERT INTO schema_migrations(version) VALUES (1); CREATE TABLE organizations(id TEXT, slug TEXT, created_at TEXT); CREATE TABLE documents(wrong TEXT); CREATE TABLE jobs(wrong TEXT);").expect("fixture");
    assert!(
        matches!(Storage::open(&path), Err(StorageError::MalformedSchema { table, .. }) if table == "documents")
    );
    std::fs::remove_file(path).expect("remove database");
}

#[test]
fn concurrent_file_open_serializes_migration_and_seed() {
    let path = path("concurrent");
    let handles: Vec<_> = (0..16)
        .map(|_| {
            let path = path.clone();
            std::thread::spawn(move || {
                Storage::open(path).and_then(|storage| storage.migration_version())
            })
        })
        .collect();
    let versions: Vec<_> = handles
        .into_iter()
        .map(|handle| handle.join().expect("thread").expect("open"))
        .collect();
    assert!(versions.iter().all(|version| *version == 3));

    let connection = Connection::open(&path).expect("inspect migrated database");
    let organizations = connection
        .prepare("SELECT id FROM organizations WHERE slug = 'local'")
        .expect("prepare organization query")
        .query_map([], |row| row.get::<_, String>(0))
        .expect("query organizations")
        .collect::<Result<Vec<_>, _>>()
        .expect("read organizations");
    assert_eq!(organizations.len(), 1);
    assert_eq!(organizations[0].len(), 22);
    drop(connection);
    std::fs::remove_file(path).expect("remove database");
}
