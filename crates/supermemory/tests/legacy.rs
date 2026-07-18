#[test]
fn legacy_snapshot_exports_with_matching_pglite_runtime() {
    if std::env::var_os("SUPERMEMORY_TEST_LEGACY").is_none() {
        return;
    }
    let Some(home) = std::env::var_os("HOME") else {
        return;
    };
    let legacy = std::path::PathBuf::from(home).join(".supermemory");
    let output = std::env::temp_dir().join(format!(
        "supermemory-export-{}.jsonl",
        storage::generate_id().expect("id")
    ));
    supermemory::legacy::export_snapshot(&legacy, &legacy.join("runtime/pglite"), &output)
        .expect("legacy export");
    let export = std::fs::read_to_string(&output).expect("export");
    assert!(
        export
            .lines()
            .next()
            .is_some_and(|line| line.contains("\"type\":\"manifest\""))
    );
    assert!(
        export
            .lines()
            .last()
            .is_some_and(|line| line == "{\"type\":\"complete\"}")
    );
    let database = std::env::temp_dir().join(format!(
        "supermemory-import-{}.db",
        storage::generate_id().expect("id")
    ));
    let mut storage = storage::Storage::open(&database).expect("import database");
    let report = storage
        .import_legacy_export(&output)
        .expect("legacy import");
    assert!(report.organizations > 0 || report.documents == 0);
    drop(storage);
    std::fs::remove_file(output).expect("cleanup");
    std::fs::remove_file(database).expect("database cleanup");
}
