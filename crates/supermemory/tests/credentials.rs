use std::collections::BTreeMap;

use supermemory::credentials::{decrypt_file, encrypt_file, load_or_import};

fn directory(label: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!(
        "supermemory-credentials-{label}-{}",
        storage::generate_id().expect("id")
    ))
}

#[test]
fn sme1_round_trip_preserves_quoted_provider_values() {
    let directory = directory("roundtrip");
    std::fs::create_dir_all(&directory).expect("directory");
    let path = directory.join("env.enc");
    let values = BTreeMap::from([
        ("OPENAI_API_KEY".to_owned(), "secret'value".to_owned()),
        (
            "OPENAI_BASE_URL".to_owned(),
            "http://localhost:11434/v1".to_owned(),
        ),
    ]);
    encrypt_file(&path, &directory, &values).expect("encrypt");
    let frame = std::fs::read(&path).expect("frame");
    assert_eq!(&frame[..4], b"SME1");
    assert_eq!(decrypt_file(&path, &directory).expect("decrypt"), values);
    std::fs::remove_dir_all(directory).expect("cleanup");
}

#[test]
fn legacy_import_reencrypts_without_modifying_source() {
    let legacy = directory("legacy");
    let current = directory("current");
    std::fs::create_dir_all(&legacy).expect("legacy directory");
    let values = BTreeMap::from([("GROQ_API_KEY".to_owned(), "secret".to_owned())]);
    let legacy_path = legacy.join("env.enc");
    encrypt_file(&legacy_path, &legacy, &values).expect("legacy encrypt");
    let original = std::fs::read(&legacy_path).expect("legacy frame");

    assert_eq!(load_or_import(&current, &legacy).expect("import"), values);
    assert_eq!(std::fs::read(&legacy_path).expect("legacy after"), original);
    assert_eq!(
        decrypt_file(&current.join("env.enc"), &current).expect("current decrypt"),
        values
    );
    std::fs::remove_dir_all(legacy).expect("legacy cleanup");
    std::fs::remove_dir_all(current).expect("current cleanup");
}
