use supermemory::model_config::load_or_create;

fn directory(label: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!(
        "supermemory-model-config-{label}-{}",
        storage::generate_id().expect("id")
    ))
}

#[test]
fn first_load_creates_requested_provider_defaults() {
    let directory = directory("defaults");
    std::fs::create_dir_all(&directory).expect("directory");

    let config = load_or_create(&directory).expect("config");

    assert_eq!(config.providers.openai.model, "gpt-5.6-luna");
    assert_eq!(config.providers.openai.reasoning_effort, "medium");
    assert_eq!(config.providers.gemini.model, "gemini-3.5-flash");
    assert!(directory.join("config.toml").is_file());
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        assert_eq!(
            std::fs::metadata(directory.join("config.toml"))
                .expect("metadata")
                .permissions()
                .mode()
                & 0o777,
            0o600
        );
    }
    std::fs::remove_dir_all(directory).expect("cleanup");
}

#[test]
fn existing_toml_overrides_every_provider_model() {
    let directory = directory("custom");
    std::fs::create_dir_all(&directory).expect("directory");
    std::fs::write(
        directory.join("config.toml"),
        r#"
[providers.openai]
model = "openai-custom"
reasoning_effort = "high"

[providers.anthropic]
model = "anthropic-custom"

[providers.gemini]
model = "gemini-custom"

[providers.groq]
model = "groq-custom"
"#,
    )
    .expect("write config");

    let config = load_or_create(&directory).expect("config");

    assert_eq!(config.providers.openai.model, "openai-custom");
    assert_eq!(config.providers.openai.reasoning_effort, "high");
    assert_eq!(config.providers.anthropic.model, "anthropic-custom");
    assert_eq!(config.providers.gemini.model, "gemini-custom");
    assert_eq!(config.providers.groq.model, "groq-custom");
    std::fs::remove_dir_all(directory).expect("cleanup");
}

#[test]
fn empty_model_is_rejected_with_its_toml_field() {
    let directory = directory("empty");
    std::fs::create_dir_all(&directory).expect("directory");
    std::fs::write(
        directory.join("config.toml"),
        r#"
[providers.openai]
model = ""
reasoning_effort = "medium"
"#,
    )
    .expect("write config");

    let error = load_or_create(&directory).expect_err("empty model should fail");

    assert!(error.to_string().contains("providers.openai.model"));
    std::fs::remove_dir_all(directory).expect("cleanup");
}
