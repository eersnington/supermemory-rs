//! Non-secret provider model configuration.

use std::{
    fs::OpenOptions,
    io::Write,
    path::{Path, PathBuf},
};

use memory_engine::ProviderModels;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// User-editable model configuration stored in `config.toml`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ModelConfig {
    pub providers: Providers,
}

/// Configuration for each supported provider.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Providers {
    pub openai: OpenAi,
    pub anthropic: Model,
    pub gemini: Model,
    pub groq: Model,
}

impl Default for Providers {
    fn default() -> Self {
        let defaults = ProviderModels::default();
        Self {
            openai: OpenAi {
                model: defaults.openai,
                reasoning_effort: defaults.openai_reasoning_effort,
            },
            anthropic: Model {
                model: defaults.anthropic,
            },
            gemini: Model {
                model: defaults.gemini,
            },
            groq: Model {
                model: defaults.groq,
            },
        }
    }
}

/// `OpenAI` model and reasoning configuration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OpenAi {
    pub model: String,
    pub reasoning_effort: String,
}

impl Default for OpenAi {
    fn default() -> Self {
        Self {
            model: ProviderModels::default().openai,
            reasoning_effort: "medium".to_owned(),
        }
    }
}

/// Model name for one provider.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Model {
    pub model: String,
}

impl From<ModelConfig> for ProviderModels {
    fn from(config: ModelConfig) -> Self {
        Self {
            openai: config.providers.openai.model,
            openai_reasoning_effort: config.providers.openai.reasoning_effort,
            anthropic: config.providers.anthropic.model,
            gemini: config.providers.gemini.model,
            groq: config.providers.groq.model,
        }
    }
}

/// Loads `config.toml`, creating it with current defaults when absent.
///
/// # Errors
/// Returns an error when the file cannot be read, parsed, or safely created.
pub fn load_or_create(data_dir: &Path) -> Result<ModelConfig, ModelConfigError> {
    let path = data_dir.join("config.toml");
    match std::fs::read_to_string(&path) {
        Ok(contents) => toml::from_str(&contents)
            .map_err(|source| ModelConfigError::Parse {
                path: path.clone(),
                source,
            })
            .and_then(|config| validate(path, config)),
        Err(source) if source.kind() == std::io::ErrorKind::NotFound => {
            let config = ModelConfig::default();
            write_new(&path, &config)?;
            Ok(config)
        }
        Err(source) => Err(ModelConfigError::Read { path, source }),
    }
}

fn validate(path: PathBuf, config: ModelConfig) -> Result<ModelConfig, ModelConfigError> {
    let values = [
        (
            "providers.openai.model",
            config.providers.openai.model.as_str(),
        ),
        (
            "providers.openai.reasoning_effort",
            config.providers.openai.reasoning_effort.as_str(),
        ),
        (
            "providers.anthropic.model",
            config.providers.anthropic.model.as_str(),
        ),
        (
            "providers.gemini.model",
            config.providers.gemini.model.as_str(),
        ),
        ("providers.groq.model", config.providers.groq.model.as_str()),
    ];
    if let Some((field, _)) = values.iter().find(|(_, value)| value.trim().is_empty()) {
        return Err(ModelConfigError::EmptyValue { path, field });
    }
    Ok(config)
}

fn write_new(path: &Path, config: &ModelConfig) -> Result<(), ModelConfigError> {
    let contents = toml::to_string_pretty(config).map_err(ModelConfigError::Serialize)?;
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let mut file = options
        .open(path)
        .map_err(|source| ModelConfigError::Write {
            path: path.to_path_buf(),
            source,
        })?;
    file.write_all(contents.as_bytes())
        .and_then(|()| file.sync_all())
        .map_err(|source| ModelConfigError::Write {
            path: path.to_path_buf(),
            source,
        })
}

/// Failure while loading or creating provider model configuration.
#[derive(Debug, Error)]
pub enum ModelConfigError {
    #[error("failed to read model configuration from {path}: {source}")]
    Read {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to parse model configuration at {path}: {source}")]
    Parse {
        path: PathBuf,
        #[source]
        source: toml::de::Error,
    },
    #[error("model configuration {path} has an empty {field}; set a non-empty value and restart")]
    EmptyValue { path: PathBuf, field: &'static str },
    #[error("failed to serialize default model configuration: {0}")]
    Serialize(#[source] toml::ser::Error),
    #[error("failed to write model configuration to {path}: {source}")]
    Write {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
}
