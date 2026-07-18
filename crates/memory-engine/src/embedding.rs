//! Local BGE embedding inference using the existing Supermemory model assets.

use std::{
    path::{Path, PathBuf},
    sync::Mutex,
};

use ort::{
    session::{Session, builder::GraphOptimizationLevel},
    value::TensorRef,
};
use thiserror::Error;
use tokenizers::{PaddingParams, PaddingStrategy, Tokenizer, TruncationParams};

/// Output width of `Xenova/bge-base-en-v1.5`.
pub const BGE_DIMENSIONS: usize = 768;
const MAX_TOKENS: usize = 512;
const MAX_UTF16_UNITS: usize = 8_000;

static ORT_LIBRARY: Mutex<Option<PathBuf>> = Mutex::new(None);

/// A normalized embedding produced by the configured BGE model.
#[derive(Debug, Clone, PartialEq)]
pub struct EmbeddingVector(Box<[f32; BGE_DIMENSIONS]>);

impl EmbeddingVector {
    /// Creates a vector after validating its dimension, values, and norm.
    ///
    /// # Errors
    /// Returns an error for malformed or non-normalized model output.
    pub fn new(values: Vec<f32>) -> Result<Self, EmbeddingError> {
        let actual = values.len();
        let values: Box<[f32; BGE_DIMENSIONS]> =
            values.into_boxed_slice().try_into().map_err(|_| {
                EmbeddingError::UnexpectedDimensions {
                    expected: BGE_DIMENSIONS,
                    actual,
                }
            })?;
        if values.iter().any(|value| !value.is_finite()) {
            return Err(EmbeddingError::NonFiniteOutput);
        }
        let norm = values.iter().map(|value| value * value).sum::<f32>().sqrt();
        if (norm - 1.0).abs() > 1e-4 {
            return Err(EmbeddingError::InvalidNorm { norm });
        }
        Ok(Self(values))
    }

    /// Returns the normalized components.
    #[must_use]
    pub fn as_slice(&self) -> &[f32] {
        self.0.as_slice()
    }

    /// Computes cosine similarity using the normalized vectors' dot product.
    #[must_use]
    pub fn similarity(&self, other: &Self) -> f32 {
        self.as_slice()
            .iter()
            .zip(other.as_slice())
            .map(|(left, right)| left * right)
            .sum()
    }
}

/// A loaded tokenizer and ONNX session shared by document and query inference.
pub struct EmbeddingModel {
    tokenizer: Tokenizer,
    session: Mutex<Session>,
}

impl EmbeddingModel {
    /// Loads Supermemory's existing local BGE model and native ONNX Runtime.
    ///
    /// # Errors
    /// Returns a structured error if assets are missing, incompatible, or cannot initialize.
    pub fn load(model_dir: &Path, ort_library: &Path) -> Result<Self, EmbeddingError> {
        let mut configured_library = ORT_LIBRARY
            .lock()
            .map_err(|_| EmbeddingError::RuntimeInitializationUnavailable)?;
        if configured_library.is_none() {
            ort::init_from(ort_library.display().to_string())
                .commit()
                .map_err(EmbeddingError::InitializeRuntime)?;
            *configured_library = Some(ort_library.to_path_buf());
        }
        let Some(configured_library) = configured_library.as_ref() else {
            return Err(EmbeddingError::RuntimeInitializationUnavailable);
        };
        if configured_library.as_path() != ort_library {
            return Err(EmbeddingError::RuntimeAlreadyInitialized {
                configured: configured_library.clone(),
                requested: ort_library.to_path_buf(),
            });
        }

        let tokenizer_path = model_dir.join("tokenizer.json");
        let mut tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|source| {
            EmbeddingError::LoadTokenizer {
                path: tokenizer_path,
                source,
            }
        })?;
        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length: MAX_TOKENS,
                ..TruncationParams::default()
            }))
            .map_err(EmbeddingError::ConfigureTokenizer)?;
        tokenizer.with_padding(Some(PaddingParams {
            strategy: PaddingStrategy::BatchLongest,
            ..PaddingParams::default()
        }));

        let model_path = model_dir.join("onnx/model_quantized.onnx");
        let session = Session::builder()
            .and_then(|builder| {
                builder
                    .with_optimization_level(GraphOptimizationLevel::Level3)?
                    .with_intra_threads(1)?
                    .with_inter_threads(1)?
                    .commit_from_file(&model_path)
            })
            .map_err(|source| EmbeddingError::LoadModel {
                path: model_path,
                source,
            })?;

        Ok(Self {
            tokenizer,
            session: Mutex::new(session),
        })
    }

    /// Embeds a batch with attention-mask mean pooling and L2 normalization.
    ///
    /// # Errors
    /// Returns an error if tokenization, inference, or output validation fails.
    pub fn embed(&self, values: &[String]) -> Result<Vec<EmbeddingVector>, EmbeddingError> {
        if values.is_empty() {
            return Ok(Vec::new());
        }
        let truncated: Vec<String> = values
            .iter()
            .map(|value| truncate_utf16(value, MAX_UTF16_UNITS))
            .collect();
        let encodings = self
            .tokenizer
            .encode_batch(truncated, true)
            .map_err(EmbeddingError::Tokenize)?;
        let sequence = encodings[0].len();
        let batch = encodings.len();
        let ids: Vec<i64> = encodings
            .iter()
            .flat_map(|encoding| encoding.get_ids().iter().map(|id| i64::from(*id)))
            .collect();
        let masks: Vec<i64> = encodings
            .iter()
            .flat_map(|encoding| {
                encoding
                    .get_attention_mask()
                    .iter()
                    .map(|mask| i64::from(*mask))
            })
            .collect();
        let type_ids: Vec<i64> = encodings
            .iter()
            .flat_map(|encoding| encoding.get_type_ids().iter().map(|id| i64::from(*id)))
            .collect();
        let input_ids = TensorRef::from_array_view(([batch, sequence], ids.as_slice()))?;
        let attention_mask = TensorRef::from_array_view(([batch, sequence], masks.as_slice()))?;
        let token_type_ids = TensorRef::from_array_view(([batch, sequence], type_ids.as_slice()))?;
        let mut session = self
            .session
            .lock()
            .map_err(|_| EmbeddingError::SessionUnavailable)?;
        let outputs = session.run(ort::inputs![
            "input_ids" => input_ids,
            "attention_mask" => attention_mask,
            "token_type_ids" => token_type_ids,
        ])?;
        let (shape, hidden) = outputs["last_hidden_state"].try_extract_tensor::<f32>()?;
        let expected_shape = [
            i64::try_from(batch).map_err(|_| EmbeddingError::InputShapeTooLarge)?,
            i64::try_from(sequence).map_err(|_| EmbeddingError::InputShapeTooLarge)?,
            i64::try_from(BGE_DIMENSIONS).map_err(|_| EmbeddingError::InputShapeTooLarge)?,
        ];
        if shape.as_ref() != expected_shape {
            return Err(EmbeddingError::UnexpectedOutputShape {
                actual: shape.to_vec(),
            });
        }

        encodings
            .iter()
            .enumerate()
            .map(|(batch_index, encoding)| {
                let mut pooled = vec![0.0_f32; BGE_DIMENSIONS];
                let mut token_count = 0.0_f32;
                for (token_index, mask) in encoding.get_attention_mask().iter().enumerate() {
                    if *mask == 0 {
                        continue;
                    }
                    token_count += 1.0;
                    let offset = (batch_index * sequence + token_index) * BGE_DIMENSIONS;
                    for (target, value) in pooled
                        .iter_mut()
                        .zip(&hidden[offset..offset + BGE_DIMENSIONS])
                    {
                        *target += *value;
                    }
                }
                if token_count == 0.0 {
                    return Err(EmbeddingError::EmptyTokenSequence);
                }
                for value in &mut pooled {
                    *value /= token_count;
                }
                let norm = pooled.iter().map(|value| value * value).sum::<f32>().sqrt();
                if !norm.is_finite() || norm == 0.0 {
                    return Err(EmbeddingError::InvalidNorm { norm });
                }
                for value in &mut pooled {
                    *value /= norm;
                }
                EmbeddingVector::new(pooled)
            })
            .collect()
    }
}

fn truncate_utf16(value: &str, maximum: usize) -> String {
    if value.encode_utf16().count() <= maximum {
        return value.to_owned();
    }
    let mut units = 0;
    value
        .chars()
        .take_while(|character| {
            let next = units + character.len_utf16();
            if next > maximum {
                false
            } else {
                units = next;
                true
            }
        })
        .collect()
}

/// Failure while initializing or running local embedding inference.
#[derive(Debug, Error)]
pub enum EmbeddingError {
    #[error("failed to initialize ONNX Runtime from the existing library: {0}")]
    InitializeRuntime(#[source] ort::Error),
    #[error(
        "ONNX Runtime initialization lock is unavailable after a previous initialization failed"
    )]
    RuntimeInitializationUnavailable,
    #[error("ONNX Runtime is already initialized from {configured}, not requested {requested}")]
    RuntimeAlreadyInitialized {
        configured: PathBuf,
        requested: PathBuf,
    },
    #[error("failed to load BGE tokenizer from {path}: {source}")]
    LoadTokenizer {
        path: PathBuf,
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },
    #[error("failed to configure BGE tokenizer: {0}")]
    ConfigureTokenizer(#[source] Box<dyn std::error::Error + Send + Sync>),
    #[error("failed to load BGE ONNX model from {path}: {source}")]
    LoadModel {
        path: PathBuf,
        #[source]
        source: ort::Error,
    },
    #[error("failed to tokenize embedding input: {0}")]
    Tokenize(#[source] Box<dyn std::error::Error + Send + Sync>),
    #[error("embedding inference failed: {0}")]
    Inference(#[from] ort::Error),
    #[error("embedding session lock is unavailable after a previous inference failed")]
    SessionUnavailable,
    #[error("embedding input shape exceeds ONNX Runtime's signed dimension range")]
    InputShapeTooLarge,
    #[error("embedding model returned shape {actual:?}, expected [batch, sequence, 768]")]
    UnexpectedOutputShape { actual: Vec<i64> },
    #[error("embedding model returned {actual} dimensions, expected {expected}")]
    UnexpectedDimensions { expected: usize, actual: usize },
    #[error("embedding model returned a non-finite component")]
    NonFiniteOutput,
    #[error("embedding vector has invalid L2 norm {norm}")]
    InvalidNorm { norm: f32 },
    #[error("embedding tokenizer produced no attended tokens")]
    EmptyTokenSequence,
}
