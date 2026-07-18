//! Core document lifecycle types and invariants.

mod chunking;
mod embedding;

pub use chunking::{
    ChunkingError, DEFAULT_CHUNK_SIZE, MAX_CHUNK_SIZE, chunk_text, normalize_extracted_text,
};
pub use embedding::{BGE_DIMENSIONS, EmbeddingError, EmbeddingModel, EmbeddingVector};

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Processing state persisted for a document.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DocumentState {
    /// The state is not known.
    Unknown,
    /// The document is waiting for a worker.
    Queued,
    /// Content is being extracted.
    Extracting,
    /// Content is being split into chunks.
    Chunking,
    /// Chunks are being embedded.
    Embedding,
    /// Chunks are being indexed.
    Indexing,
    /// Processing completed successfully.
    Done,
    /// Processing stopped with an error.
    Failed,
}

impl DocumentState {
    /// Moves a document to `next` when that lifecycle edge is legal.
    ///
    /// # Errors
    ///
    /// Returns [`TransitionError`] when the requested edge is not part of the
    /// document lifecycle.
    pub fn transition(self, next: Self) -> Result<Self, TransitionError> {
        match (self, next) {
            (Self::Unknown | Self::Failed, Self::Queued)
            | (Self::Queued, Self::Extracting)
            | (Self::Extracting, Self::Chunking)
            | (Self::Chunking, Self::Embedding)
            | (Self::Embedding, Self::Indexing)
            | (Self::Indexing, Self::Done)
            | (
                Self::Queued | Self::Extracting | Self::Chunking | Self::Embedding | Self::Indexing,
                Self::Failed,
            ) => Ok(next),
            _ => Err(TransitionError {
                from: self,
                to: next,
            }),
        }
    }

    /// Returns the stable value used by storage and API boundaries.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Unknown => "unknown",
            Self::Queued => "queued",
            Self::Extracting => "extracting",
            Self::Chunking => "chunking",
            Self::Embedding => "embedding",
            Self::Indexing => "indexing",
            Self::Done => "done",
            Self::Failed => "failed",
        }
    }
}

/// A rejected document lifecycle transition.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
#[error("document cannot transition from {from:?} to {to:?}")]
pub struct TransitionError {
    /// Current document state.
    pub from: DocumentState,
    /// Requested document state.
    pub to: DocumentState,
}
