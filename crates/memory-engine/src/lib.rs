//! Local chunking, embedding, and provider integration.

mod chunking;
mod embedding;
mod embedding_executor;
mod provider;

pub use chunking::{
    ChunkingError, DEFAULT_CHUNK_SIZE, MAX_CHUNK_SIZE, chunk_text, normalize_extracted_text,
};
pub use embedding::{
    BGE_DIMENSIONS, BGE_MODEL_ID, EmbeddingError, EmbeddingModel, EmbeddingVector,
};
pub use embedding_executor::{EmbeddingExecutor, EmbeddingExecutorError, EmbeddingPriority};
pub use provider::{
    ExtractionFailure, ExtractionOutcome, ExtractionUsage, MemoryCandidate, MemoryProvider,
    ParentRelation, ProviderConfig, ProviderError, ProviderKind, ProviderModels, RelationKind,
    TemporalContext,
};
