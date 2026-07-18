use memory_engine::{DocumentState, TransitionError};

#[test]
fn transition_accepts_ordered_progress() {
    assert_eq!(
        DocumentState::Embedding.transition(DocumentState::Indexing),
        Ok(DocumentState::Indexing)
    );
}

#[test]
fn transition_accepts_failed_retry() {
    assert_eq!(
        DocumentState::Failed.transition(DocumentState::Queued),
        Ok(DocumentState::Queued)
    );
}

#[test]
fn transition_rejects_skipping_processing() {
    assert_eq!(
        DocumentState::Queued.transition(DocumentState::Done),
        Err(TransitionError {
            from: DocumentState::Queued,
            to: DocumentState::Done,
        })
    );
}
