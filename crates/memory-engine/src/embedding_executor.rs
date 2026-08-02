//! Bounded, priority-aware access to the local embedding model.

use std::{sync::Arc, time::Instant};

use thiserror::Error;
use tokio::sync::{mpsc, oneshot};

use crate::{EmbeddingError, EmbeddingModel, EmbeddingVector, PreparedEmbeddingInput};

/// Scheduling class for embedding work.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbeddingPriority {
    Query,
    Document,
    Memory,
    /// Low-priority cache/model warmup; shares the bounded bulk queue.
    Warmup,
}

struct Request {
    values: Vec<PreparedEmbeddingInput>,
    item_count: usize,
    estimated_tokens: usize,
    max_item_tokens: usize,
    enqueued_at: Instant,
    reply: oneshot::Sender<Result<Vec<EmbeddedText>, EmbeddingExecutorError>>,
}

/// One source text paired with its embedding after executor-owned inference.
#[derive(Debug)]
pub struct EmbeddedText {
    pub text: String,
    pub vector: EmbeddingVector,
}

/// Bounded executor that prevents callers from accumulating blocking inference tasks.
#[derive(Clone)]
pub struct EmbeddingExecutor {
    model: Arc<EmbeddingModel>,
    query: mpsc::Sender<Request>,
    bulk: mpsc::Sender<Request>,
    max_items: usize,
    max_padded_tokens: usize,
}

impl EmbeddingExecutor {
    /// Starts one model-owning inference task with bounded query and bulk queues.
    #[must_use]
    pub fn new(model: Arc<EmbeddingModel>, queue_capacity: usize, max_items: usize) -> Self {
        Self::with_limits(model, queue_capacity, max_items, 4_096)
    }

    /// Starts an executor with both item and padded-token batch limits.
    #[must_use]
    pub fn with_limits(
        model: Arc<EmbeddingModel>,
        queue_capacity: usize,
        max_items: usize,
        max_padded_tokens: usize,
    ) -> Self {
        let (query_tx, query_rx) = mpsc::channel(queue_capacity);
        let (bulk_tx, bulk_rx) = mpsc::channel(queue_capacity);
        tokio::spawn(run(
            Arc::clone(&model),
            query_rx,
            bulk_rx,
            max_items,
            max_padded_tokens,
        ));
        Self {
            model,
            query: query_tx,
            bulk: bulk_tx,
            max_items,
            max_padded_tokens,
        }
    }

    /// Enqueues length-aware microbatches and reassembles vectors in caller
    /// order. Each microbatch is independently admitted to the bounded queue,
    /// so a large document cannot occupy unbounded scheduler state or starve
    /// interactive queries.
    ///
    /// # Errors
    /// Returns an error if one item cannot fit in a model batch or the executor stops.
    pub async fn embed(
        &self,
        priority: EmbeddingPriority,
        values: Vec<String>,
    ) -> Result<Vec<EmbeddingVector>, EmbeddingExecutorError> {
        Ok(self
            .embed_owned(priority, values)
            .await?
            .into_iter()
            .map(|item| item.vector)
            .collect())
    }

    /// Embeds owned text and returns each original input with its vector. This
    /// lets publication reuse document chunks without recomputing boundaries.
    ///
    /// # Errors
    /// Returns an error if tokenization, batch admission, or inference fails.
    pub async fn embed_owned(
        &self,
        priority: EmbeddingPriority,
        values: Vec<String>,
    ) -> Result<Vec<EmbeddedText>, EmbeddingExecutorError> {
        let values = values
            .into_iter()
            .map(|value| {
                self.model
                    .prepare(&value)
                    .map_err(EmbeddingExecutorError::Embedding)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let lengths = values
            .iter()
            .map(|value| value.token_count)
            .collect::<Vec<_>>();
        let batches = microbatch_lengths(&lengths, self.max_items, self.max_padded_tokens)?;
        let mut output = Vec::with_capacity(values.len());
        let mut values = values.into_iter();
        for count in batches {
            // Awaiting each bounded submission gives queued queries a scheduling
            // opportunity between pieces of one logical document.
            output.extend(
                self.submit(priority, values.by_ref().take(count).collect())
                    .await?,
            );
        }
        Ok(output)
    }

    async fn submit(
        &self,
        priority: EmbeddingPriority,
        values: Vec<PreparedEmbeddingInput>,
    ) -> Result<Vec<EmbeddedText>, EmbeddingExecutorError> {
        let (reply, response) = oneshot::channel();
        let total_estimated_tokens: usize = values.iter().map(|value| value.token_count).sum();
        let max_item_tokens = values
            .iter()
            .map(|value| value.token_count)
            .max()
            .unwrap_or(1);
        let request = Request {
            item_count: values.len(),
            values,
            estimated_tokens: total_estimated_tokens,
            max_item_tokens,
            enqueued_at: Instant::now(),
            reply,
        };
        let queue = if priority == EmbeddingPriority::Query {
            &self.query
        } else {
            &self.bulk
        };
        queue
            .send(request)
            .await
            .map_err(|_| EmbeddingExecutorError::Closed)?;
        response.await.map_err(|_| EmbeddingExecutorError::Closed)?
    }
}

/// Calculates ordered, length-aware bounded microbatch sizes.
fn microbatch_lengths(
    lengths: &[usize],
    max_items: usize,
    max_padded_tokens: usize,
) -> Result<Vec<usize>, EmbeddingExecutorError> {
    let mut batches = Vec::new();
    let mut items = 0;
    let mut longest = 1;
    for &tokens in lengths {
        if tokens > max_padded_tokens {
            return Err(EmbeddingExecutorError::TooManyTokens {
                maximum: max_padded_tokens,
                actual: tokens,
            });
        }
        let next_items = items + 1;
        let next_longest = longest.max(tokens);
        if items > 0 && (next_items > max_items || next_items * next_longest > max_padded_tokens) {
            batches.push(items);
            items = 0;
            longest = 1;
        }
        items += 1;
        longest = longest.max(tokens);
    }
    if items > 0 {
        batches.push(items);
    }
    Ok(batches)
}

#[expect(
    clippy::too_many_lines,
    reason = "the scheduler's admission, batching, and response ordering share state"
)]
async fn run(
    model: Arc<EmbeddingModel>,
    mut query: mpsc::Receiver<Request>,
    mut bulk: mpsc::Receiver<Request>,
    max_items: usize,
    max_padded_tokens: usize,
) {
    let mut query_burst = 0_usize;
    // At most one request is held between batches when it would exceed the
    // current padded-token budget. It remains ahead of later work.
    let mut deferred = None;
    loop {
        // Serve a bounded burst of queries before a bulk request. Query latency
        // stays low without allowing an endless request stream to starve indexing.
        let request = deferred.take().or_else(|| {
            if query_burst >= 4 {
                bulk.try_recv()
                    .ok()
                    .map(|request| (EmbeddingPriority::Document, request))
            } else {
                None
            }
            .or_else(|| {
                query
                    .try_recv()
                    .ok()
                    .map(|request| (EmbeddingPriority::Query, request))
            })
            .or_else(|| {
                bulk.try_recv()
                    .ok()
                    .map(|request| (EmbeddingPriority::Document, request))
            })
        });
        let request = if let Some(request) = request {
            Some(request)
        } else {
            tokio::select! {
                biased;
                request = query.recv() => request.map(|request| (EmbeddingPriority::Query, request)),
                request = bulk.recv() => request.map(|request| (EmbeddingPriority::Document, request)),
            }
        };
        let Some((priority, request)) = request else {
            break;
        };
        query_burst = if priority == EmbeddingPriority::Query {
            query_burst.saturating_add(1)
        } else {
            0
        };
        let mut requests = vec![(priority, request)];
        let mut items = requests[0].1.values.len();
        let mut longest = requests[0].1.max_item_tokens;
        while items < max_items {
            // Never merge interactive queries with bulk work: a single long
            // document chunk would otherwise determine a query's padded shape.
            let next = match priority {
                EmbeddingPriority::Query => query
                    .try_recv()
                    .ok()
                    .map(|request| (EmbeddingPriority::Query, request)),
                EmbeddingPriority::Document
                | EmbeddingPriority::Memory
                | EmbeddingPriority::Warmup => bulk
                    .try_recv()
                    .ok()
                    .map(|request| (EmbeddingPriority::Document, request)),
            };
            let Some(next) = next else { break };
            let padded_tokens = (items + next.1.values.len()) * longest.max(next.1.max_item_tokens);
            if items + next.1.values.len() > max_items || padded_tokens > max_padded_tokens {
                // Requests are individually admissible; keep it for the next
                // batch instead of turning a scheduling boundary into failure.
                deferred = Some(next);
                break;
            }
            items += next.1.values.len();
            longest = longest.max(next.1.max_item_tokens);
            requests.push(next);
        }
        let queue_wait = requests
            .iter()
            .map(|(_, request)| request.enqueued_at.elapsed())
            .max()
            .unwrap_or_default();
        let mut values = requests
            .iter_mut()
            .flat_map(|(_, request)| std::mem::take(&mut request.values))
            .collect::<Vec<_>>();
        let started = Instant::now();
        let result = tokio::task::spawn_blocking({
            let model = Arc::clone(&model);
            move || {
                let result = model.embed_prepared(&mut values);
                (values, result)
            }
        })
        .await
        .map_err(|_| EmbeddingExecutorError::Closed);
        tracing::info!(
            ?priority,
            requests = requests.len(),
            items,
            padded_tokens = items * longest,
            token_count = requests
                .iter()
                .map(|(_, request)| request.estimated_tokens)
                .sum::<usize>(),
            queue_wait_ms = queue_wait.as_millis(),
            inference_ms = started.elapsed().as_millis(),
            "embedding_batch"
        );
        match result {
            Ok((mut values, Ok(mut vectors))) => {
                for (_, request) in requests {
                    let count = request.item_count;
                    let vectors = vectors.drain(..count);
                    let embedded = values
                        .drain(..count)
                        .zip(vectors)
                        .map(|(input, vector)| EmbeddedText {
                            text: input.text,
                            vector,
                        })
                        .collect();
                    let _ = request.reply.send(Ok(embedded));
                }
            }
            Ok((_, Err(error))) => {
                let error = EmbeddingExecutorError::Embedding(error);
                let message = error.to_string();
                for (_, request) in requests {
                    let _ = request.reply.send(Err(EmbeddingExecutorError::BatchFailed {
                        message: message.clone(),
                    }));
                }
            }
            Err(error) => {
                let message = error.to_string();
                for (_, request) in requests {
                    let _ = request.reply.send(Err(EmbeddingExecutorError::BatchFailed {
                        message: message.clone(),
                    }));
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::microbatch_lengths;

    #[test]
    fn large_mixed_document_is_partitioned_in_order_with_bounded_batches() {
        let batches = microbatch_lengths(&[2, 7, 2, 7, 2, 2], 3, 16).expect("admissible");
        assert_eq!(batches, [2, 2, 2]);
        assert_eq!(batches.iter().sum::<usize>(), 6);
    }

    #[test]
    fn one_item_over_the_safe_bound_is_rejected() {
        assert!(microbatch_lengths(&[17], 3, 16).is_err());
    }
}

/// Failure while submitting or executing an embedding request.
#[derive(Debug, Error)]
pub enum EmbeddingExecutorError {
    #[error("embedding executor has stopped")]
    Closed,
    #[error("embedding request has {actual} items; maximum is {maximum}")]
    TooManyItems { maximum: usize, actual: usize },
    #[error("embedding request has {actual} estimated tokens; maximum is {maximum}")]
    TooManyTokens { maximum: usize, actual: usize },
    #[error("request could not be added without exceeding batch limits")]
    BatchLimit,
    #[error("embedding batch failed: {message}")]
    BatchFailed { message: String },
    #[error("embedding inference failed: {0}")]
    Embedding(#[source] EmbeddingError),
}
