use super::{
    ClaimedJob, DocumentState, EmbeddedChunk, HashMap, OptionalExtension, Storage, StorageError,
    TransactionBehavior, generate_id, params, validate_vector, vector_bytes,
};

impl Storage {
    /// Atomically claims the oldest available document job.
    ///
    /// # Errors
    /// Returns an error if the claim transaction cannot be read, written, or committed.
    pub fn claim_job(&mut self) -> Result<Option<ClaimedJob>, StorageError> {
        let tx = self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(StorageError::Write)?;
        let job = tx
            .query_row(
                "SELECT jobs.id, documents.id, documents.content, jobs.revision FROM jobs JOIN documents ON documents.id=jobs.document_id WHERE jobs.kind='document' AND jobs.status='queued' AND jobs.revision=documents.revision AND jobs.available_at <= CURRENT_TIMESTAMP ORDER BY jobs.created_at, jobs.rowid LIMIT 1",
                [],
                |row| {
                    Ok(ClaimedJob {
                        id: row.get(0)?,
                        document_id: row.get(1)?,
                        content: row.get(2)?,
                        revision: row.get(3)?,
                    })
                },
            )
            .optional()
            .map_err(StorageError::Read)?;
        let Some(job) = job else {
            tx.commit().map_err(StorageError::Write)?;
            return Ok(None);
        };
        tx.execute(
            "UPDATE jobs SET status='extracting', attempts=attempts+1, updated_at=CURRENT_TIMESTAMP WHERE id=?1 AND revision=?2 AND status='queued'",
            params![job.id, job.revision],
        )
        .map_err(StorageError::Write)?;
        tx.execute(
            "UPDATE documents SET status='extracting', updated_at=CURRENT_TIMESTAMP WHERE id=?1 AND revision=?2",
            params![job.document_id, job.revision],
        )
        .map_err(StorageError::Write)?;
        tx.commit().map_err(StorageError::Write)?;
        Ok(Some(job))
    }

    /// Replaces a document's searchable chunks and completes its claimed job atomically.
    ///
    /// # Errors
    /// Returns an error if searchable content or lifecycle state cannot be persisted.
    pub fn complete_job(
        &mut self,
        job: &ClaimedJob,
        chunks: &[String],
    ) -> Result<(), StorageError> {
        let embedded: Vec<_> = chunks
            .iter()
            .map(|content| EmbeddedChunk {
                content,
                vector: &[],
            })
            .collect();
        self.publish_job(job, &embedded, None, false)
    }

    /// Atomically publishes chunks and normalized vectors for a claimed revision.
    ///
    /// # Errors
    /// Returns an error when vector validation fails or the revision became stale.
    pub fn complete_embedded_job(
        &mut self,
        job: &ClaimedJob,
        chunks: &[EmbeddedChunk<'_>],
        model_id: &str,
        dimensions: usize,
    ) -> Result<(), StorageError> {
        if dimensions == 0 {
            return Err(StorageError::InvalidVectorDimensions { dimensions });
        }
        for chunk in chunks {
            validate_vector(chunk.vector, dimensions)?;
        }
        self.publish_job(job, chunks, Some((model_id, dimensions)), false)
    }

    /// Publishes chunks and schedules durable memory extraction in the same transaction.
    ///
    /// # Errors
    /// Returns an error for invalid vectors, stale revisions, or failed persistence.
    pub fn complete_embedded_job_with_memory_extraction(
        &mut self,
        job: &ClaimedJob,
        chunks: &[EmbeddedChunk<'_>],
        model_id: &str,
        dimensions: usize,
    ) -> Result<(), StorageError> {
        if dimensions != 768 {
            return Err(StorageError::InvalidVectorDimensions { dimensions });
        }
        for chunk in chunks {
            validate_vector(chunk.vector, dimensions)?;
        }
        self.publish_job(job, chunks, Some((model_id, dimensions)), true)
    }

    fn publish_job(
        &mut self,
        job: &ClaimedJob,
        chunks: &[EmbeddedChunk<'_>],
        embedding: Option<(&str, usize)>,
        schedule_memory_extraction: bool,
    ) -> Result<(), StorageError> {
        let tx = self.connection.transaction().map_err(StorageError::Write)?;
        let current_revision: Option<i64> = tx
            .query_row(
                "SELECT revision FROM documents WHERE id=?1",
                [&job.document_id],
                |row| row.get(0),
            )
            .optional()
            .map_err(StorageError::Read)?;
        if current_revision != Some(job.revision) {
            return Err(StorageError::StaleRevision {
                document_id: job.document_id.clone(),
                claimed: job.revision,
                current: current_revision,
            });
        }
        let mut existing = HashMap::new();
        {
            let mut statement = tx
                .prepare(
                    "SELECT ordinal, content, stable_id FROM document_chunks WHERE document_id=?1",
                )
                .map_err(StorageError::Read)?;
            let rows = statement
                .query_map([&job.document_id], |row| {
                    Ok((
                        row.get::<_, usize>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, String>(2)?,
                    ))
                })
                .map_err(StorageError::Read)?;
            for row in rows {
                let (ordinal, content, stable_id) = row.map_err(StorageError::Read)?;
                existing.insert((ordinal, content), stable_id);
            }
        }
        tx.execute(
            "DELETE FROM document_chunks WHERE document_id=?1",
            [&job.document_id],
        )
        .map_err(StorageError::Write)?;
        for (ordinal, chunk) in chunks.iter().enumerate() {
            let stable_id = existing
                .remove(&(ordinal, chunk.content.to_owned()))
                .map_or_else(generate_id, Ok)?;
            tx.execute(
                "INSERT INTO document_chunks (document_id, ordinal, content, stable_id) VALUES (?1, ?2, ?3, ?4)",
                params![job.document_id, ordinal, chunk.content, stable_id],
            )
            .map_err(StorageError::Write)?;
            if let Some((model_id, dimensions)) = embedding {
                let chunk_id = tx.last_insert_rowid();
                tx.execute(
                    "INSERT INTO chunk_embeddings (chunk_id, model_id, dimensions, vector) VALUES (?1, ?2, ?3, ?4)",
                    params![chunk_id, model_id, dimensions, vector_bytes(chunk.vector)],
                )
                .map_err(StorageError::Write)?;
            }
        }
        tx.execute(
            "UPDATE jobs SET status='done', last_error=NULL, updated_at=CURRENT_TIMESTAMP WHERE id=?1 AND revision=?2 AND status IN ('extracting','chunking','embedding','indexing')",
            params![job.id, job.revision],
        )
        .map_err(StorageError::Write)?;
        if schedule_memory_extraction {
            tx.execute(
                "INSERT INTO jobs (id, document_id, kind, status, revision) VALUES (?1, ?2, 'memory', 'queued', ?3)",
                params![generate_id()?, job.document_id, job.revision],
            )
            .map_err(StorageError::Write)?;
            tx.execute(
                "UPDATE documents SET status='indexing', updated_at=CURRENT_TIMESTAMP WHERE id=?1 AND revision=?2",
                params![job.document_id, job.revision],
            )
            .map_err(StorageError::Write)?;
        } else {
            tx.execute(
                "UPDATE documents SET status='done', updated_at=CURRENT_TIMESTAMP WHERE id=?1 AND revision=?2",
                params![job.document_id, job.revision],
            )
            .map_err(StorageError::Write)?;
        }
        tx.commit().map_err(StorageError::Write)
    }

    /// Advances a claimed job and its current document revision to a processing stage.
    ///
    /// # Errors
    /// Returns an error for an unsupported stage, stale revision, or failed transaction.
    pub fn mark_job_stage(
        &mut self,
        job: &ClaimedJob,
        stage: DocumentState,
    ) -> Result<(), StorageError> {
        let expected = match stage {
            DocumentState::Chunking => DocumentState::Extracting,
            DocumentState::Embedding => DocumentState::Chunking,
            DocumentState::Indexing => DocumentState::Embedding,
            _ => return Err(StorageError::InvalidJobStage(stage.as_str().to_owned())),
        };
        let tx = self.connection.transaction().map_err(StorageError::Write)?;
        let jobs = tx
            .execute(
                "UPDATE jobs SET status=?3, updated_at=CURRENT_TIMESTAMP WHERE id=?1 AND revision=?2 AND status=?4",
                params![job.id, job.revision, stage.as_str(), expected.as_str()],
            )
            .map_err(StorageError::Write)?;
        let documents = tx
            .execute(
                "UPDATE documents SET status=?3, updated_at=CURRENT_TIMESTAMP WHERE id=?1 AND revision=?2",
                params![job.document_id, job.revision, stage.as_str()],
            )
            .map_err(StorageError::Write)?;
        if jobs != 1 || documents != 1 {
            return Err(StorageError::StaleRevision {
                document_id: job.document_id.clone(),
                claimed: job.revision,
                current: None,
            });
        }
        tx.commit().map_err(StorageError::Write)
    }

    /// Marks a claimed job and its document failed while preserving prior indexed content.
    ///
    /// # Errors
    /// Returns an error if the failure state cannot be persisted atomically.
    pub fn fail_job(&mut self, job: &ClaimedJob, error: &str) -> Result<(), StorageError> {
        let tx = self.connection.transaction().map_err(StorageError::Write)?;
        tx.execute(
            "UPDATE jobs SET status='failed', last_error=?3, updated_at=CURRENT_TIMESTAMP WHERE id=?1 AND revision=?2",
            params![job.id, job.revision, error],
        )
        .map_err(StorageError::Write)?;
        tx.execute(
            "UPDATE documents SET status='failed', updated_at=CURRENT_TIMESTAMP WHERE id=?1 AND revision=?2",
            params![job.document_id, job.revision],
        )
        .map_err(StorageError::Write)?;
        tx.commit().map_err(StorageError::Write)
    }

    /// Deletes a claimed document whose extracted content produced no chunks.
    ///
    /// # Errors
    /// Returns an error if the document cleanup transaction cannot be committed.
    pub fn delete_empty_document(&mut self, job: &ClaimedJob) -> Result<(), StorageError> {
        let tx = self.connection.transaction().map_err(StorageError::Write)?;
        tx.execute(
            "DELETE FROM documents WHERE id=?1 AND revision=?2 AND status IN ('extracting','chunking')",
            params![job.document_id, job.revision],
        )
        .map_err(StorageError::Write)?;
        tx.commit().map_err(StorageError::Write)
    }
}
