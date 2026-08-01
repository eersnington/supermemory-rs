use super::{
    ClaimedMemoryJob, Duration, ExistingMemory, Map, OptionalExtension, Storage, StorageError,
    TransactionBehavior, Value, mark_memory_job_stale, params, parse_json,
};

impl Storage {
    /// Claims one provider-extraction job and increments its durable attempt count.
    ///
    /// # Errors
    /// Returns an error when the claim transaction cannot be completed.
    pub fn claim_memory_job(&mut self) -> Result<Option<ClaimedMemoryJob>, StorageError> {
        let tx = self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(StorageError::Write)?;
        let job = tx
            .query_row(
                "SELECT jobs.id, documents.id, documents.content, jobs.revision, documents.org_id, documents.container_tags, documents.metadata, jobs.extraction_result, jobs.attempts FROM jobs JOIN documents ON documents.id=jobs.document_id WHERE jobs.kind='memory' AND jobs.status='queued' AND jobs.revision=documents.revision AND jobs.available_at<=CURRENT_TIMESTAMP ORDER BY jobs.created_at, jobs.rowid LIMIT 1",
                [],
                |row| {
                    let tags: Vec<String> = parse_json(&row.get::<_, String>(5)?, 5)?;
                    let metadata: Map<String, Value> = parse_json(&row.get::<_, String>(6)?, 6)?;
                    Ok(ClaimedMemoryJob {
                        id: row.get(0)?,
                        document_id: row.get(1)?,
                        content: row.get(2)?,
                        revision: row.get(3)?,
                        organization_id: row.get(4)?,
                        container_tag: tags.into_iter().next().unwrap_or_else(|| "sm_project_default".to_owned()),
                        document_date: metadata.get("date").and_then(Value::as_str).map(str::to_owned),
                        extraction_result: row.get(7)?,
                        attempts: row.get::<_, i64>(8)?.saturating_add(1),
                    })
                },
            )
            .optional()
            .map_err(StorageError::Read)?;
        let Some(job) = job else {
            tx.commit().map_err(StorageError::Write)?;
            return Ok(None);
        };
        let updated = tx
            .execute(
                "UPDATE jobs SET status='extracting', attempts=attempts+1, updated_at=CURRENT_TIMESTAMP WHERE id=?1 AND revision=?2 AND status='queued'",
                params![job.id, job.revision],
            )
            .map_err(StorageError::Write)?;
        if updated != 1 {
            return Err(StorageError::StaleRevision {
                document_id: job.document_id,
                claimed: job.revision,
                current: None,
            });
        }
        tx.commit().map_err(StorageError::Write)?;
        Ok(Some(job))
    }

    /// Reports whether document or memory publication work is still active.
    ///
    /// # Errors
    /// Returns an error when job state cannot be read.
    pub fn has_active_processing_jobs(&self) -> Result<bool, StorageError> {
        self.connection
            .query_row(
                "SELECT EXISTS(SELECT 1 FROM jobs WHERE status IN ('queued','extracting','chunking','embedding','indexing'))",
                [],
                |row| row.get(0),
            )
            .map_err(StorageError::Read)
    }

    /// Returns current active memories used as provider extraction context.
    ///
    /// # Errors
    /// Returns an error when stored memories cannot be read.
    pub fn existing_memories_for_extraction(
        &self,
        org_id: &str,
        container_tag: &str,
    ) -> Result<Vec<ExistingMemory>, StorageError> {
        let mut statement = self.connection.prepare(
            "SELECT id, content FROM memories WHERE org_id=?1 AND container_tag=?2 AND is_latest=1 AND is_forgotten=0 AND (forget_after IS NULL OR datetime(forget_after)>CURRENT_TIMESTAMP) ORDER BY updated_at DESC, id LIMIT 100",
        ).map_err(StorageError::Read)?;
        statement
            .query_map(params![org_id, container_tag], |row| {
                Ok(ExistingMemory {
                    id: row.get(0)?,
                    content: row.get(1)?,
                })
            })
            .map_err(StorageError::Read)?
            .collect::<Result<Vec<_>, _>>()
            .map_err(StorageError::Read)
    }

    /// Persists provider output separately from retry error text.
    ///
    /// # Errors
    /// Returns an error when the revision is stale or the write fails.
    pub fn cache_memory_extraction(
        &mut self,
        job: &ClaimedMemoryJob,
        result: &str,
    ) -> Result<(), StorageError> {
        let tx = self.connection.transaction().map_err(StorageError::Write)?;
        let updated = tx.execute(
            "UPDATE jobs SET extraction_result=?3, updated_at=CURRENT_TIMESTAMP WHERE id=?1 AND revision=?2 AND status='extracting' AND EXISTS(SELECT 1 FROM documents WHERE id=jobs.document_id AND revision=jobs.revision)",
            params![job.id, job.revision, result],
        ).map_err(StorageError::Write)?;
        if updated == 1 {
            return tx.commit().map_err(StorageError::Write);
        }
        mark_memory_job_stale(&tx, job)?;
        tx.commit().map_err(StorageError::Write)?;
        Err(StorageError::StaleRevision {
            document_id: job.document_id.clone(),
            claimed: job.revision,
            current: None,
        })
    }

    /// Marks a memory job complete only while its claimed document revision is current.
    ///
    /// # Errors
    /// Returns an error when the completion transaction fails.
    pub fn complete_memory_job(&mut self, job: &ClaimedMemoryJob) -> Result<(), StorageError> {
        let tx = self.connection.transaction().map_err(StorageError::Write)?;
        let current: Option<i64> = tx
            .query_row(
                "SELECT revision FROM documents WHERE id=?1",
                [&job.document_id],
                |row| row.get(0),
            )
            .optional()
            .map_err(StorageError::Read)?;
        if current != Some(job.revision) {
            tx.execute(
                "UPDATE jobs SET status='failed', last_error_kind='stale_revision', last_error=NULL, updated_at=CURRENT_TIMESTAMP WHERE id=?1 AND revision=?2",
                params![job.id, job.revision],
            )
            .map_err(StorageError::Write)?;
            tx.commit().map_err(StorageError::Write)?;
            return Ok(());
        }
        let jobs = tx.execute(
            "UPDATE jobs SET status='done', extraction_result=NULL, last_error_kind=NULL, last_error=NULL, updated_at=CURRENT_TIMESTAMP WHERE id=?1 AND revision=?2 AND status='extracting'",
            params![job.id, job.revision],
        ).map_err(StorageError::Write)?;
        let documents = tx.execute(
            "UPDATE documents SET status='done', updated_at=CURRENT_TIMESTAMP WHERE id=?1 AND revision=?2",
            params![job.document_id, job.revision],
        ).map_err(StorageError::Write)?;
        if jobs != 1 || documents != 1 {
            return Err(StorageError::StaleRevision {
                document_id: job.document_id.clone(),
                claimed: job.revision,
                current,
            });
        }
        tx.commit().map_err(StorageError::Write)
    }

    /// Persists a bounded retry or terminal memory-extraction failure.
    ///
    /// # Errors
    /// Returns an error when the revision is stale or retry state cannot be committed.
    pub fn retry_memory_job(
        &mut self,
        job: &ClaimedMemoryJob,
        error_kind: &str,
        error: &str,
        retry_delay: Option<Duration>,
    ) -> Result<(), StorageError> {
        const MAX_ATTEMPTS: i64 = 4;
        let tx = self.connection.transaction().map_err(StorageError::Write)?;
        let current_revision = tx
            .query_row(
                "SELECT revision FROM documents WHERE id=?1",
                [&job.document_id],
                |row| row.get::<_, i64>(0),
            )
            .optional()
            .map_err(StorageError::Read)?;
        if current_revision != Some(job.revision) {
            mark_memory_job_stale(&tx, job)?;
            return tx.commit().map_err(StorageError::Write);
        }
        let terminal = retry_delay.is_none() || job.attempts >= MAX_ATTEMPTS;
        let delay_seconds = retry_delay
            .map(|delay| delay.as_secs().max(1))
            .unwrap_or_default();
        let modifier = format!("+{delay_seconds} seconds");
        let status = if terminal { "failed" } else { "queued" };
        let updated = tx.execute(
            "UPDATE jobs SET status=?3, last_error_kind=?4, last_error=?5, available_at=datetime(CURRENT_TIMESTAMP, ?6), updated_at=CURRENT_TIMESTAMP WHERE id=?1 AND revision=?2 AND status='extracting'",
            params![job.id, job.revision, status, error_kind, error, modifier],
        ).map_err(StorageError::Write)?;
        if updated != 1 {
            return Err(StorageError::StaleRevision {
                document_id: job.document_id.clone(),
                claimed: job.revision,
                current: None,
            });
        }
        if terminal {
            tx.execute(
                "UPDATE documents SET status='failed', updated_at=CURRENT_TIMESTAMP WHERE id=?1 AND revision=?2",
                params![job.document_id, job.revision],
            )
            .map_err(StorageError::Write)?;
        }
        tx.commit().map_err(StorageError::Write)
    }
}
