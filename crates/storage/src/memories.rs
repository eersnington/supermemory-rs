use super::{
    HashMap, MemoryProposal, MemoryRecord, OptionalExtension, Storage, StorageError, Value,
    active_memory_ids_by_content, deduplicate_memories, generate_id, json, normalized_memory,
    params, read_memory, read_memory_by_id, resolve_memory_parents, valid_future_datetime,
    validate_vector, vector_bytes,
};

impl Storage {
    /// Reconciles extracted memories, lineage, sources, and vectors in one transaction.
    ///
    /// # Errors
    /// Returns an error if proposals are malformed or persistence cannot commit atomically.
    pub fn reconcile_memories(
        &mut self,
        document_id: &str,
        container_tag: &str,
        proposals: &[MemoryProposal],
        model_id: &str,
        dimensions: usize,
    ) -> Result<Vec<MemoryRecord>, StorageError> {
        let org_id = self.local_org_id.clone();
        self.reconcile_memories_for(
            &org_id,
            document_id,
            None,
            None,
            container_tag,
            proposals,
            model_id,
            dimensions,
        )
    }

    /// Reconciles extracted memories in an explicit organization.
    ///
    /// # Errors
    /// Returns an error if proposals or persistence are invalid.
    #[expect(
        clippy::too_many_arguments,
        clippy::too_many_lines,
        reason = "revision-gated proposal publication and completion share one transaction"
    )]
    pub fn reconcile_memories_for(
        &mut self,
        org_id: &str,
        document_id: &str,
        expected_revision: Option<i64>,
        completion_job_id: Option<&str>,
        container_tag: &str,
        proposals: &[MemoryProposal],
        model_id: &str,
        dimensions: usize,
    ) -> Result<Vec<MemoryRecord>, StorageError> {
        for proposal in proposals {
            validate_vector(&proposal.vector, dimensions)?;
            if proposal.temporary_id.is_empty() || proposal.content.trim().is_empty() {
                return Err(StorageError::InvalidMemoryProposal);
            }
        }
        let org_id = org_id.to_owned();
        let tx = self.connection.transaction().map_err(StorageError::Write)?;
        let source_revision: Option<i64> = tx
            .query_row(
                "SELECT revision FROM documents WHERE id=?1 AND org_id=?2",
                params![document_id, org_id],
                |row| row.get(0),
            )
            .optional()
            .map_err(StorageError::Read)?;
        let Some(source_revision) = source_revision else {
            return Err(StorageError::UnknownMemorySource(document_id.to_owned()));
        };
        if let Some(expected_revision) = expected_revision
            && expected_revision != source_revision
        {
            return Err(StorageError::StaleRevision {
                document_id: document_id.to_owned(),
                claimed: expected_revision,
                current: Some(source_revision),
            });
        }
        let mut temporary_ids = HashMap::new();
        let mut exact_memories = active_memory_ids_by_content(&tx, &org_id, container_tag)?;
        let mut record_ids = Vec::new();
        for proposal in proposals {
            let normalized_content = normalized_memory(&proposal.content);
            if let Some(existing_id) = exact_memories.get(&normalized_content).cloned() {
                tx.execute(
                    "INSERT OR IGNORE INTO memory_sources (memory_id, document_id) VALUES (?1, ?2)",
                    params![existing_id, document_id],
                )
                .map_err(StorageError::Write)?;
                temporary_ids.insert(proposal.temporary_id.clone(), existing_id.clone());
                record_ids.push(existing_id);
                continue;
            }
            let parents = resolve_memory_parents(
                &tx,
                &org_id,
                container_tag,
                &proposal.parents,
                &temporary_ids,
            )?;
            let primary = parents.first();
            let id = format!("mem_{}", generate_id()?);
            let root_memory_id = primary.map(|parent| {
                parent
                    .root_memory_id
                    .clone()
                    .unwrap_or_else(|| parent.id.clone())
            });
            let parent_memory_id = primary.map(|parent| parent.id.clone());
            let version = primary.map_or(1, |parent| parent.version + 1);
            let is_inferred =
                proposal.is_inferred || parents.iter().any(|parent| parent.relation == "derives");
            let forget_after = valid_future_datetime(&tx, proposal.forget_after.as_deref())?;
            tx.execute(
                "INSERT INTO memories (id, org_id, container_tag, content, metadata, is_inferred, is_static, root_memory_id, parent_memory_id, version, forget_after, forget_reason) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)",
                params![id, org_id, container_tag, proposal.content, json(&proposal.metadata)?, is_inferred, proposal.is_static, root_memory_id, parent_memory_id, version, forget_after, proposal.forget_reason],
            ).map_err(StorageError::Write)?;
            tx.execute(
                "INSERT INTO memory_sources (memory_id, document_id) VALUES (?1, ?2)",
                params![id, document_id],
            )
            .map_err(StorageError::Write)?;
            for parent in &parents {
                tx.execute(
                    "INSERT OR IGNORE INTO memory_relations (parent_id, child_id, relation) VALUES (?1, ?2, ?3)",
                    params![parent.id, id, parent.relation],
                )
                .map_err(StorageError::Write)?;
                if parent.relation == "updates" {
                    tx.execute(
                        "UPDATE memories SET is_latest=0, is_static=0, updated_at=CURRENT_TIMESTAMP WHERE id=?1",
                        [&parent.id],
                    )
                    .map_err(StorageError::Write)?;
                    tx.execute(
                        "UPDATE memory_embeddings SET active=0 WHERE memory_id=?1",
                        [&parent.id],
                    )
                    .map_err(StorageError::Write)?;
                }
            }
            tx.execute(
                "INSERT INTO memory_embeddings (memory_id, model_id, dimensions, vector) VALUES (?1, ?2, ?3, ?4)",
                params![id, model_id, dimensions, vector_bytes(&proposal.vector)],
            )
            .map_err(StorageError::Write)?;
            temporary_ids.insert(proposal.temporary_id.clone(), id.clone());
            exact_memories.insert(normalized_content, id.clone());
            record_ids.push(id);
        }
        let records = record_ids
            .iter()
            .map(|id| read_memory_by_id(&tx, id))
            .collect::<Result<Vec<_>, _>>()?;
        if let Some(job_id) = completion_job_id {
            let jobs = tx.execute(
                "UPDATE jobs SET status='done', extraction_result=NULL, last_error_kind=NULL, last_error=NULL, updated_at=CURRENT_TIMESTAMP WHERE id=?1 AND document_id=?2 AND revision=?3 AND status='extracting'",
                params![job_id, document_id, source_revision],
            ).map_err(StorageError::Write)?;
            let documents = tx.execute(
                "UPDATE documents SET status='done', updated_at=CURRENT_TIMESTAMP WHERE id=?1 AND revision=?2",
                params![document_id, source_revision],
            ).map_err(StorageError::Write)?;
            if jobs != 1 || documents != 1 {
                return Err(StorageError::StaleRevision {
                    document_id: document_id.to_owned(),
                    claimed: source_revision,
                    current: None,
                });
            }
        }
        tx.commit().map_err(StorageError::Write)?;
        Ok(records)
    }

    /// Returns active static profile memories in recovered profile order.
    ///
    /// # Errors
    /// Returns an error when stored memory data cannot be read.
    pub fn static_profile(&self, container_tag: &str) -> Result<Vec<MemoryRecord>, StorageError> {
        self.static_profile_for(&self.local_org_id, container_tag)
    }

    /// Returns static profile memories in an explicit organization.
    ///
    /// # Errors
    /// Returns an error when stored memory data cannot be read.
    pub fn static_profile_for(
        &self,
        org_id: &str,
        container_tag: &str,
    ) -> Result<Vec<MemoryRecord>, StorageError> {
        let mut statement = self.connection.prepare(
            "SELECT id, content, metadata, is_inferred, is_static, is_latest, is_forgotten, root_memory_id, parent_memory_id, version, forget_after, forget_reason, created_at, updated_at FROM memories WHERE org_id=?1 AND container_tag=?2 AND is_static=1 AND is_latest=1 AND is_forgotten=0 AND (forget_after IS NULL OR datetime(forget_after) > CURRENT_TIMESTAMP) ORDER BY (SELECT COUNT(*) FROM memory_sources WHERE memory_id=memories.id) DESC, updated_at DESC LIMIT 300",
        ).map_err(StorageError::Read)?;
        let rows = statement
            .query_map(params![org_id, container_tag], read_memory)
            .map_err(StorageError::Read)?
            .collect::<Result<Vec<_>, _>>()
            .map_err(StorageError::Read)?;
        Ok(deduplicate_memories(
            rows,
            100,
            &std::collections::HashSet::new(),
        ))
    }

    /// Returns active dynamic profile memories ordered by recency.
    ///
    /// # Errors
    /// Returns an error when stored memory data cannot be read.
    pub fn dynamic_profile(
        &self,
        container_tag: &str,
        static_memories: &[MemoryRecord],
    ) -> Result<Vec<MemoryRecord>, StorageError> {
        self.dynamic_profile_for(&self.local_org_id, container_tag, static_memories)
    }

    /// Returns dynamic profile memories in an explicit organization.
    ///
    /// # Errors
    /// Returns an error when stored memory data cannot be read.
    pub fn dynamic_profile_for(
        &self,
        org_id: &str,
        container_tag: &str,
        static_memories: &[MemoryRecord],
    ) -> Result<Vec<MemoryRecord>, StorageError> {
        let mut statement = self.connection.prepare(
            "SELECT id, content, metadata, is_inferred, is_static, is_latest, is_forgotten, root_memory_id, parent_memory_id, version, forget_after, forget_reason, created_at, updated_at FROM memories WHERE org_id=?1 AND container_tag=?2 AND is_static=0 AND is_latest=1 AND is_forgotten=0 AND (forget_after IS NULL OR datetime(forget_after) > CURRENT_TIMESTAMP) ORDER BY updated_at DESC LIMIT 300",
        ).map_err(StorageError::Read)?;
        let rows = statement
            .query_map(params![org_id, container_tag], read_memory)
            .map_err(StorageError::Read)?
            .collect::<Result<Vec<_>, _>>()
            .map_err(StorageError::Read)?;
        let excluded = static_memories
            .iter()
            .map(|memory| normalized_memory(&memory.memory))
            .collect();
        Ok(deduplicate_memories(rows, 100, &excluded))
    }

    /// Returns memories assigned to a profile bucket.
    ///
    /// # Errors
    /// Returns an error when stored memory data cannot be read.
    pub fn bucket_profile(
        &self,
        container_tag: &str,
        bucket: &str,
    ) -> Result<Vec<MemoryRecord>, StorageError> {
        self.bucket_profile_for(&self.local_org_id, container_tag, bucket)
    }

    /// Returns bucket memories in an explicit organization.
    ///
    /// # Errors
    /// Returns an error when stored memory data cannot be read.
    pub fn bucket_profile_for(
        &self,
        org_id: &str,
        container_tag: &str,
        bucket: &str,
    ) -> Result<Vec<MemoryRecord>, StorageError> {
        let static_memories = self.static_profile_for(org_id, container_tag)?;
        let dynamic = self.dynamic_profile_for(org_id, container_tag, &static_memories)?;
        Ok(static_memories
            .into_iter()
            .chain(dynamic)
            .filter(|memory| {
                memory
                    .metadata
                    .get("buckets")
                    .and_then(Value::as_array)
                    .is_some_and(|buckets| {
                        buckets.iter().any(|value| value.as_str() == Some(bucket))
                    })
            })
            .take(100)
            .collect())
    }

    /// Soft-forgets one active memory while preserving graph and source history.
    ///
    /// # Errors
    /// Returns an error if the memory is absent or the transition cannot commit.
    pub fn forget_memory(
        &mut self,
        id: Option<&str>,
        content: Option<&str>,
        container_tag: &str,
        reason: Option<&str>,
    ) -> Result<String, StorageError> {
        let org_id = self.local_org_id.clone();
        self.forget_memory_for(&org_id, id, content, container_tag, reason)
    }

    /// Soft-forgets one active memory in an explicit organization.
    ///
    /// # Errors
    /// Returns an error if the memory is absent or persistence fails.
    pub fn forget_memory_for(
        &mut self,
        org_id: &str,
        id: Option<&str>,
        content: Option<&str>,
        container_tag: &str,
        reason: Option<&str>,
    ) -> Result<String, StorageError> {
        let tx = self.connection.transaction().map_err(StorageError::Write)?;
        let memory_id: Option<String> = if let Some(id) = id {
            tx.query_row(
                "SELECT id FROM memories WHERE id=?1 AND org_id=?2 AND container_tag=?3 AND is_forgotten=0 AND (forget_after IS NULL OR datetime(forget_after) > CURRENT_TIMESTAMP)",
                params![id, org_id, container_tag],
                |row| row.get(0),
            )
            .optional()
            .map_err(StorageError::Read)?
        } else if let Some(content) = content {
            tx.query_row(
                "SELECT id FROM memories WHERE content=?1 AND org_id=?2 AND container_tag=?3 AND is_forgotten=0 AND (forget_after IS NULL OR datetime(forget_after) > CURRENT_TIMESTAMP) ORDER BY rowid LIMIT 1",
                params![content, org_id, container_tag],
                |row| row.get(0),
            )
            .optional()
            .map_err(StorageError::Read)?
        } else {
            None
        };
        let memory_id = memory_id.ok_or(StorageError::MemoryNotFound)?;
        tx.execute(
            "UPDATE memories SET is_forgotten=1, is_latest=0, is_static=0, forget_after=CURRENT_TIMESTAMP, forget_reason=?2, updated_at=CURRENT_TIMESTAMP WHERE id=?1",
            params![memory_id, reason.unwrap_or("user_requested")],
        )
        .map_err(StorageError::Write)?;
        tx.execute(
            "UPDATE memory_embeddings SET active=0 WHERE memory_id=?1",
            [&memory_id],
        )
        .map_err(StorageError::Write)?;
        tx.commit().map_err(StorageError::Write)?;
        Ok(memory_id)
    }
}
