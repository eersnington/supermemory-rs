use super::{
    Document, OptionalExtension, Storage, StorageError, UpsertDocument, UpsertResult,
    apply_existing, content_hash, find_custom, find_duplicate, insert_new, params, query_one,
    read_document, sanitize_content,
};

/// A stable chunk returned in document order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DocumentChunk {
    pub id: String,
    pub content: String,
    pub ordinal: usize,
}

impl Storage {
    /// Sanitizes, identifies, updates, and if needed queues a document atomically.
    ///
    /// # Errors
    /// Returns an error when randomness, serialization, or the transaction fails.
    pub fn upsert_document(&mut self, input: UpsertDocument) -> Result<UpsertResult, StorageError> {
        let org_id = self.local_org_id.clone();
        self.upsert_document_for(&org_id, input)
    }

    /// Applies document identity and upsert rules in an explicit organization.
    ///
    /// # Errors
    /// Returns an error when identity resolution or persistence fails.
    pub fn upsert_document_for(
        &mut self,
        org_id: &str,
        mut input: UpsertDocument,
    ) -> Result<UpsertResult, StorageError> {
        input.content = sanitize_content(&input.content);
        let hash = content_hash(&input.content);
        let org_id = org_id.to_owned();
        let tx = self.connection.transaction().map_err(StorageError::Write)?;

        let existing = if let Some(custom_id) = input.custom_id.as_deref() {
            find_custom(&tx, &org_id, custom_id, &input.container_tags)?
        } else {
            find_duplicate(&tx, &org_id, &hash, &input.metadata, &input.container_tags)?
        };

        let result = if let Some(existing) = existing {
            apply_existing(&tx, &existing, &input, &hash)?
        } else {
            insert_new(&tx, &org_id, &input, &hash)?
        };
        tx.commit().map_err(StorageError::Write)?;
        Ok(result)
    }

    /// Finds by internal ID first, then custom ID, within the local organization.
    ///
    /// # Errors
    /// Returns an error when the query fails or stored JSON is malformed.
    pub fn find_document(&self, identifier: &str) -> Result<Option<Document>, StorageError> {
        self.find_document_for(&self.local_org_id, identifier)
    }

    /// Finds a document by internal then custom ID in an explicit organization.
    ///
    /// # Errors
    /// Returns an error when stored data cannot be read.
    pub fn find_document_for(
        &self,
        org_id: &str,
        identifier: &str,
    ) -> Result<Option<Document>, StorageError> {
        if let Some(found) = query_one(&self.connection, "id = ?2", org_id, identifier)? {
            return Ok(Some(found.document));
        }
        Ok(
            query_one(&self.connection, "custom_id = ?2", org_id, identifier)?
                .map(|found| found.document),
        )
    }

    /// Lists documents in creation order for one organization.
    ///
    /// # Errors
    /// Returns an error when persisted JSON cannot be decoded or the query fails.
    pub fn list_documents_for(
        &self,
        org_id: &str,
        limit: usize,
        offset: usize,
        container_tag: Option<&str>,
    ) -> Result<Vec<Document>, StorageError> {
        let mut statement = self.connection.prepare(
            "SELECT id, content, content_hash, custom_id, status, container_tags, entity_context, metadata, task_type, filepath, filter_by_metadata, dreaming, created_at, updated_at FROM documents WHERE org_id=?1 AND (?2 IS NULL OR EXISTS(SELECT 1 FROM json_each(documents.container_tags) WHERE value=?2)) ORDER BY created_at DESC, rowid DESC LIMIT ?3 OFFSET ?4",
        ).map_err(StorageError::Read)?;
        statement
            .query_map(params![org_id, container_tag, limit, offset], read_document)
            .map_err(StorageError::Read)?
            .map(|row| {
                row.map(|document| document.document)
                    .map_err(StorageError::Read)
            })
            .collect()
    }

    /// Deletes one document by internal or custom ID and cancels all pending jobs atomically.
    ///
    /// # Errors
    /// Returns an error when storage cannot find or delete the organization-scoped document.
    pub fn delete_document_for(
        &mut self,
        org_id: &str,
        identifier: &str,
    ) -> Result<bool, StorageError> {
        let tx = self.connection.transaction().map_err(StorageError::Write)?;
        let id: Option<String> = tx
            .query_row(
                "SELECT id FROM documents WHERE org_id=?1 AND (id=?2 OR custom_id=?2) ORDER BY CASE WHEN id=?2 THEN 0 ELSE 1 END, rowid LIMIT 1",
                params![org_id, identifier],
                |row| row.get(0),
            )
            .optional()
            .map_err(StorageError::Read)?;
        let Some(id) = id else {
            tx.commit().map_err(StorageError::Write)?;
            return Ok(false);
        };
        tx.execute(
            "DELETE FROM documents WHERE id=?1 AND org_id=?2",
            params![id, org_id],
        )
        .map_err(StorageError::Write)?;
        tx.commit().map_err(StorageError::Write)?;
        Ok(true)
    }

    /// Returns a document's current revision chunks in their original order.
    ///
    /// # Errors
    /// Returns an error if the document is outside the organization or chunks cannot be read.
    pub fn document_chunks_for(
        &self,
        org_id: &str,
        identifier: &str,
    ) -> Result<Option<Vec<DocumentChunk>>, StorageError> {
        let id: Option<String> = self.connection.query_row(
            "SELECT id FROM documents WHERE org_id=?1 AND (id=?2 OR custom_id=?2) ORDER BY CASE WHEN id=?2 THEN 0 ELSE 1 END, rowid LIMIT 1",
            params![org_id, identifier],
            |row| row.get(0),
        ).optional().map_err(StorageError::Read)?;
        let Some(id) = id else {
            return Ok(None);
        };
        let mut statement = self.connection.prepare(
            "SELECT stable_id, content, ordinal FROM document_chunks WHERE document_id=?1 ORDER BY ordinal, id",
        ).map_err(StorageError::Read)?;
        let chunks = statement
            .query_map([id], |row| {
                Ok(DocumentChunk {
                    id: row.get(0)?,
                    content: row.get(1)?,
                    ordinal: row.get(2)?,
                })
            })
            .map_err(StorageError::Read)?
            .collect::<Result<Vec<_>, _>>()
            .map_err(StorageError::Read)?;
        Ok(Some(chunks))
    }
}
