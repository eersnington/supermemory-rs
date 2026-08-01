use super::{
    Document, Storage, StorageError, UpsertDocument, UpsertResult, apply_existing, content_hash,
    find_custom, find_duplicate, insert_new, query_one, sanitize_content,
};

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
}
