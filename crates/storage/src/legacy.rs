use super::{
    ApiKeyIdentity, BufRead, BufReader, Digest, HashMap, LegacyImportReport, Map, Path, Sha256,
    Storage, StorageError, Value, bool_field, compatible_document_status, compatible_relation,
    copy_optional_metadata, field, integer_field, json, json_field, object_field, optional_field,
    params, required_string, table_rows, value_field, vector_bytes, vector_field,
};

impl Storage {
    /// Imports a stable JSONL export from the read-only legacy database.
    ///
    /// Existing IDs are retained and duplicate rows make the operation idempotent.
    ///
    /// # Errors
    /// Returns an error and rolls back the complete import if parsing or validation fails.
    pub fn import_legacy_export(
        &mut self,
        path: &Path,
    ) -> Result<LegacyImportReport, StorageError> {
        let file = std::fs::File::open(path).map_err(|source| StorageError::ReadLegacyExport {
            path: path.to_path_buf(),
            source,
        })?;
        let mut tables: HashMap<String, Vec<Map<String, Value>>> = HashMap::new();
        let mut complete = false;
        for (index, line) in BufReader::new(file).lines().enumerate() {
            let line = line.map_err(|source| StorageError::ReadLegacyExport {
                path: path.to_path_buf(),
                source,
            })?;
            let value: Value = serde_json::from_str(&line).map_err(|source| {
                StorageError::MalformedLegacyExport {
                    line: index + 1,
                    source,
                }
            })?;
            match value.get("type").and_then(Value::as_str) {
                Some("row") => {
                    let table = required_string(&value, "table")?.to_owned();
                    let row = value
                        .get("row")
                        .and_then(Value::as_object)
                        .cloned()
                        .ok_or(StorageError::InvalidLegacyRow { line: index + 1 })?;
                    tables.entry(table).or_default().push(row);
                }
                Some("complete") => complete = true,
                Some("manifest" | "table") => {}
                _ => return Err(StorageError::InvalidLegacyRow { line: index + 1 }),
            }
        }
        if !complete {
            return Err(StorageError::IncompleteLegacyExport);
        }
        self.import_legacy_tables(&tables)
    }

    #[expect(
        clippy::too_many_lines,
        reason = "foreign-key import order is kept visible in one transaction"
    )]
    fn import_legacy_tables(
        &mut self,
        tables: &HashMap<String, Vec<Map<String, Value>>>,
    ) -> Result<LegacyImportReport, StorageError> {
        let tx = self.connection.transaction().map_err(StorageError::Write)?;
        let mut report = LegacyImportReport::default();
        for row in table_rows(tables, "organization") {
            let id = field(row, "id")?;
            let slug = field(row, "slug")?;
            tx.execute(
                "UPDATE organizations SET slug=slug || '-rs-' || substr(id, 1, 6) WHERE slug=?1 AND id<>?2 AND id=?3 AND NOT EXISTS(SELECT 1 FROM documents WHERE org_id=organizations.id)",
                params![slug, id, self.local_org_id],
            ).map_err(StorageError::Write)?;
            report.organizations += tx.execute(
                "INSERT OR IGNORE INTO organizations (id, slug, created_at, name, metadata) VALUES (?1, ?2, ?3, ?4, ?5)",
                params![id, slug, field(row, "created_at")?, optional_field(row, "name"), json_field(row, "metadata", &json!({}))?],
            ).map_err(StorageError::Write)?;
        }
        let mut spaces = HashMap::new();
        for row in table_rows(tables, "space") {
            let id = field(row, "id")?.to_owned();
            let container_tag = field(row, "container_tag")?.to_owned();
            spaces.insert(id.clone(), container_tag.clone());
            report.spaces += tx.execute(
                "INSERT OR IGNORE INTO spaces (id, org_id, container_tag, entity_context, name, description, metadata, profile_buckets, created_at, updated_at) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
                params![id, field(row, "org_id")?, container_tag, optional_field(row, "entity_context"), optional_field(row, "name"), optional_field(row, "description"), json_field(row, "metadata", &json!({}))?, json_field(row, "profile_buckets", &json!([]))?, field(row, "created_at")?, field(row, "updated_at")?],
            ).map_err(StorageError::Write)?;
        }
        for row in table_rows(tables, "document") {
            let mut metadata = object_field(row, "metadata")?;
            copy_optional_metadata(row, &mut metadata, "title");
            copy_optional_metadata(row, &mut metadata, "summary");
            copy_optional_metadata(row, &mut metadata, "type");
            let status = compatible_document_status(optional_field(row, "status"));
            let task_type = match optional_field(row, "task_type") {
                Some("superrag") => "superrag",
                _ => "memory",
            };
            report.documents += tx.execute(
                "INSERT OR IGNORE INTO documents (id, org_id, content, content_hash, custom_id, status, container_tags, entity_context, metadata, task_type, filepath, filter_by_metadata, dreaming, created_at, updated_at, title, summary, document_type, source, url, user_id) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, NULL, ?8, ?9, ?10, '{}', 'dynamic', ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18)",
                params![field(row, "id")?, field(row, "org_id")?, optional_field(row, "content").unwrap_or_default(), field(row, "content_hash")?, optional_field(row, "custom_id"), status, json_field(row, "container_tags", &json!(["sm_project_default"]))?, json(&metadata)?, task_type, optional_field(row, "filepath"), field(row, "created_at")?, field(row, "updated_at")?, optional_field(row, "title"), optional_field(row, "summary"), optional_field(row, "type"), optional_field(row, "source"), optional_field(row, "url"), optional_field(row, "user_id")],
            ).map_err(StorageError::Write)?;
        }
        for row in table_rows(tables, "chunk") {
            let changed = tx.execute(
                "INSERT OR IGNORE INTO document_chunks (document_id, ordinal, content, stable_id) VALUES (?1, ?2, ?3, ?4)",
                params![field(row, "document_id")?, integer_field(row, "position")?, field(row, "content")?, field(row, "id")?],
            ).map_err(StorageError::Write)?;
            report.chunks += changed;
            if changed == 1
                && let Some(vector) = vector_field(row, "embedding")?
            {
                let chunk_id = tx.last_insert_rowid();
                tx.execute(
                    "INSERT INTO chunk_embeddings (chunk_id, model_id, dimensions, vector) VALUES (?1, ?2, ?3, ?4)",
                    params![chunk_id, optional_field(row, "embedding_model").unwrap_or("legacy"), vector.len(), vector_bytes(&vector)],
                ).map_err(StorageError::Write)?;
            }
        }
        let mut pending_parents = Vec::new();
        for row in table_rows(tables, "memory_entry") {
            let id = field(row, "id")?.to_owned();
            let container_tag = optional_field(row, "space_id")
                .and_then(|space| spaces.get(space))
                .cloned()
                .unwrap_or_else(|| "sm_project_default".to_owned());
            let mut metadata = object_field(row, "metadata")?;
            metadata.insert(
                "buckets".to_owned(),
                value_field(row, "buckets")
                    .cloned()
                    .unwrap_or_else(|| json!([])),
            );
            report.memories += tx.execute(
                "INSERT OR IGNORE INTO memories (id, org_id, container_tag, content, metadata, is_inferred, is_static, is_latest, is_forgotten, root_memory_id, parent_memory_id, version, forget_after, forget_reason, created_at, updated_at, source_count) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, NULL, ?11, ?12, ?13, ?14, ?15, ?16)",
                params![id, field(row, "org_id")?, container_tag, field(row, "memory")?, json(&metadata)?, bool_field(row, "is_inference"), bool_field(row, "is_static"), bool_field(row, "is_latest"), bool_field(row, "is_forgotten"), optional_field(row, "root_memory_id"), integer_field(row, "version")?.max(1), optional_field(row, "forget_after"), optional_field(row, "forget_reason"), field(row, "created_at")?, field(row, "updated_at")?, integer_field(row, "source_count")?.max(0)],
            ).map_err(StorageError::Write)?;
            if let Some(parent) = optional_field(row, "parent_memory_id") {
                pending_parents.push((id.clone(), parent.to_owned()));
            }
            if let Some(vector) = vector_field(row, "memory_embedding")? {
                tx.execute(
                    "INSERT OR IGNORE INTO memory_embeddings (memory_id, model_id, dimensions, vector, active) VALUES (?1, ?2, ?3, ?4, ?5)",
                    params![id, optional_field(row, "memory_embedding_model").unwrap_or("legacy"), vector.len(), vector_bytes(&vector), bool_field(row, "is_latest") && !bool_field(row, "is_forgotten")],
                ).map_err(StorageError::Write)?;
            }
        }
        for (id, parent) in pending_parents {
            tx.execute(
                "UPDATE memories SET parent_memory_id=?2 WHERE id=?1 AND EXISTS(SELECT 1 FROM memories WHERE id=?2)",
                params![id, parent],
            ).map_err(StorageError::Write)?;
        }
        for row in table_rows(tables, "memory_relations") {
            let relation = compatible_relation(field(row, "relation")?);
            report.relations += tx.execute(
                "INSERT OR IGNORE INTO memory_relations (parent_id, child_id, relation) SELECT ?1, ?2, ?3 WHERE EXISTS(SELECT 1 FROM memories WHERE id=?1) AND EXISTS(SELECT 1 FROM memories WHERE id=?2)",
                params![field(row, "from_id")?, field(row, "to_id")?, relation],
            ).map_err(StorageError::Write)?;
        }
        for row in table_rows(tables, "memory_document_source") {
            report.sources += tx.execute(
                "INSERT OR IGNORE INTO memory_sources (memory_id, document_id, created_at) SELECT ?1, ?2, ?3 WHERE EXISTS(SELECT 1 FROM memories WHERE id=?1) AND EXISTS(SELECT 1 FROM documents WHERE id=?2)",
                params![field(row, "memory_entry_id")?, field(row, "document_id")?, field(row, "added_at")?],
            ).map_err(StorageError::Write)?;
        }
        let member_organizations = table_rows(tables, "member")
            .iter()
            .filter_map(|row| {
                optional_field(row, "user_id")
                    .zip(optional_field(row, "organization_id"))
                    .map(|(user, org)| (user.to_owned(), org.to_owned()))
            })
            .collect::<HashMap<_, _>>();
        let imported_organizations = table_rows(tables, "organization")
            .iter()
            .filter_map(|row| optional_field(row, "id"))
            .collect::<std::collections::HashSet<_>>();
        for row in table_rows(tables, "apikey") {
            let Some(key) = optional_field(row, "key") else {
                continue;
            };
            let organization_id = optional_field(row, "reference_id")
                .and_then(|reference| member_organizations.get(reference))
                .filter(|organization| imported_organizations.contains(organization.as_str()));
            report.api_keys += tx.execute(
                "INSERT OR IGNORE INTO api_keys (id, org_id, key_hash, name, enabled, expires_at, created_at, updated_at) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
                params![field(row, "id")?, organization_id, Sha256::digest(key.as_bytes()).as_slice(), optional_field(row, "name"), bool_field(row, "enabled"), optional_field(row, "expires_at"), field(row, "created_at")?, optional_field(row, "updated_at")],
            ).map_err(StorageError::Write)?;
        }
        tx.commit().map_err(StorageError::Write)?;
        Ok(report)
    }

    /// Returns active imported API-key hashes for authentication.
    ///
    /// # Errors
    /// Returns an error if the key store cannot be read.
    pub fn api_key_hashes(&self) -> Result<Vec<[u8; 32]>, StorageError> {
        Ok(self
            .api_key_identities()?
            .into_iter()
            .map(|(hash, _)| hash)
            .collect())
    }

    /// Returns active imported API-key hashes with organization identity.
    ///
    /// # Errors
    /// Returns an error if a key hash is malformed or cannot be read.
    pub fn api_key_identities(&self) -> Result<Vec<ApiKeyIdentity>, StorageError> {
        let mut statement = self.connection.prepare(
            "SELECT key_hash, org_id FROM api_keys WHERE enabled=1 AND (expires_at IS NULL OR datetime(expires_at) > CURRENT_TIMESTAMP)",
        ).map_err(StorageError::Read)?;
        statement
            .query_map([], |row| {
                let bytes: Vec<u8> = row.get(0)?;
                let hash = bytes.try_into().map_err(|bytes: Vec<u8>| {
                    rusqlite::Error::FromSqlConversionFailure(
                        0,
                        rusqlite::types::Type::Blob,
                        format!("API key hash has {} bytes, expected 32", bytes.len()).into(),
                    )
                })?;
                Ok((hash, row.get(1)?))
            })
            .map_err(StorageError::Read)?
            .collect::<Result<Vec<_>, _>>()
            .map_err(StorageError::Read)
    }

    /// Returns whether a legacy snapshot fingerprint was fully imported.
    ///
    /// # Errors
    /// Returns an error if the migration ledger cannot be read.
    pub fn has_legacy_import(&self, source_hash: &str) -> Result<bool, StorageError> {
        self.connection
            .query_row(
                "SELECT EXISTS(SELECT 1 FROM legacy_imports WHERE source_hash=?1)",
                [source_hash],
                |row| row.get(0),
            )
            .map_err(StorageError::Read)
    }

    /// Records a completed, validated legacy import fingerprint.
    ///
    /// # Errors
    /// Returns an error if the completion marker cannot be persisted.
    pub fn record_legacy_import(
        &mut self,
        source_hash: &str,
        report: &LegacyImportReport,
    ) -> Result<(), StorageError> {
        self.connection
            .execute(
                "INSERT OR IGNORE INTO legacy_imports (source_hash, report) VALUES (?1, ?2)",
                params![source_hash, format!("{report:?}")],
            )
            .map(|_| ())
            .map_err(StorageError::Write)
    }
}
