use super::{
    ChunkCandidate, ChunkHydration, HashMap, MemoryCandidate, MemoryHydration, MemorySearchHit,
    MemoryVisibility, SearchHit, SearchOptions, Storage, StorageError, hydrate_memory_hits, json,
    params, parse_json, read_memory, validate_vector, vector_bytes,
};

impl Storage {
    /// Searches completed local documents using `SQLite` FTS5 relevance.
    ///
    /// # Errors
    /// Returns an error if the search query cannot be executed or decoded.
    pub fn search(&self, query: &str, limit: usize) -> Result<Vec<SearchHit>, StorageError> {
        self.search_for(&self.local_org_id, query, limit)
    }

    /// Lexically searches one explicit organization.
    ///
    /// # Errors
    /// Returns an error if the FTS query cannot be executed.
    pub fn search_for(
        &self,
        org_id: &str,
        query: &str,
        limit: usize,
    ) -> Result<Vec<SearchHit>, StorageError> {
        let query = format!("\"{}\"", query.replace('"', "\"\""));
        let mut statement = self.connection.prepare(
            "SELECT document_chunks.stable_id, documents.id, document_chunks.content, -bm25(document_chunks_fts), document_chunks.ordinal, documents.custom_id, documents.metadata, documents.filepath, documents.created_at, documents.updated_at, documents.content FROM document_chunks_fts JOIN document_chunks ON document_chunks.id=document_chunks_fts.rowid JOIN documents ON documents.id=document_chunks.document_id WHERE document_chunks_fts MATCH ?1 AND documents.org_id=?2 AND documents.status='done' ORDER BY bm25(document_chunks_fts), document_chunks.ordinal LIMIT ?3",
        ).map_err(StorageError::Read)?;
        statement
            .query_map(params![query, org_id, limit], |row| {
                Ok(SearchHit {
                    id: row.get(0)?,
                    document_id: row.get(1)?,
                    chunk: row.get(2)?,
                    score: row.get(3)?,
                    position: row.get(4)?,
                    custom_id: row.get(5)?,
                    metadata: parse_json(&row.get::<_, String>(6)?, 6)?,
                    filepath: row.get(7)?,
                    created_at: row.get(8)?,
                    updated_at: row.get(9)?,
                    document_content: row.get(10)?,
                })
            })
            .map_err(StorageError::Read)?
            .collect::<Result<Vec<_>, _>>()
            .map_err(StorageError::Read)
    }

    /// Lexically ranks narrow document chunk candidates without hydrating their text.
    ///
    /// The query is quoted as one FTS phrase, so user input is never interpreted as
    /// FTS syntax. Metadata and document constraints match semantic retrieval.
    ///
    /// # Errors
    /// Returns an error if the FTS query cannot be executed.
    pub fn search_chunk_lexical_candidates(
        &self,
        query: &str,
        limit: usize,
        options: &SearchOptions,
    ) -> Result<Vec<ChunkCandidate>, StorageError> {
        let organization_id = options
            .organization_id
            .as_deref()
            .unwrap_or(&self.local_org_id);
        let container_tags = json(&options.container_tags)?;
        let metadata_filter = options.filters.as_ref().map(json).transpose()?;
        let phrase = format!("\"{}\"", query.replace('"', "\"\""));
        let mut statement = self.connection.prepare(
            "SELECT document_chunks.id, document_chunks.stable_id, documents.id, document_chunks.ordinal FROM document_chunks_fts JOIN document_chunks ON document_chunks.id=document_chunks_fts.rowid JOIN documents ON documents.id=document_chunks.document_id WHERE document_chunks_fts MATCH ?1 AND documents.org_id=?2 AND documents.status='done' AND (?3='[]' OR EXISTS(SELECT 1 FROM json_each(documents.container_tags) stored JOIN json_each(?3) requested ON stored.value=requested.value)) AND (?4 IS NULL OR documents.id=?4 OR documents.custom_id=?4) AND (?5 IS NULL OR documents.filepath=?5 OR (substr(?5, -1)='/' AND substr(documents.filepath, 1, length(?5)-1)=substr(?5, 1, length(?5)-1))) AND (?6 IS NULL OR matches_metadata_filter(documents.metadata, ?6)) ORDER BY bm25(document_chunks_fts), documents.id, document_chunks.ordinal, document_chunks.stable_id LIMIT ?7",
        ).map_err(StorageError::Read)?;
        statement
            .query_map(
                params![
                    phrase,
                    organization_id,
                    container_tags,
                    options.document_id,
                    options.filepath,
                    metadata_filter,
                    limit
                ],
                |row| {
                    Ok(ChunkCandidate {
                        chunk_id: row.get(0)?,
                        stable_id: row.get(1)?,
                        document_id: row.get(2)?,
                        ordinal: row.get(3)?,
                        semantic_score: None,
                        lexical_rank: None,
                    })
                },
            )
            .map_err(StorageError::Read)?
            .enumerate()
            .map(|(rank, row)| {
                row.map(|mut candidate| {
                    candidate.lexical_rank = Some(rank + 1);
                    candidate
                })
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(StorageError::Read)
    }

    /// Searches current chunks by exact cosine similarity over normalized vectors.
    ///
    /// # Errors
    /// Returns an error for malformed query/stored vectors or a failed database read.
    pub fn search_semantic(
        &self,
        query: &[f32],
        model_id: &str,
        limit: usize,
        threshold: f32,
        options: &SearchOptions,
    ) -> Result<Vec<SearchHit>, StorageError> {
        validate_vector(query, query.len())?;
        let dimensions = query.len();
        let organization_id = options
            .organization_id
            .as_deref()
            .unwrap_or(&self.local_org_id);
        let query_bytes = vector_bytes(query);
        let container_tags = json(&options.container_tags)?;
        let metadata_filter = options.filters.as_ref().map(json).transpose()?;
        let mut statement = self.connection.prepare(
            "WITH filtered AS MATERIALIZED (SELECT document_chunks.stable_id, documents.id AS document_id, document_chunks.content AS chunk, chunk_embeddings.vector, document_chunks.ordinal, documents.custom_id, documents.metadata, documents.filepath, documents.created_at, documents.updated_at, documents.content AS document_content FROM chunk_embeddings JOIN document_chunks ON document_chunks.id=chunk_embeddings.chunk_id JOIN documents ON documents.id=document_chunks.document_id WHERE chunk_embeddings.model_id=?3 AND chunk_embeddings.dimensions=?2 AND documents.org_id=?4 AND documents.status='done' AND (?5='[]' OR EXISTS(SELECT 1 FROM json_each(documents.container_tags) stored JOIN json_each(?5) requested ON stored.value=requested.value)) AND (?6 IS NULL OR documents.id=?6 OR documents.custom_id=?6) AND (?7 IS NULL OR documents.filepath=?7 OR (substr(?7, -1)='/' AND substr(documents.filepath, 1, length(?7)-1)=substr(?7, 1, length(?7)-1))) AND (?8 IS NULL OR matches_metadata_filter(documents.metadata, ?8))), ranked AS MATERIALIZED (SELECT *, vector_dot(vector, ?1, ?2) AS score FROM filtered) SELECT stable_id, document_id, chunk, score, ordinal, custom_id, metadata, filepath, created_at, updated_at, document_content FROM ranked WHERE score>=?9 ORDER BY score DESC, document_id, ordinal, stable_id LIMIT ?10",
        ).map_err(StorageError::Read)?;
        statement
            .query_map(
                params![
                    query_bytes,
                    dimensions,
                    model_id,
                    organization_id,
                    container_tags,
                    options.document_id,
                    options.filepath,
                    metadata_filter,
                    threshold,
                    limit
                ],
                |row| {
                    Ok(SearchHit {
                        id: row.get(0)?,
                        document_id: row.get(1)?,
                        chunk: row.get(2)?,
                        score: row.get(3)?,
                        position: row.get(4)?,
                        custom_id: row.get(5)?,
                        metadata: parse_json(&row.get::<_, String>(6)?, 6)?,
                        filepath: row.get(7)?,
                        created_at: row.get(8)?,
                        updated_at: row.get(9)?,
                        document_content: row.get(10)?,
                    })
                },
            )
            .map_err(StorageError::Read)?
            .collect::<Result<Vec<_>, _>>()
            .map_err(StorageError::Read)
    }

    /// Searches canonical memories with exact cosine scoring.
    ///
    /// # Errors
    /// Returns an error for malformed vectors or stored metadata.
    pub fn search_memories(
        &self,
        query: &[f32],
        model_id: &str,
        container_tag: &str,
        limit: usize,
        threshold: f32,
        include_forgotten: bool,
    ) -> Result<Vec<MemorySearchHit>, StorageError> {
        self.search_memories_for(
            &self.local_org_id,
            query,
            model_id,
            container_tag,
            limit,
            threshold,
            MemoryVisibility {
                include_forgotten,
                include_expired: include_forgotten,
                include_superseded: include_forgotten,
                include_inactive_embeddings: include_forgotten,
            },
            MemoryHydration {
                relations: true,
                documents: true,
            },
        )
    }

    /// Searches memories in an explicit organization.
    ///
    /// # Errors
    /// Returns an error for malformed stored vectors or metadata.
    #[expect(clippy::too_many_arguments, reason = "search contract parameters")]
    pub fn search_memories_for(
        &self,
        org_id: &str,
        query: &[f32],
        model_id: &str,
        container_tag: &str,
        limit: usize,
        threshold: f32,
        visibility: MemoryVisibility,
        hydration: MemoryHydration,
    ) -> Result<Vec<MemorySearchHit>, StorageError> {
        validate_vector(query, query.len())?;
        let query_bytes = vector_bytes(query);
        let mut statement = self.connection.prepare(
            "WITH ranked AS MATERIALIZED (SELECT memories.id, memories.content, memories.metadata, memories.is_inferred, memories.is_static, memories.is_latest, memories.is_forgotten, memories.root_memory_id, memories.parent_memory_id, memories.version, memories.forget_after, memories.forget_reason, memories.created_at, memories.updated_at, vector_dot(memory_embeddings.vector, ?1, ?2) AS similarity FROM memory_embeddings JOIN memories ON memories.id=memory_embeddings.memory_id WHERE memories.org_id=?3 AND memories.container_tag=?4 AND memory_embeddings.model_id=?5 AND memory_embeddings.dimensions=?2 AND (memory_embeddings.active=1 OR ?6=1) AND (memories.is_latest=1 OR ?7=1) AND (memories.is_forgotten=0 OR ?8=1) AND (memories.forget_after IS NULL OR datetime(memories.forget_after)>CURRENT_TIMESTAMP OR ?9=1)) SELECT * FROM ranked WHERE similarity>=?10 ORDER BY similarity DESC, id LIMIT ?11",
        ).map_err(StorageError::Read)?;
        let rows = statement
            .query_map(
                params![
                    query_bytes,
                    query.len(),
                    org_id,
                    container_tag,
                    model_id,
                    visibility.include_inactive_embeddings,
                    visibility.include_superseded,
                    visibility.include_forgotten,
                    visibility.include_expired,
                    threshold,
                    limit
                ],
                |row| {
                    Ok(MemorySearchHit {
                        record: read_memory(row)?,
                        similarity: row.get(14)?,
                        parents: Vec::new(),
                        children: Vec::new(),
                        related: Vec::new(),
                        documents: Vec::new(),
                    })
                },
            )
            .map_err(StorageError::Read)?;
        let mut hits = rows
            .collect::<Result<Vec<_>, _>>()
            .map_err(StorageError::Read)?;
        hydrate_memory_hits(&self.connection, &mut hits, hydration)?;
        Ok(hits)
    }

    /// Lexically ranks visible memories without materializing their facts.
    ///
    /// # Errors
    /// Returns an error if FTS or the visibility filter cannot be evaluated.
    pub fn search_memory_lexical_candidates(
        &self,
        org_id: &str,
        query: &str,
        container_tag: &str,
        limit: usize,
        visibility: MemoryVisibility,
    ) -> Result<Vec<MemoryCandidate>, StorageError> {
        let phrase = format!("\"{}\"", query.replace('"', "\"\""));
        let mut statement = self.connection.prepare(
            "SELECT memories.id FROM memory_fts JOIN memories ON memories.rowid=memory_fts.rowid LEFT JOIN memory_embeddings ON memory_embeddings.memory_id=memories.id WHERE memory_fts MATCH ?1 AND memories.org_id=?2 AND memories.container_tag=?3 AND (memory_embeddings.active=1 OR ?4=1) AND (memories.is_latest=1 OR ?5=1) AND (memories.is_forgotten=0 OR ?6=1) AND (memories.forget_after IS NULL OR datetime(memories.forget_after)>CURRENT_TIMESTAMP OR ?7=1) ORDER BY bm25(memory_fts), memories.id LIMIT ?8",
        ).map_err(StorageError::Read)?;
        statement
            .query_map(
                params![
                    phrase,
                    org_id,
                    container_tag,
                    visibility.include_inactive_embeddings,
                    visibility.include_superseded,
                    visibility.include_forgotten,
                    visibility.include_expired,
                    limit
                ],
                |row| {
                    Ok(MemoryCandidate {
                        id: row.get(0)?,
                        semantic_score: None,
                        lexical_rank: None,
                    })
                },
            )
            .map_err(StorageError::Read)?
            .enumerate()
            .map(|(rank, row)| {
                row.map(|mut candidate| {
                    candidate.lexical_rank = Some(rank + 1);
                    candidate
                })
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(StorageError::Read)
    }

    /// Ranks memories without materializing memory records or relation context.
    ///
    /// # Errors
    /// Returns an error for malformed vectors or database reads.
    #[expect(clippy::too_many_arguments, reason = "candidate search constraints")]
    pub fn search_memory_candidates(
        &self,
        org_id: &str,
        query: &[f32],
        model_id: &str,
        container_tag: &str,
        limit: usize,
        threshold: f32,
        visibility: MemoryVisibility,
    ) -> Result<Vec<MemoryCandidate>, StorageError> {
        validate_vector(query, query.len())?;
        let mut statement = self.connection.prepare(
            "SELECT memories.id, vector_dot(memory_embeddings.vector, ?1, ?2) AS score FROM memory_embeddings JOIN memories ON memories.id=memory_embeddings.memory_id WHERE memories.org_id=?3 AND memories.container_tag=?4 AND memory_embeddings.model_id=?5 AND memory_embeddings.dimensions=?2 AND (memory_embeddings.active=1 OR ?6=1) AND (memories.is_latest=1 OR ?7=1) AND (memories.is_forgotten=0 OR ?8=1) AND (memories.forget_after IS NULL OR datetime(memories.forget_after)>CURRENT_TIMESTAMP OR ?9=1) AND vector_dot(memory_embeddings.vector, ?1, ?2)>=?10 ORDER BY score DESC, memories.id LIMIT ?11",
        ).map_err(StorageError::Read)?;
        statement
            .query_map(
                params![
                    vector_bytes(query),
                    query.len(),
                    org_id,
                    container_tag,
                    model_id,
                    visibility.include_inactive_embeddings,
                    visibility.include_superseded,
                    visibility.include_forgotten,
                    visibility.include_expired,
                    threshold,
                    limit
                ],
                |row| {
                    Ok(MemoryCandidate {
                        id: row.get(0)?,
                        semantic_score: Some(row.get(1)?),
                        lexical_rank: None,
                    })
                },
            )
            .map_err(StorageError::Read)?
            .collect::<Result<Vec<_>, _>>()
            .map_err(StorageError::Read)
    }

    /// Hydrates selected memory candidates in their original ranking order.
    ///
    /// # Errors
    /// Returns an error if a selected memory cannot be loaded or decoded.
    pub fn hydrate_memories(
        &self,
        candidates: &[MemoryCandidate],
        hydration: MemoryHydration,
    ) -> Result<Vec<MemorySearchHit>, StorageError> {
        if candidates.is_empty() {
            return Ok(Vec::new());
        }
        let ids = candidates
            .iter()
            .map(|candidate| candidate.id.clone())
            .collect::<Vec<_>>();
        let scores = candidates
            .iter()
            .map(|candidate| (candidate.id.clone(), candidate.semantic_score))
            .collect::<HashMap<_, _>>();
        let placeholders = std::iter::repeat_n("?", ids.len())
            .collect::<Vec<_>>()
            .join(",");
        let sql = format!(
            "SELECT id, content, metadata, is_inferred, is_static, is_latest, is_forgotten, root_memory_id, parent_memory_id, version, forget_after, forget_reason, created_at, updated_at FROM memories WHERE id IN ({placeholders})"
        );
        let mut statement = self.connection.prepare(&sql).map_err(StorageError::Read)?;
        let by_id = statement
            .query_map(rusqlite::params_from_iter(ids.iter()), |row| {
                let record = read_memory(row)?;
                let similarity = scores.get(&record.id).copied().flatten().unwrap_or(0.0);
                Ok((
                    record.id.clone(),
                    MemorySearchHit {
                        record,
                        similarity,
                        parents: Vec::new(),
                        children: Vec::new(),
                        related: Vec::new(),
                        documents: Vec::new(),
                    },
                ))
            })
            .map_err(StorageError::Read)?
            .collect::<Result<HashMap<_, _>, _>>()
            .map_err(StorageError::Read)?;
        let mut hits = ids
            .into_iter()
            .filter_map(|id| by_id.get(&id).cloned())
            .collect::<Vec<_>>();
        hydrate_memory_hits(&self.connection, &mut hits, hydration)?;
        Ok(hits)
    }

    /// Ranks document chunks without materializing chunk text or document records.
    ///
    /// # Errors
    /// Returns an error for malformed vectors, filters, or database reads.
    pub fn search_chunk_candidates(
        &self,
        query: &[f32],
        model_id: &str,
        limit: usize,
        threshold: f32,
        options: &SearchOptions,
    ) -> Result<Vec<ChunkCandidate>, StorageError> {
        validate_vector(query, query.len())?;
        let organization_id = options
            .organization_id
            .as_deref()
            .unwrap_or(&self.local_org_id);
        let container_tags = json(&options.container_tags)?;
        let metadata_filter = options.filters.as_ref().map(json).transpose()?;
        let mut statement = self.connection.prepare(
            "SELECT document_chunks.id, document_chunks.stable_id, documents.id, document_chunks.ordinal, vector_dot(chunk_embeddings.vector, ?1, ?2) AS score FROM chunk_embeddings JOIN document_chunks ON document_chunks.id=chunk_embeddings.chunk_id JOIN documents ON documents.id=document_chunks.document_id WHERE chunk_embeddings.model_id=?3 AND chunk_embeddings.dimensions=?2 AND documents.org_id=?4 AND documents.status='done' AND (?5='[]' OR EXISTS(SELECT 1 FROM json_each(documents.container_tags) stored JOIN json_each(?5) requested ON stored.value=requested.value)) AND (?6 IS NULL OR documents.id=?6 OR documents.custom_id=?6) AND (?7 IS NULL OR documents.filepath=?7 OR (substr(?7, -1)='/' AND substr(documents.filepath, 1, length(?7)-1)=substr(?7, 1, length(?7)-1))) AND (?8 IS NULL OR matches_metadata_filter(documents.metadata, ?8)) AND vector_dot(chunk_embeddings.vector, ?1, ?2)>=?9 ORDER BY score DESC, documents.id, document_chunks.ordinal, document_chunks.stable_id LIMIT ?10",
        ).map_err(StorageError::Read)?;
        statement
            .query_map(
                params![
                    vector_bytes(query),
                    query.len(),
                    model_id,
                    organization_id,
                    container_tags,
                    options.document_id,
                    options.filepath,
                    metadata_filter,
                    threshold,
                    limit
                ],
                |row| {
                    Ok(ChunkCandidate {
                        chunk_id: row.get(0)?,
                        stable_id: row.get(1)?,
                        document_id: row.get(2)?,
                        ordinal: row.get(3)?,
                        semantic_score: Some(row.get(4)?),
                        lexical_rank: None,
                    })
                },
            )
            .map_err(StorageError::Read)?
            .collect::<Result<Vec<_>, _>>()
            .map_err(StorageError::Read)
    }

    /// Hydrates selected chunk candidates while preserving their ranked order.
    ///
    /// # Errors
    /// Returns an error if a selected chunk cannot be loaded or decoded.
    pub fn hydrate_chunks(
        &self,
        candidates: &[ChunkCandidate],
        policy: ChunkHydration,
    ) -> Result<Vec<SearchHit>, StorageError> {
        if candidates.is_empty() {
            return Ok(Vec::new());
        }
        let ids = candidates
            .iter()
            .map(|candidate| candidate.chunk_id)
            .collect::<Vec<_>>();
        let scores = candidates
            .iter()
            .map(|candidate| (candidate.chunk_id, candidate.semantic_score))
            .collect::<HashMap<_, _>>();
        let placeholders = std::iter::repeat_n("?", ids.len())
            .collect::<Vec<_>>()
            .join(",");
        let document_content = if policy.full_document {
            "documents.content"
        } else {
            "NULL"
        };
        let sql = format!(
            "SELECT document_chunks.id, document_chunks.stable_id, documents.id, document_chunks.content, document_chunks.ordinal, documents.custom_id, documents.metadata, documents.filepath, documents.created_at, documents.updated_at, {document_content} FROM document_chunks JOIN documents ON documents.id=document_chunks.document_id WHERE document_chunks.id IN ({placeholders})"
        );
        let mut statement = self.connection.prepare(&sql).map_err(StorageError::Read)?;
        let mut hits = statement
            .query_map(rusqlite::params_from_iter(ids.iter()), |row| {
                let chunk_id: i64 = row.get(0)?;
                Ok((
                    chunk_id,
                    SearchHit {
                        id: row.get(1)?,
                        document_id: row.get(2)?,
                        chunk: row.get(3)?,
                        score: scores.get(&chunk_id).copied().flatten().unwrap_or(0.0),
                        position: row.get(4)?,
                        custom_id: row.get(5)?,
                        metadata: parse_json(&row.get::<_, String>(6)?, 6)?,
                        filepath: row.get(7)?,
                        created_at: row.get(8)?,
                        updated_at: row.get(9)?,
                        document_content: row.get::<_, Option<String>>(10)?.unwrap_or_default(),
                    },
                ))
            })
            .map_err(StorageError::Read)?
            .collect::<Result<HashMap<_, _>, _>>()
            .map_err(StorageError::Read)?;
        Ok(ids.into_iter().filter_map(|id| hits.remove(&id)).collect())
    }
}
