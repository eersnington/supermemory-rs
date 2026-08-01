use super::{
    Connection, Duration, Path, Storage, StorageError, initialize, register_vector_functions,
};

impl Storage {
    /// Opens a database and applies all embedded migrations.
    ///
    /// # Errors
    /// Returns an error when the database cannot be opened, migrated, or seeded.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, StorageError> {
        let database_path = path.as_ref().to_path_buf();
        let mut connection = Connection::open(&database_path).map_err(StorageError::Open)?;
        let local_org_id = initialize(&mut connection)?;
        Ok(Self {
            connection,
            local_org_id,
            database_path: Some(database_path),
        })
    }

    /// Opens an isolated in-memory database and applies all migrations.
    ///
    /// # Errors
    /// Returns an error when the database cannot be migrated or seeded.
    pub fn in_memory() -> Result<Self, StorageError> {
        let mut connection = Connection::open_in_memory().map_err(StorageError::Open)?;
        let local_org_id = initialize(&mut connection)?;
        Ok(Self {
            connection,
            local_org_id,
            database_path: None,
        })
    }

    /// Opens an independent read connection without rerunning migrations or job recovery.
    ///
    /// # Errors
    /// Returns an error for in-memory storage or when the read connection cannot be configured.
    pub fn fork(&self) -> Result<Self, StorageError> {
        let path = self
            .database_path
            .as_ref()
            .ok_or(StorageError::CannotForkInMemory)?;
        let connection = Connection::open(path).map_err(StorageError::Open)?;
        connection
            .busy_timeout(Duration::from_secs(5))
            .map_err(StorageError::Configure)?;
        connection
            .execute_batch("PRAGMA foreign_keys=ON; PRAGMA query_only=ON;")
            .map_err(StorageError::Configure)?;
        register_vector_functions(&connection)?;
        Ok(Self {
            connection,
            local_org_id: self.local_org_id.clone(),
            database_path: Some(path.clone()),
        })
    }

    /// Returns the organization used for local unauthenticated requests.
    #[must_use]
    pub fn local_organization_id(&self) -> &str {
        &self.local_org_id
    }
}
