use super::{Storage, StorageError};

impl Storage {
    /// Returns the current migration version.
    ///
    /// # Errors
    /// Returns an error when the migration ledger cannot be read.
    pub fn migration_version(&self) -> Result<i64, StorageError> {
        self.connection
            .query_row(
                "SELECT COALESCE(MAX(version), 0) FROM schema_migrations",
                [],
                |row| row.get(0),
            )
            .map_err(StorageError::Read)
    }
}
