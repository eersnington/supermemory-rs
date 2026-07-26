//! Read-only export of the encrypted v0.0.5 `PGlite` snapshot.

use std::{
    fs::OpenOptions,
    io::Write,
    path::{Path, PathBuf},
    process::Command,
};

use thiserror::Error;

use crate::credentials;

/// Decrypts a legacy snapshot into protected temporary storage and exports stable JSONL.
///
/// The source snapshot is opened read-only and is never passed to `PGlite` directly.
///
/// # Errors
/// Returns an error if decryption, temporary storage, or the constrained exporter fails.
pub fn export_snapshot(
    legacy_data_dir: &Path,
    runtime_dir: &Path,
    output_path: &Path,
) -> Result<(), LegacyExportError> {
    let source = legacy_data_dir.join("data");
    let decrypted =
        credentials::decrypt_frame(&source, legacy_data_dir, b"SMD1", b"supermemory-pglite-v1")?;
    let temporary = std::env::temp_dir().join(format!(
        "supermemory-legacy-{}.tgz",
        storage::generate_id()?
    ));
    write_private(&temporary, &decrypted)?;
    drop(decrypted);

    let exporter = std::env::var_os("SUPERMEMORY_MIGRATION_EXPORTER").map_or_else(
        || Path::new(env!("CARGO_MANIFEST_DIR")).join("../../migration/export.mjs"),
        PathBuf::from,
    );
    let migration_dir = exporter
        .parent()
        .ok_or_else(|| LegacyExportError::InvalidExporterPath(exporter.clone()))?;
    let result = Command::new("node")
        .arg(&exporter)
        .arg(&temporary)
        .arg(runtime_dir.join("pglite.wasm"))
        .arg(runtime_dir.join("pglite.data"))
        .arg(output_path)
        .current_dir(migration_dir)
        .env_clear()
        .env("PATH", std::env::var_os("PATH").unwrap_or_default())
        .output()
        .map_err(LegacyExportError::StartExporter);
    let cleanup = std::fs::remove_file(&temporary);
    let output = result?;
    cleanup.map_err(|source| LegacyExportError::Cleanup {
        path: temporary,
        source,
    })?;
    if !output.status.success() {
        return Err(LegacyExportError::ExporterFailed {
            status: output.status.code(),
            stderr: String::from_utf8_lossy(&output.stderr).trim().to_owned(),
        });
    }
    Ok(())
}

fn write_private(path: &Path, contents: &[u8]) -> Result<(), LegacyExportError> {
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let mut file = options
        .open(path)
        .map_err(|source| LegacyExportError::WriteTemporary {
            path: path.to_path_buf(),
            source,
        })?;
    file.write_all(contents)
        .and_then(|()| file.sync_all())
        .map_err(|source| LegacyExportError::WriteTemporary {
            path: path.to_path_buf(),
            source,
        })
}

/// Failure while exporting an encrypted legacy data snapshot.
#[derive(Debug, Error)]
pub enum LegacyExportError {
    #[error(transparent)]
    Credentials(#[from] credentials::CredentialError),
    #[error(transparent)]
    Storage(#[from] storage::StorageError),
    #[error("failed to write protected temporary snapshot {path}: {source}")]
    WriteTemporary {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("migration exporter path has no parent directory: {0}")]
    InvalidExporterPath(PathBuf),
    #[error("failed to start the read-only PGlite exporter: {0}")]
    StartExporter(#[source] std::io::Error),
    #[error("PGlite exporter failed with status {status:?}: {stderr}")]
    ExporterFailed { status: Option<i32>, stderr: String },
    #[error("failed to remove protected temporary snapshot {path}: {source}")]
    Cleanup {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
}
