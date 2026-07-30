//! Read-only export of the encrypted v0.0.5 `PGlite` snapshot.

use std::{
    fs::{File, OpenOptions},
    io::{BufWriter, Write},
    path::{Path, PathBuf},
};

use pglite_oxide::{Pglite, extensions};
use serde_json::{Map, Value, json};
use thiserror::Error;

use crate::credentials;

/// Decrypts a legacy snapshot and exports stable JSONL with embedded `PGlite`.
///
/// The source snapshot is never modified. The embedded `PGlite` engine exists only for this export.
///
/// # Errors
/// Returns an error if decryption, `PGlite` startup, querying, or export fails.
pub fn export_snapshot(
    legacy_data_dir: &Path,
    output_path: &Path,
) -> Result<(), LegacyExportError> {
    let source = legacy_data_dir.join("data");
    let decrypted =
        credentials::decrypt_frame(&source, legacy_data_dir, b"SMD1", b"supermemory-pglite-v1")?;
    let mut database = Pglite::builder()
        .temporary()
        .template_cache(false)
        .load_data_dir_archive(decrypted)
        .extension(extensions::VECTOR)
        .open()
        .map_err(LegacyExportError::OpenDatabase)?;

    let export_result = export_database(&mut database, output_path);
    let close_result = database.close().map_err(LegacyExportError::CloseDatabase);
    if export_result.is_err() {
        let _ = std::fs::remove_file(output_path);
    }
    export_result.and(close_result)
}

fn export_database(database: &mut Pglite, output_path: &Path) -> Result<(), LegacyExportError> {
    let tables = database
        .query(
            "SELECT table_name::text FROM information_schema.tables WHERE table_schema='public' AND table_type='BASE TABLE' ORDER BY table_name",
            &[],
            None,
        )
        .map_err(LegacyExportError::Query)?;
    let table_names = tables
        .rows
        .iter()
        .map(|row| object_string(row, "table_name").map(str::to_owned))
        .collect::<Result<Vec<_>, _>>()?;

    let file = private_output(output_path)?;
    let mut output = BufWriter::new(file);
    write_record(
        &mut output,
        &json!({ "type": "manifest", "format": 1, "tables": table_names }),
    )?;
    for table in &table_names {
        let quoted = table.replace('"', "\"\"");
        let query = format!(
            "SELECT row_to_json(export_row) AS row FROM (SELECT * FROM \"{quoted}\") AS export_row"
        );
        for result in database
            .query(&query, &[], None)
            .map_err(LegacyExportError::Query)?
            .rows
        {
            let row = result
                .as_object()
                .and_then(|result| result.get("row"))
                .and_then(Value::as_object)
                .ok_or_else(|| LegacyExportError::InvalidRow(table.clone()))?;
            write_row(&mut output, table, row)?;
        }
    }
    write_record(&mut output, &json!({ "type": "complete" }))?;
    output.flush().map_err(LegacyExportError::WriteExport)?;
    output
        .get_ref()
        .sync_all()
        .map_err(LegacyExportError::WriteExport)
}

fn object_string<'a>(value: &'a Value, key: &str) -> Result<&'a str, LegacyExportError> {
    value
        .as_object()
        .and_then(|object| object.get(key))
        .and_then(Value::as_str)
        .ok_or_else(|| LegacyExportError::InvalidRow(key.to_owned()))
}

fn write_row(
    output: &mut impl Write,
    table: &str,
    row: &Map<String, Value>,
) -> Result<(), LegacyExportError> {
    write_record(
        output,
        &json!({ "type": "row", "table": table, "row": row }),
    )
}

fn write_record(output: &mut impl Write, value: &Value) -> Result<(), LegacyExportError> {
    serde_json::to_writer(&mut *output, value).map_err(LegacyExportError::Json)?;
    output
        .write_all(b"\n")
        .map_err(LegacyExportError::WriteExport)
}

fn private_output(path: &Path) -> Result<File, LegacyExportError> {
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    options
        .open(path)
        .map_err(|source| LegacyExportError::CreateExport {
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
    #[error("could not open the legacy PGlite database: {0}")]
    OpenDatabase(#[source] anyhow::Error),
    #[error("could not read the legacy PGlite database: {0}")]
    Query(#[source] anyhow::Error),
    #[error("could not close the legacy PGlite database: {0}")]
    CloseDatabase(#[source] anyhow::Error),
    #[error("legacy PGlite returned an invalid row for {0}")]
    InvalidRow(String),
    #[error("could not create migration export {path}: {source}")]
    CreateExport {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("could not encode migration data: {0}")]
    Json(#[source] serde_json::Error),
    #[error("could not write migration data: {0}")]
    WriteExport(#[source] std::io::Error),
}
