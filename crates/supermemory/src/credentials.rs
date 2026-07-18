//! Compatible encrypted model-provider configuration.

use std::{
    collections::BTreeMap,
    fs::OpenOptions,
    io::Write,
    path::{Path, PathBuf},
    process::Command,
};

use aes_gcm::{
    Aes256Gcm, KeyInit,
    aead::{AeadInPlace, generic_array::GenericArray},
};
use pbkdf2::pbkdf2_hmac;
use sha2::{Digest, Sha256};
use thiserror::Error;

const MAGIC: &[u8; 4] = b"SME1";
const CONTEXT: &[u8] = b"supermemory-model-keys-v1";
const BUILD_SECRET: [u8; 32] = [
    0x4f, 0x73, 0x8a, 0xe1, 0x1c, 0xb9, 0x44, 0x2d, 0x96, 0xc7, 0x05, 0xfb, 0x88, 0x12, 0xa4, 0x6e,
    0xd0, 0x37, 0x91, 0x5b, 0x4a, 0xfe, 0x2c, 0x70, 0xa3, 0x18, 0xc5, 0x8d, 0x6b, 0x29, 0xe4, 0x77,
];

/// Loads the Rust secret store or imports the v0.0.5 store without modifying it.
///
/// # Errors
/// Returns an error when an existing encrypted store cannot be authenticated or safely copied.
pub fn load_or_import(
    data_dir: &Path,
    legacy_data_dir: &Path,
) -> Result<BTreeMap<String, String>, CredentialError> {
    let current = data_dir.join("env.enc");
    if current.exists() {
        return decrypt_file(&current, data_dir);
    }
    let legacy = legacy_data_dir.join("env.enc");
    if !legacy.exists() {
        return Ok(BTreeMap::new());
    }
    let values = decrypt_file(&legacy, legacy_data_dir)?;
    std::fs::create_dir_all(data_dir).map_err(|source| CredentialError::CreateDirectory {
        path: data_dir.to_path_buf(),
        source,
    })?;
    encrypt_file(&current, data_dir, &values)?;
    Ok(values)
}

/// Decrypts one SME1 env document with the v0.0.5 key derivation.
///
/// # Errors
/// Returns an error if the frame is malformed or no machine identity authenticates it.
pub fn decrypt_file(
    path: &Path,
    data_dir: &Path,
) -> Result<BTreeMap<String, String>, CredentialError> {
    let frame = std::fs::read(path).map_err(|source| CredentialError::Read {
        path: path.to_path_buf(),
        source,
    })?;
    if frame.len() < 32 || frame.get(..4) != Some(MAGIC) {
        return Err(CredentialError::MalformedFrame {
            path: path.to_path_buf(),
        });
    }
    let nonce = GenericArray::from_slice(&frame[4..16]);
    let tag = GenericArray::from_slice(&frame[16..32]);
    for machine_id in machine_ids(data_dir) {
        let key = derive_key(&machine_id);
        let cipher = Aes256Gcm::new_from_slice(&key).map_err(|_| CredentialError::InvalidKey)?;
        let mut plaintext = frame[32..].to_vec();
        if cipher
            .decrypt_in_place_detached(nonce, b"", &mut plaintext, tag)
            .is_ok()
        {
            let plaintext = String::from_utf8(plaintext).map_err(CredentialError::Utf8)?;
            return parse_env(&plaintext);
        }
    }
    Err(CredentialError::Authentication {
        path: path.to_path_buf(),
    })
}

/// Writes an atomic SME1 env document with owner-only permissions.
///
/// # Errors
/// Returns an error before replacing the destination if encryption or writing fails.
pub fn encrypt_file(
    path: &Path,
    data_dir: &Path,
    values: &BTreeMap<String, String>,
) -> Result<(), CredentialError> {
    let machine_id = machine_ids(data_dir)
        .into_iter()
        .next()
        .ok_or(CredentialError::NoMachineIdentity)?;
    let key = derive_key(&machine_id);
    let cipher = Aes256Gcm::new_from_slice(&key).map_err(|_| CredentialError::InvalidKey)?;
    let mut nonce = [0_u8; 12];
    getrandom::fill(&mut nonce).map_err(CredentialError::Random)?;
    let mut plaintext = serialize_env(values).into_bytes();
    let tag = cipher
        .encrypt_in_place_detached(GenericArray::from_slice(&nonce), b"", &mut plaintext)
        .map_err(|_| CredentialError::Encrypt)?;
    let mut frame = Vec::with_capacity(32 + plaintext.len());
    frame.extend_from_slice(MAGIC);
    frame.extend_from_slice(&nonce);
    frame.extend_from_slice(&tag);
    frame.extend_from_slice(&plaintext);

    let temporary = path.with_extension("enc.tmp");
    let mut options = OpenOptions::new();
    options.write(true).create(true).truncate(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let mut file = options
        .open(&temporary)
        .map_err(|source| CredentialError::Write {
            path: temporary.clone(),
            source,
        })?;
    file.write_all(&frame)
        .and_then(|()| file.sync_all())
        .map_err(|source| CredentialError::Write {
            path: temporary.clone(),
            source,
        })?;
    std::fs::rename(&temporary, path).map_err(|source| CredentialError::Write {
        path: path.to_path_buf(),
        source,
    })
}

fn derive_key(machine_id: &str) -> [u8; 32] {
    let machine_hash = Sha256::digest(machine_id.as_bytes());
    let mut mixed = [0_u8; 32];
    for (index, byte) in mixed.iter_mut().enumerate() {
        *byte = BUILD_SECRET[index] ^ machine_hash[index];
    }
    let mut key = [0_u8; 32];
    pbkdf2_hmac::<Sha256>(&mixed, CONTEXT, 100_000, &mut key);
    key
}

fn machine_ids(data_dir: &Path) -> Vec<String> {
    let mut ids = Vec::new();
    for path in ["/etc/machine-id", "/var/lib/dbus/machine-id"] {
        if let Ok(value) = std::fs::read_to_string(path) {
            push_unique(&mut ids, value.trim());
            if !ids.is_empty() {
                break;
            }
        }
    }
    if ids.is_empty() {
        if let Ok(output) = Command::new("/usr/sbin/ioreg")
            .args(["-rd1", "-c", "IOPlatformExpertDevice"])
            .output()
        {
            if output.status.success() {
                let output = String::from_utf8_lossy(&output.stdout);
                if let Some(uuid) = platform_uuid(&output) {
                    push_unique(&mut ids, uuid);
                }
            }
        }
    }
    if ids.is_empty() {
        let path = data_dir.join("machine-key");
        if let Ok(value) = std::fs::read_to_string(&path) {
            push_unique(&mut ids, value.trim());
        }
    }
    if ids.is_empty() {
        let hostname = Command::new("hostname")
            .output()
            .ok()
            .filter(|output| output.status.success())
            .map_or_else(
                || "unknown".to_owned(),
                |output| String::from_utf8_lossy(&output.stdout).trim().to_owned(),
            );
        push_unique(
            &mut ids,
            &format!("fallback:{hostname}:{}", std::env::consts::OS),
        );
    }
    ids
}

fn platform_uuid(output: &str) -> Option<&str> {
    output.lines().find_map(|line| {
        let marker = "\"IOPlatformUUID\" = \"";
        let value = line.trim().strip_prefix(marker)?;
        value.strip_suffix('"')
    })
}

fn push_unique(values: &mut Vec<String>, value: &str) {
    if !value.is_empty() && !values.iter().any(|existing| existing == value) {
        values.push(value.to_owned());
    }
}

fn parse_env(contents: &str) -> Result<BTreeMap<String, String>, CredentialError> {
    let mut values = BTreeMap::new();
    for line in contents.lines().map(str::trim) {
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let (name, value) = line
            .split_once('=')
            .ok_or_else(|| CredentialError::MalformedAssignment(line.to_owned()))?;
        if name.is_empty()
            || !name
                .bytes()
                .all(|byte| byte.is_ascii_uppercase() || byte.is_ascii_digit() || byte == b'_')
        {
            return Err(CredentialError::MalformedAssignment(line.to_owned()));
        }
        values.insert(name.to_owned(), unquote(value));
    }
    Ok(values)
}

fn unquote(value: &str) -> String {
    let value = value.trim();
    if value.len() >= 2
        && ((value.starts_with('\'') && value.ends_with('\''))
            || (value.starts_with('"') && value.ends_with('"')))
    {
        value[1..value.len() - 1].replace("'\\''", "'")
    } else {
        value.to_owned()
    }
}

fn serialize_env(values: &BTreeMap<String, String>) -> String {
    values
        .iter()
        .map(|(name, value)| format!("{name}='{}'", value.replace('\'', "'\\''")))
        .collect::<Vec<_>>()
        .join("\n")
        + "\n"
}

/// Failure while reading, authenticating, or writing encrypted credentials.
#[derive(Debug, Error)]
pub enum CredentialError {
    #[error(
        "failed to create credential directory {path}; existing credentials were not changed: {source}"
    )]
    CreateDirectory {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to read encrypted credentials from {path}: {source}")]
    Read {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("encrypted credential file {path} is not a valid SME1 frame")]
    MalformedFrame { path: PathBuf },
    #[error(
        "encrypted credential file {path} could not be authenticated with this machine identity; the file was not changed"
    )]
    Authentication { path: PathBuf },
    #[error("decrypted credential data is not UTF-8: {0}")]
    Utf8(#[source] std::string::FromUtf8Error),
    #[error("invalid env assignment in decrypted credentials: {0}")]
    MalformedAssignment(String),
    #[error("no machine identity is available for credential encryption")]
    NoMachineIdentity,
    #[error("credential key has an invalid size")]
    InvalidKey,
    #[error("failed to encrypt credentials")]
    Encrypt,
    #[error(
        "failed to write encrypted credentials to {path}; prior credentials remain intact: {source}"
    )]
    Write {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("operating-system randomness failed; credentials were not written: {0}")]
    Random(getrandom::Error),
}
