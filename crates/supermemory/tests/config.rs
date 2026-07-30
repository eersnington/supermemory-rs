use std::net::{IpAddr, Ipv4Addr, SocketAddr};

use supermemory::Config;

#[test]
fn config_parses_explicit_startup_values() {
    let config = Config::try_parse_from([
        "supermemory",
        "--bind",
        "0.0.0.0:8080",
        "--database",
        "data.db",
        "--monitor",
    ])
    .expect("valid configuration");
    assert_eq!(
        config.bind,
        SocketAddr::new(IpAddr::V4(Ipv4Addr::UNSPECIFIED), 8080)
    );
    assert_eq!(config.database.to_string_lossy(), "data.db");
    assert!(config.monitor);
}

#[test]
fn config_uses_compatibility_default_port() {
    let config = Config::try_parse_from(["supermemory"]).expect("config");
    assert_eq!(config.bind.port(), 6767);
    assert!(!config.monitor);
}

#[test]
fn config_uses_stable_per_user_database_by_default() {
    let config = Config::try_parse_from(["supermemory-rs"]).expect("config");
    assert!(config.database.ends_with(".supermemory-rs/supermemory.db"));
}

#[test]
fn config_does_not_expose_api_key_argument() {
    assert!(Config::try_parse_from(["supermemory", "--api-key", "secret"]).is_err());
}
