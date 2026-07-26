use std::net::{IpAddr, Ipv4Addr, SocketAddr};

use supermemory::Config;

#[test]
fn config_parses_explicit_startup_values() {
    let config =
        Config::try_parse_from(["supermemory", "--bind", "0.0.0.0:8080", "--data", "data"])
            .expect("valid configuration");
    assert_eq!(
        config.bind,
        SocketAddr::new(IpAddr::V4(Ipv4Addr::UNSPECIFIED), 8080)
    );
    assert_eq!(config.data.to_string_lossy(), "data");
}

#[test]
fn config_uses_compatibility_default_port() {
    assert_eq!(
        Config::try_parse_from(["supermemory"])
            .expect("config")
            .bind
            .port(),
        6767
    );
}

#[test]
fn config_uses_local_turso_defaults() {
    let config = Config::try_parse_from(["supermemory-rs"]).expect("config");
    assert!(config.data.ends_with(".supermemory-rs"));
}

#[test]
fn config_does_not_expose_api_key_argument() {
    assert!(Config::try_parse_from(["supermemory", "--api-key", "secret"]).is_err());
}
