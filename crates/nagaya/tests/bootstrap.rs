use std::net::SocketAddr;
use std::path::{Path, PathBuf};

use nagaya::bootstrap::Bootstrap;

const REQUIRED_ONLY: &str = r#"
http_bind = "127.0.0.1:8090"
gossip_bind = "127.0.0.1:8091"
store_dir = "/tmp/nagaya"
"#;

const FULL: &str = r#"
http_bind = "198.51.100.20:8090"
gossip_bind = "198.51.100.20:8091"
store_dir = "/var/lib/nagaya/store"
seed_peers = ["198.51.100.21:8091", "198.51.100.22:8091"]
park_budget_secs = 12
"#;

#[test]
fn the_config_reads_every_key() {
    let config = Bootstrap::parse(FULL).unwrap();
    assert_eq!(config.http_bind, "198.51.100.20:8090".parse::<SocketAddr>().unwrap());
    assert_eq!(config.gossip_bind, "198.51.100.20:8091".parse::<SocketAddr>().unwrap());
    assert_eq!(config.store_dir, Path::new("/var/lib/nagaya/store"));
    assert_eq!(
        config.seed_peers,
        vec![
            "198.51.100.21:8091".parse::<SocketAddr>().unwrap(),
            "198.51.100.22:8091".parse::<SocketAddr>().unwrap(),
        ]
    );
    assert_eq!(config.park_budget_secs, 12);
}

#[test]
fn the_optional_keys_have_defaults() {
    let config = Bootstrap::parse(REQUIRED_ONLY).unwrap();
    assert_eq!(config.seed_peers, Vec::new());
    assert_eq!(config.park_budget_secs, 10);
}

#[test]
fn an_unknown_key_is_refused() {
    let text = format!("{REQUIRED_ONLY}park_budget_seconds = 10\n");
    let error = Bootstrap::parse(&text).unwrap_err();
    assert!(
        error.to_string().contains("park_budget_seconds"),
        "the message must name the unknown key: {error}"
    );
}

#[test]
fn a_missing_required_key_is_refused() {
    let text = "http_bind = \"127.0.0.1:8090\"\nstore_dir = \"/tmp/nagaya\"\n";
    let error = Bootstrap::parse(text).unwrap_err();
    assert!(
        error.to_string().contains("gossip_bind"),
        "the message must name the missing key: {error}"
    );
}

#[test]
fn the_reader_uses_the_file() {
    let directory = tempfile::tempdir().unwrap();
    let path: PathBuf = directory.path().join("nagaya.toml");
    std::fs::write(&path, FULL).unwrap();
    assert_eq!(Bootstrap::read(&path).unwrap(), Bootstrap::parse(FULL).unwrap());
}

#[test]
fn a_missing_file_is_an_error_that_names_the_path() {
    let path = PathBuf::from("/nonexistent/nagaya.toml");
    let error = Bootstrap::read(&path).unwrap_err();
    assert!(error.to_string().contains("/nonexistent/nagaya.toml"), "{error}");
}
