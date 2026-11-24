//! GPU test of the factory of `mk-inkling2-sm121` (milestone 37 step 11): the open gives an engine whose limits are
//! the limits that the factory planned, and whose identity names the cache format of the config.
//!
//! `INKLING2_MODEL_DIR=<folder> NAGAYA_TEST_CARDS=<pci address> cargo test -p nagaya --features cuda --test
//!  inkling2_sm121_open -- --test-threads=1`

#![cfg(feature = "cuda")]

use nagaya::admission::{Factory, GroupDevice};
use nagaya::inkling2_sm121::Inkling2Sm121Factory;

#[test]
fn the_open_gives_the_limits_that_the_factory_planned() {
    let require = std::env::var_os("NAGAYA_REQUIRE_MODELS").is_some();
    let Some(dir) = std::env::var_os("INKLING2_MODEL_DIR").filter(|dir| std::path::Path::new(dir).is_dir()) else {
        assert!(
            !require,
            "INKLING2_MODEL_DIR is absent and NAGAYA_REQUIRE_MODELS is set"
        );
        eprintln!("inkling2_sm121_open: skipped, INKLING2_MODEL_DIR is absent");
        return;
    };
    let device = GroupDevice {
        id: 0,
        name: "NVIDIA GB10".to_owned(),
        pci_address: std::env::var("NAGAYA_TEST_CARDS")
            .expect("NAGAYA_TEST_CARDS names the free GPU")
            .split(',')
            .next()
            .unwrap()
            .trim()
            .to_ascii_lowercase(),
        memory_bytes: 128 << 30,
    };
    for cache in ["f32", "e4m3"] {
        let mut table = toml::Table::new();
        table.insert(
            "model_dir".to_owned(),
            toml::Value::String(dir.to_string_lossy().into_owned()),
        );
        table.insert(
            "slots".to_owned(),
            toml::Value::Array(vec![toml::Value::Integer(1024), toml::Value::Integer(2048)]),
        );
        table.insert("cache".to_owned(), toml::Value::String(cache.to_owned()));
        let devices = [device.clone()];
        let planned = Inkling2Sm121Factory.plan(&table, &devices).unwrap();
        let engine = Inkling2Sm121Factory
            .open(&table, &devices)
            .unwrap_or_else(|e| panic!("{}", e.message));
        assert_eq!(
            engine.limits(),
            planned,
            "the limits differ from the plan with cache {cache}"
        );
        assert_eq!(engine.identity().numbers["cache"], cache);
    }
}
