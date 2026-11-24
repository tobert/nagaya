//! GPU gate of `describe` of the factory of `mk-inkling2-sm121` (milestone 37 step 11, spec 11.1): the description gives
//! the identity and the limits of an open, with no load of weights. An open that reports another identity is refused.
//!
//! `INKLING2_MODEL_DIR=<folder> NAGAYA_TEST_CARDS=<pci address> cargo test -p nagaya --features cuda --test
//!  inkling2_sm121_describe -- --test-threads=1`

#![cfg(feature = "cuda")]

use std::time::Instant;

use nagaya::admission::{Factory, GroupDevice};
use nagaya::inkling2_sm121::Inkling2Sm121Factory;

fn setup() -> Option<(std::path::PathBuf, GroupDevice)> {
    let require = std::env::var_os("NAGAYA_REQUIRE_MODELS").is_some();
    let Some(dir) = std::env::var_os("INKLING2_MODEL_DIR").filter(|dir| std::path::Path::new(dir).is_dir()) else {
        assert!(
            !require,
            "INKLING2_MODEL_DIR is absent and NAGAYA_REQUIRE_MODELS is set"
        );
        eprintln!("inkling2_sm121_describe: skipped, INKLING2_MODEL_DIR is absent");
        return None;
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
    Some((dir.into(), device))
}

fn table(dir: &std::path::Path, cache: &str) -> toml::Table {
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
    table
}

#[test]
fn the_description_equals_the_identity_and_the_limits_of_the_open() {
    let Some((dir, device)) = setup() else { return };
    let devices = [device];
    let factory = Inkling2Sm121Factory::default();
    let mut cubins = Vec::new();
    for cache in ["f32", "e4m3"] {
        let table = table(&dir, cache);
        let start = Instant::now();
        let (identity, limits) = factory
            .describe(&table, &devices)
            .unwrap_or_else(|e| panic!("{}", e.message))
            .expect("the kind describes");
        eprintln!("describe ({cache}): {} ms", start.elapsed().as_millis());
        assert_eq!(
            limits,
            factory.plan(&table, &devices).unwrap(),
            "the limits equal the plan"
        );
        let start = Instant::now();
        let engine = factory
            .open(&table, &devices)
            .unwrap_or_else(|e| panic!("{}", e.message));
        eprintln!("open after describe ({cache}): {} ms", start.elapsed().as_millis());
        assert_eq!(engine.identity(), identity, "the open reports the described identity");
        assert_eq!(engine.limits(), limits);
        cubins.push(identity.numbers["cubin_blake3"].clone());
        drop(engine);
    }
    assert_ne!(cubins[0], cubins[1], "the cubin of the cache format is in the identity");
}

#[test]
fn an_open_that_reports_another_identity_than_its_description_is_refused() {
    let Some((dir, device)) = setup() else { return };
    let devices = [device];
    let factory = Inkling2Sm121Factory::default();
    let table = table(&dir, "f32");
    factory.describe(&table, &devices).unwrap().expect("the kind describes");
    factory.corrupt_descriptions_for_test();
    let error = match factory.open(&table, &devices) {
        Ok(_) => panic!("the open must refuse another identity than its description"),
        Err(error) => error.message,
    };
    assert!(
        error.contains("another identity than its description") && error.contains("restart"),
        "{error}"
    );
}

/// Print the time of an open with no description before it. Run it twice with one `XDG_CACHE_HOME`: the first run has a
/// cold digest cache, and the second has a warm one. The test asserts no time.
#[test]
fn the_open_prints_its_time() {
    let Some((dir, device)) = setup() else { return };
    let start = Instant::now();
    let engine = Inkling2Sm121Factory::default()
        .open(&table(&dir, "f32"), &[device])
        .unwrap_or_else(|e| panic!("{}", e.message));
    eprintln!("open with no description: {} ms", start.elapsed().as_millis());
    drop(engine);
}
