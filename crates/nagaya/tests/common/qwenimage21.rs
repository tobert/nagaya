//! The model folders of Qwen-Image 2.1 for the tests that plan the Strix Halo example.
//!
//! The example profile names its files under `/srv/models`, as a deploy does. A test puts the folders of the machine
//! in their place: the base folder from `QWENIMAGE21_MODEL_DIR` and FUSAL's folder from `QWENIMAGE21_FUSAL_DIR`.
//! Neither variable has a default. An absent folder skips the test, and `NAGAYA_REQUIRE_MODELS=1` makes it fail.
#![allow(dead_code)]

use std::io::ErrorKind;
use std::path::{Path, PathBuf};

pub const MODEL_VARIABLE: &str = "QWENIMAGE21_MODEL_DIR";
pub const FUSAL_VARIABLE: &str = "QWENIMAGE21_FUSAL_DIR";
const REQUIRE_VARIABLE: &str = "NAGAYA_REQUIRE_MODELS";

/// The folders that the example profile names. The test replaces these prefixes.
const EXAMPLE_MODEL: &str = "/srv/models/qwen-image-2.1/Qwen-Image-2.1";
const EXAMPLE_FUSAL: &str = "/srv/models/qwen-image-2.1/Qwen-Image-2.1-Anime-Fusal";

fn required() -> bool {
    std::env::var(REQUIRE_VARIABLE).is_ok_and(|value| value == "1")
}

/// Skip the test, or panic when `NAGAYA_REQUIRE_MODELS=1`.
fn skip(test: &str, what: String) -> Option<PathBuf> {
    if required() {
        panic!(
            "{what}, and {REQUIRE_VARIABLE} is set. Set {MODEL_VARIABLE} and {FUSAL_VARIABLE} to the model folders."
        );
    }
    eprintln!("skipped: {test}: {what}");
    None
}

/// The folder that `variable` names. An unset variable or an absent folder skips. Any other error panics.
fn folder(variable: &str, test: &str) -> Option<PathBuf> {
    let Some(path) = std::env::var_os(variable).map(PathBuf::from) else {
        return skip(test, format!("{variable} is not set"));
    };
    match std::fs::metadata(&path) {
        Ok(metadata) if metadata.is_dir() => Some(path),
        Ok(_) => panic!(
            "{variable} names {}, which is not a folder. Set it to a folder.",
            path.display()
        ),
        Err(error) if error.kind() == ErrorKind::NotFound => skip(
            test,
            format!("the folder {} of {variable} does not exist", path.display()),
        ),
        Err(error) => panic!("cannot read the folder {} of {variable}: {error}", path.display()),
    }
}

/// The base folder and FUSAL's folder of this machine, or `None` when the test skips.
pub fn folders(test: &str) -> Option<(PathBuf, PathBuf)> {
    let model = folder(MODEL_VARIABLE, test)?;
    let fusal = folder(FUSAL_VARIABLE, test)?;
    Some((model, fusal))
}

/// Put the folders of this machine in place of the example's folders, in each path key of a Qwen-Image config.
/// `plan_dir` goes to `store`, because the plan reads no plan table. A path under no example folder panics: then
/// the example changed, and the test must change with it.
pub fn localize(config: &mut toml::Table, model: &Path, fusal: &Path, store: &Path) {
    for key in ["model", "encoder", "lora"] {
        let Some(value) = config.get(key) else { continue };
        let example = value
            .as_str()
            .unwrap_or_else(|| panic!("the key `{key}` is not a string"));
        let local = if let Ok(rest) = Path::new(example).strip_prefix(EXAMPLE_FUSAL) {
            fusal.join(rest)
        } else if let Ok(rest) = Path::new(example).strip_prefix(EXAMPLE_MODEL) {
            model.join(rest)
        } else {
            panic!("the example names `{key}` = {example}, under no folder that the test knows: change the test");
        };
        config.insert(key.to_owned(), toml::Value::String(local.display().to_string()));
    }
    if config.contains_key("plan_dir") {
        config.insert("plan_dir".to_owned(), toml::Value::String(store.display().to_string()));
    }
}
