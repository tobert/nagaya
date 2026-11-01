//! The plan of the engine from its config and the headers of its files (spec 2.4). The weight count needs the files,
//! and skips when they are absent; the config checks need nothing.

mod common;

use nagaya_mk_qwenimage21_gfx1151::plan::{self, Config};

fn table(text: &str) -> toml::Table {
    text.parse().unwrap()
}

#[test]
fn the_config_names_each_file_and_refuses_an_unknown_key() {
    let config = Config::from_toml(&table(
        "model = \"/m\"\nencoder = \"/e.safetensors\"\nlora = \"/l.safetensors\"\nlora_strength = 0.5\nplan_dir = \"/p\"",
    ))
    .unwrap();
    assert_eq!(config.lora, Some(("/l.safetensors".into(), 0.5)));
    assert_eq!(config.plan_dir, Some("/p".into()));
    assert!(config.flash, "the flash kernel is the default");
    let gemm = Config::from_toml(&table("model = \"/m\"\nencoder = \"/e\"\nattention = \"gemm\"")).unwrap();
    assert!(!gemm.flash);
    let plain = Config::from_toml(&table("model = \"/m\"\nencoder = \"/e\"")).unwrap();
    assert_eq!((plain.lora, plain.plan_dir), (None, None));
    for (text, word) in [
        ("model = \"/m\"", "`encoder`"),
        ("model = \"/m\"\nencoder = \"/e\"\nvae = \"/v\"", "`vae`"),
        ("model = \"/m\"\nencoder = 3", "`encoder`"),
        ("model = \"/m\"\nencoder = \"/e\"\nlora_strength = 1.0", "no `lora`"),
        ("model = \"/m\"\nencoder = \"/e\"\nattention = \"fast\"", "`attention`"),
    ] {
        let error = Config::from_toml(&table(text)).unwrap_err();
        assert!(error.contains(word), "{text}: {error}");
    }
}

#[test]
fn the_plan_counts_the_bf16_weights_of_the_load() {
    let Some(model) = common::model_dir() else { return };
    let Some(fusal) = common::fusal_dir() else { return };
    let Some(oracle) = common::oracle() else { return };
    let encode =
        nagaya_mk_qwenimage21_gfx1151::safetensors::Safetensors::open(&oracle.join("encode.safetensors")).unwrap();
    let request: serde_json::Value = serde_json::from_str(&encode.metadata()["request"]).unwrap();
    let config = Config {
        model,
        encoder: fusal.join(request["encoder"].as_str().unwrap()),
        lora: None,
        plan_dir: None,
        flash: false,
    };
    let bytes = plan::weights_bytes(&config).unwrap();
    let gib = bytes as f64 / (1u64 << 30) as f64;
    // The text tower is 14.10 GiB and the transformer 13.25 GiB (spec 11.9c); the VAE decoder adds less than 1 GiB.
    assert!((27.35..28.4).contains(&gib), "{gib:.3} GiB");
    let limits = plan::plan(&config).unwrap();
    assert_eq!(limits.memory.mandatory[0].bytes, bytes);
    eprintln!(
        "weights {gib:.3} GiB, scratch {:.2} GiB",
        plan::scratch_bytes() as f64 / (1u64 << 30) as f64
    );
}
