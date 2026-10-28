//! The single file of SDXL base 1.0 (`sd_xl_base_1.0.safetensors`) against its folder in the layout of diffusers. Both
//! publish the same weights, so each tensor that the map of `single.rs` gives must equal the tensor of the folder. The
//! VAE is the exception: the single file holds the VAE of `vae_1_0/`, and `vae/` of the folder is a later VAE.
//! The test reads `SDXL_MODEL_DIR` and `SDXL_SINGLE_FILE`, and skips when either is absent or the folder has no
//! `vae_1_0/diffusion_pytorch_model.safetensors`.

use std::collections::BTreeMap;
use std::path::PathBuf;

use nagaya_mk_bmg::safetensors::MappedSafeTensors;
use nagaya_mk_sdxl_bmg::Config;
use nagaya_mk_sdxl_bmg::weights::{Count, Item, Weights, items, sources};

fn paths() -> Option<(PathBuf, PathBuf)> {
    let dir = std::env::var_os("SDXL_MODEL_DIR").map(PathBuf::from);
    let file = std::env::var_os("SDXL_SINGLE_FILE").map(PathBuf::from);
    if let (Some(dir), Some(file)) = (&dir, &file)
        && dir.join("unet").is_dir()
        && file.is_file()
        && dir.join(VAE_1_0).is_file()
    {
        return Some((dir.clone(), file.clone()));
    }
    let require = std::env::var("NAGAYA_REQUIRE_MODELS").is_ok_and(|v| v == "1");
    assert!(
        !require,
        "NAGAYA_REQUIRE_MODELS is set: set SDXL_MODEL_DIR and SDXL_SINGLE_FILE (sd_xl_base_1.0.safetensors)"
    );
    eprintln!("skipped: SDXL_MODEL_DIR or SDXL_SINGLE_FILE does not name the SDXL base 1.0 checkpoint");
    None
}

const VAE_1_0: &str = "vae_1_0/diffusion_pytorch_model.safetensors";

fn mapped(config: &Config) -> Vec<(&'static str, MappedSafeTensors)> {
    sources(config)
        .unwrap()
        .into_iter()
        .map(|(prefix, path)| (prefix, MappedSafeTensors::open(&path).unwrap()))
        .collect()
}

#[test]
fn the_single_file_gives_each_tensor_of_the_folder() {
    let Some((dir, file)) = paths() else { return };
    let folder = Config::folder(dir.clone());
    let single = Config {
        model_dir: dir,
        checkpoint: Some(file),
    };
    let (folder_maps, single_maps) = (mapped(&folder), mapped(&single));
    let vae_maps = [(
        "vae.",
        MappedSafeTensors::open(&folder.model_dir.join(VAE_1_0)).unwrap(),
    )];
    let mut want: BTreeMap<String, Item> = items(&folder_maps, false)
        .unwrap()
        .into_iter()
        .map(|item| (item.name.clone(), item))
        .collect();
    for item in items(&vae_maps, false).unwrap() {
        assert!(
            want.contains_key(&item.name),
            "{} is in vae_1_0 and not in vae",
            item.name
        );
        want.insert(item.name.clone(), item);
    }
    let got: BTreeMap<String, Item> = items(&single_maps, true)
        .unwrap()
        .into_iter()
        .map(|item| (item.name.clone(), item))
        .collect();
    let only_folder: Vec<&String> = want.keys().filter(|name| !got.contains_key(*name)).collect();
    let only_single: Vec<&String> = got.keys().filter(|name| !want.contains_key(*name)).collect();
    assert!(
        only_folder.is_empty() && only_single.is_empty(),
        "only in the folder: {only_folder:?}; only in the single file: {only_single:?}"
    );
    for (name, w) in &want {
        let g = &got[name];
        assert_eq!(g.shape, w.shape, "{name}");
        assert_eq!(
            g.dtype, w.dtype,
            "{name}: the VAE is F32 and the rest F16, as in the folder"
        );
        // The F32 values of vae_1_0 are each exact in F16, so the VAE of the single file gives the same bytes.
        assert!(g.bytes == w.bytes, "{name}: the bytes differ");
    }
    eprintln!("{} tensors, each with the same bytes", want.len());
}

#[test]
fn the_single_file_plans_as_the_folder_and_a_missing_tensor_is_an_error() {
    let Some((dir, file)) = paths() else { return };
    let folder = nagaya_mk_sdxl_bmg::plan::weights_bytes(&Config::folder(dir.clone())).unwrap();
    let single = Config {
        model_dir: dir,
        checkpoint: Some(file),
    };
    assert_eq!(nagaya_mk_sdxl_bmg::plan::weights_bytes(&single).unwrap(), folder);
    let maps = mapped(&single);
    let mut all = items(&maps, true).unwrap();
    let gone = "unet.up_blocks.1.attentions.2.transformer_blocks.0.attn2.to_v.weight";
    all.retain(|item| item.name != gone);
    let weights = Weights::from_items(&all, &mut Count::default()).unwrap();
    let error = nagaya_mk_sdxl_bmg::model::check(&weights).unwrap_err();
    assert!(error.contains(gone) && error.contains("lacks 1 tensors"), "{error}");
}
