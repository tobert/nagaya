//! The input channels of the UNet (spec 11.9b, "The 9-channel UNet"), with no card. The engine reads them from the
//! shape of `unet.conv_in.weight`: 4 or 9. Another count refuses the plan and the open. The arena holds the input of
//! that many channels.

use std::borrow::Cow;
use std::path::PathBuf;

use nagaya_mk_bmg::safetensors::{Dtype, MappedSafeTensors};
use nagaya_mk_sdxl_bmg::model::{check, unet_in_channels};
use nagaya_mk_sdxl_bmg::plan::{MAX_LATENT_ROWS, arena_bytes, limits};
use nagaya_mk_sdxl_bmg::spec::*;
use nagaya_mk_sdxl_bmg::weights::{Count, Item, Weights, items};

/// Weights that hold only `unet.conv_in.weight` of `shape`, in F16.
fn conv_in(shape: &[usize]) -> Weights {
    let item = Item {
        name: "unet.conv_in.weight".to_owned(),
        dtype: Dtype::F16,
        shape: shape.to_vec(),
        bytes: Cow::Owned(vec![0u8; shape.iter().product::<usize>() * 2]),
    };
    Weights::from_items(&[item], &mut Count::default()).unwrap()
}

#[test]
fn the_unet_takes_4_or_9_input_channels_from_conv_in() {
    assert_eq!(unet_in_channels(&conv_in(&[320, 4, 3, 3])), Ok(LATENT));
    assert_eq!(unet_in_channels(&conv_in(&[320, 9, 3, 3])), Ok(INPAINT_CHANNELS));
    for shape in [[320, 5, 3, 3], [320, 8, 3, 3], [640, 9, 3, 3], [320, 9, 1, 1]] {
        let error = unet_in_channels(&conv_in(&shape)).unwrap_err();
        assert!(
            error.contains("conv_in.weight") && error.contains(&format!("{shape:?}")),
            "{error}"
        );
    }
    let none = Weights::from_items(&[], &mut Count::default()).unwrap();
    assert!(
        unet_in_channels(&none)
            .unwrap_err()
            .contains("no tensor unet.conv_in.weight")
    );
}

/// The check of the plan and of the open refuses another channel count before it looks for the other tensors.
#[test]
fn the_check_refuses_another_channel_count_by_its_shape() {
    let error = check(&conv_in(&[320, 5, 3, 3])).unwrap_err();
    assert!(error.contains("[320, 5, 3, 3]"), "{error}");
    let error = check(&conv_in(&[320, 9, 3, 3])).unwrap_err();
    assert!(
        error.contains("lacks"),
        "a 9-channel UNet passes to the check of the graphs: {error}"
    );
}

/// The arena of a 9-channel UNet holds 5 more input values for each latent row of the largest batch, and the limits
/// count it. The arena of a 4-channel UNet does not change.
#[test]
fn the_arena_and_the_limits_count_the_input_of_9_channels() {
    let (four, nine) = (arena_bytes(LATENT), arena_bytes(INPAINT_CHANNELS));
    assert_eq!(nine - four, (5 * 2 * MAX_LATENT_ROWS * 4) as u64);
    let activations = |in_channels| {
        let limits = limits(1000, in_channels);
        let term = limits
            .memory
            .mandatory
            .iter()
            .find(|t| t.name == "activations")
            .unwrap();
        term.bytes
    };
    assert_eq!(activations(INPAINT_CHANNELS) - activations(LATENT), nine - four);
}

/// A single file with Stability's names and 9 input channels gives the same channel count (spec 11.9b). The test
/// writes a file that holds only `conv_in` under its name in a single file.
#[test]
fn a_single_file_with_9_input_channels_gives_9() {
    let directory = tempfile::tempdir().unwrap();
    let path: PathBuf = directory.path().join("inpaint.safetensors");
    let shape = [320usize, 9, 3, 3];
    let bytes = shape.iter().product::<usize>() * 2;
    let header = format!(
        r#"{{"model.diffusion_model.input_blocks.0.0.weight":{{"dtype":"F16","shape":[320,9,3,3],"data_offsets":[0,{bytes}]}}}}"#
    );
    let mut file = (header.len() as u64).to_le_bytes().to_vec();
    file.extend_from_slice(header.as_bytes());
    file.resize(file.len() + bytes, 0);
    std::fs::write(&path, file).unwrap();
    let maps = [("", MappedSafeTensors::open(&path).unwrap())];
    let all = items(&maps, true).unwrap();
    assert_eq!(all.len(), 1);
    assert_eq!(
        (all[0].name.as_str(), &all[0].shape[..]),
        ("unet.conv_in.weight", &shape[..])
    );
    let weights = Weights::from_items(&all, &mut Count::default()).unwrap();
    assert_eq!(unet_in_channels(&weights), Ok(INPAINT_CHANNELS));
}
