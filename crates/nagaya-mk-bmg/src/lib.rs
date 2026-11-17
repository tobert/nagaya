//! The megakernel interpreter of the Arc Pro B70 cards (`bmg-g31`), which the engines of this target share.
//!
//! The interpreter is one OpenCL C kernel in `kernels/`. Each work-group walks a ring of task descriptors, and each
//! task names an op (`protocol`). `level_zero` opens the cards, loads a kernel binary and runs a ring. `build.rs`
//! compiles one binary for each engine with `ocloc`, so an op of one engine does not change the binary of another
//! (`AGENTS.md` milestone 17 step 4). The parts that touch a card need the feature `xe`.

// The copy of the code of the first Gemma 4 engine keeps these lints until the port of each part
// (`nagaya-mk-gemma4-bmg`).
#![allow(
    clippy::manual_div_ceil,
    clippy::missing_safety_doc,
    clippy::not_unsafe_ptr_arg_deref,
    clippy::too_many_arguments
)]

#[cfg(feature = "xe")]
pub mod level_zero;
pub mod protocol;
#[cfg(feature = "xe")]
pub mod ring;
pub mod safetensors;
pub mod tasks;
pub mod test_cards;

/// The kernel binary of the Gemma 4 engine for `bmg-g31`, from `kernels/megakernel.cl`.
#[cfg(feature = "xe")]
pub const GEMMA4_KERNEL: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/megakernel_bmg.bin"));

/// The kernel binary of the speech engine (Qwen3-TTS) for `bmg-g31`: `kernels/megakernel.cl` with `MK_QWEN3TTS`, which
/// adds the ops of `kernels/qwen3tts_ops.cl`.
#[cfg(feature = "xe")]
pub const QWEN3TTS_KERNEL: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/megakernel_qwen3tts_bmg.bin"));

/// The kernel binary of the image engine (SDXL) for `bmg-g31`: `kernels/megakernel.cl` with `MK_QWEN3TTS` and
/// `MK_SDXL`, which add the ops of `kernels/qwen3tts_ops.cl` and `kernels/sdxl_ops.cl`.
#[cfg(feature = "xe")]
pub const SDXL_KERNEL: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/megakernel_sdxl_bmg.bin"));

/// The kernel binary of the embedding engine (EmbeddingGemma 2) for `bmg-g31`: `kernels/megakernel.cl` with
/// `MK_QWEN3TTS` and `MK_EMBEDDINGGEMMA2`, which add the ops of `kernels/qwen3tts_ops.cl` and
/// `kernels/embeddinggemma2_ops.cl`. Its `OP_GEGLU` uses gelu_tanh, as the MLP of EmbeddingGemma 2 does.
#[cfg(feature = "xe")]
pub const EMBEDDINGGEMMA2_KERNEL: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/megakernel_embeddinggemma2_bmg.bin"));

/// The kernel binary of the LFM3 engine for `bmg-g31`: `kernels/megakernel.cl` with `MK_QWEN3TTS` and `MK_LFM3`, which
/// add the ops of `kernels/qwen3tts_ops.cl`, `kernels/lfm3_ops.cl`, `kernels/lfm3_vision_ops.cl` and
/// `kernels/embeddinggemma2_ops.cl`. Its `OP_RMSNORM` takes a partial rotation, and its `OP_GEGLU` is SwiGLU.
#[cfg(feature = "xe")]
pub const LFM3_KERNEL: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/megakernel_lfm3_bmg.bin"));

/// The name of the entry point of each kernel binary.
pub const KERNEL_ENTRY: &str = "mk_persistent_interpreter";

/// The version of `ocloc` that compiled the kernels, for the plan table lineage of an identity (spec 12.1).
#[cfg(feature = "xe")]
pub const OCLOC_VERSION: &str = env!("BMG_OCLOC_VERSION");

/// The revision of the graphics compiler inside `ocloc`, for the plan table lineage of an identity (spec 12.1).
#[cfg(feature = "xe")]
pub const IGC_REVISION: &str = env!("BMG_IGC_REVISION");

/// The index of the card at each wanted PCI address, in the order of `wanted`. `cards` holds the address of each card.
pub fn match_addresses(cards: &[String], wanted: &[String]) -> Result<Vec<usize>, String> {
    let mut found = Vec::with_capacity(wanted.len());
    for (index, address) in wanted.iter().enumerate() {
        let address = address.to_ascii_lowercase();
        if wanted[..index].iter().any(|other| other.eq_ignore_ascii_case(&address)) {
            return Err(format!("the profile names the PCI address {address} twice"));
        }
        let matches: Vec<usize> = (0..cards.len()).filter(|card| cards[*card] == address).collect();
        match matches[..] {
            [card] => found.push(card),
            [] => {
                return Err(format!(
                    "no Level Zero card has the PCI address {address}; the cards are at {}",
                    cards.join(", ")
                ));
            }
            _ => return Err(format!("two Level Zero cards give the PCI address {address}")),
        }
    }
    Ok(found)
}

#[cfg(all(test, feature = "xe"))]
mod tests {
    use super::*;

    /// `build.rs` embeds a native module: `ocloc` writes an ELF file, and Level Zero loads it with
    /// `ZE_MODULE_FORMAT_NATIVE`.
    #[test]
    fn the_kernel_binary_is_an_elf_module() {
        assert!(
            GEMMA4_KERNEL.len() > 4096,
            "the kernel binary holds {} bytes",
            GEMMA4_KERNEL.len()
        );
        assert_eq!(&GEMMA4_KERNEL[..4], b"\x7fELF");
    }
}
