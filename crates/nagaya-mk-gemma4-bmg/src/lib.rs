//! The Gemma 4 26B-A4B engine on two Arc Pro B70 cards: a copy of mkviolet's engine (milestone 15s, step 4).
//!
//! Amy, 2026-10-21: "copy code in, we own all of it 100%". The copy keeps mkviolet's behavior. The `Engine` trait
//! comes in step 5. The kernels are OpenCL C in `kernels/`, and `build.rs` compiles them with `ocloc` for `bmg-g31`.
//! The parts that touch a card need the feature `xe`. With no feature, the crate builds and tests with no card.

/// The kernel binary for `bmg-g31` that `build.rs` compiled from `kernels/megakernel.cl`.
#[cfg(feature = "xe")]
pub const MEGAKERNEL_BIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/megakernel_bmg.bin"));

/// The name of the kernel's entry point in `MEGAKERNEL_BIN`.
pub const MEGAKERNEL_ENTRY: &str = "mk_persistent_interpreter";

#[cfg(all(test, feature = "xe"))]
mod tests {
    use super::*;

    /// `build.rs` embeds a native module: `ocloc` writes an ELF file, and Level Zero loads it with
    /// `ZE_MODULE_FORMAT_NATIVE`.
    #[test]
    fn the_kernel_binary_is_an_elf_module() {
        assert!(
            MEGAKERNEL_BIN.len() > 4096,
            "the kernel binary holds {} bytes",
            MEGAKERNEL_BIN.len()
        );
        assert_eq!(&MEGAKERNEL_BIN[..4], b"\x7fELF");
    }
}
