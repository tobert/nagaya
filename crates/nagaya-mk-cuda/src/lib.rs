//! The shared persistent megakernel interpreter for CUDA GPUs (`sm_121`).
//!
//! `kernels/megakernel.cu` is compiled ahead of time by `nvcc` into one `.cubin` per engine under the feature `cuda`.
//! `driver` loads the `.cubin` through the CUDA Driver API (`libcuda`) and `ring` drives the 128-byte task ring, and `graph` plans its dependencies.
//! Without the feature `cuda`, the crate needs no GPU and no CUDA toolkit.

pub mod attn;
#[cfg(feature = "cuda")]
pub mod driver;
pub mod experts;
pub mod graph;
pub mod protocol;
pub mod ptxas;
pub mod repack;
#[cfg(feature = "cuda")]
pub mod ring;
pub mod safetensors;
pub mod test_cards;

/// Entry point of every CUDA persistent megakernel `.cubin`.
pub const KERNEL_ENTRY: &str = "mk_persistent_interpreter";

/// Compiled `sm_121a` `.cubin` for the Inkling 2 engine.
#[cfg(feature = "cuda")]
pub const INKLING2_CUBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/megakernel_inkling2_sm121.cubin"));

/// Compiled `sm_121a` `.cubin` for Inkling 2 profiling (`MK_PROFILE`).
#[cfg(feature = "cuda")]
pub const INKLING2_PROFILE_CUBIN: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/megakernel_inkling2_profile_sm121.cubin"));

/// Test cubin with a spin limit of 65,536 polls (`MK_SPIN_LIMIT`). A wait that never ends traps in about 10 ms.
/// A production engine must not load it.
#[cfg(feature = "cuda")]
pub const INKLING2_SPIN_TEST_CUBIN: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/megakernel_inkling2_spin_test_sm121.cubin"));

/// Version string reported by `nvcc --version` at build time, for engine identity (spec 12.1).
#[cfg(feature = "cuda")]
pub const NVCC_VERSION: &str = env!("CUDA_NVCC_VERSION");

/// Return the index of each wanted PCI address in `cards`, in the order of `wanted`.
pub fn match_addresses(cards: &[String], wanted: &[String]) -> Result<Vec<usize>, String> {
    let mut found = Vec::with_capacity(wanted.len());
    for (index, address) in wanted.iter().enumerate() {
        let address = address.to_ascii_lowercase();
        if wanted[..index].iter().any(|other| other.eq_ignore_ascii_case(&address)) {
            return Err(format!("the profile names the PCI address {address} twice"));
        }
        let matches: Vec<usize> = (0..cards.len())
            .filter(|card| cards[*card].eq_ignore_ascii_case(&address))
            .collect();
        match matches[..] {
            [card] => found.push(card),
            [] => {
                return Err(format!(
                    "no CUDA GPU has the PCI address {address}; the GPUs are at {}",
                    cards.join(", ")
                ));
            }
            _ => return Err(format!("two CUDA GPUs give the PCI address {address}")),
        }
    }
    Ok(found)
}

#[cfg(test)]
mod tests {
    use std::mem::{align_of, offset_of, size_of};

    use super::*;
    use crate::protocol::{self, BlockStats, ExpertPair, ExpertPairHeader, MkTaskDescriptor, MkTaskTable};
    use crate::safetensors::SafeTensorsHeader;

    #[test]
    fn descriptor_and_stats_layouts_are_fixed() {
        assert_eq!(size_of::<MkTaskDescriptor>(), 128);
        assert_eq!(align_of::<MkTaskDescriptor>(), 128);
        let offsets = [
            (offset_of!(MkTaskDescriptor, opcode), 0),
            (offset_of!(MkTaskDescriptor, flags), 2),
            (offset_of!(MkTaskDescriptor, items), 4),
            (offset_of!(MkTaskDescriptor, wait0_event), 8),
            (offset_of!(MkTaskDescriptor, wait0_target), 12),
            (offset_of!(MkTaskDescriptor, wait1_event), 16),
            (offset_of!(MkTaskDescriptor, wait1_target), 20),
            (offset_of!(MkTaskDescriptor, signal), 24),
            (offset_of!(MkTaskDescriptor, rows), 28),
            (offset_of!(MkTaskDescriptor, a), 32),
            (offset_of!(MkTaskDescriptor, b), 40),
            (offset_of!(MkTaskDescriptor, c), 48),
            (offset_of!(MkTaskDescriptor, d), 56),
            (offset_of!(MkTaskDescriptor, e), 64),
            (offset_of!(MkTaskDescriptor, f), 72),
            (offset_of!(MkTaskDescriptor, n), 80),
            (offset_of!(MkTaskDescriptor, k), 84),
            (offset_of!(MkTaskDescriptor, kt), 88),
            (offset_of!(MkTaskDescriptor, group_items), 92),
            (offset_of!(MkTaskDescriptor, item_bytes), 96),
            (offset_of!(MkTaskDescriptor, scale_bytes), 100),
            (offset_of!(MkTaskDescriptor, pos0), 104),
            (offset_of!(MkTaskDescriptor, layer), 108),
            (offset_of!(MkTaskDescriptor, p0), 112),
            (offset_of!(MkTaskDescriptor, p1), 116),
            (offset_of!(MkTaskDescriptor, p2), 120),
            (offset_of!(MkTaskDescriptor, p3), 124),
        ];
        for (index, (got, want)) in offsets.into_iter().enumerate() {
            assert_eq!(got, want, "descriptor field {index}");
        }
        assert_eq!(size_of::<BlockStats>(), 24);
        assert_eq!(offset_of!(BlockStats, tasks_run), 0);
        assert_eq!(offset_of!(BlockStats, pad), 4);
        assert_eq!(offset_of!(BlockStats, wait_ns), 8);
        assert_eq!(offset_of!(BlockStats, exec_ns), 16);
    }

    #[test]
    fn the_task_table_is_128_bytes_of_16_pointers_and_the_header_says_so() {
        assert_eq!(size_of::<MkTaskTable>(), 128);
        assert_eq!(align_of::<MkTaskTable>(), 128);
        assert_eq!(offset_of!(MkTaskTable, ptr), 0);
        assert_eq!(size_of::<[u64; protocol::TABLE_POINTERS]>(), 128);
        const { assert!(protocol::EXPERT_TABLE_AMAX < protocol::TABLE_POINTERS) };
        let header = include_str!(concat!(env!("OUT_DIR"), "/mk_protocol.h"));
        for line in [
            "#define MK_TABLE_BYTES 128\n",
            "#define MK_TABLE_ALIGN 128\n",
            "#define MK_TABLE_POINTERS 16\n",
            "#define MK_TABLE_OFF_ptr 0\n",
            "#define MK_EXPERT_TABLE_AMAX 0\n",
        ] {
            assert!(header.contains(line), "{line:?} is not in the header");
        }
    }

    #[test]
    fn the_expert_pair_and_its_list_have_the_layout_that_the_kernel_checks() {
        assert_eq!(size_of::<ExpertPair>(), 136);
        assert_eq!(offset_of!(ExpertPair, expert), 0);
        assert_eq!(offset_of!(ExpertPair, count), 4);
        assert_eq!(offset_of!(ExpertPair, x_row), 8);
        assert_eq!(offset_of!(ExpertPair, y_row), 40);
        assert_eq!(offset_of!(ExpertPair, weight), 72);
        assert_eq!(size_of::<ExpertPairHeader>(), protocol::EXPERT_PAIR_HEADER_BYTES);
        assert_eq!(offset_of!(ExpertPairHeader, pairs), 0);
        let header = include_str!(concat!(env!("OUT_DIR"), "/mk_protocol.h"));
        for line in [
            "#define MK_EXPERT_PAIR_BYTES 136\n",
            "#define MK_EXPERT_PAIR_HEADER_BYTES 16\n",
            "#define MK_EXPERT_PAIR_OFF_weight 72\n",
            "#define MK_EXPERT_PAIR_OFF_y_row 40\n",
            "#define MK_EXPERT_PAIR_HEADER_OFF_pairs 0\n",
        ] {
            assert!(header.contains(line), "{line:?} is not in the header");
        }
    }

    #[test]
    fn the_bound_of_the_pairs_is_exact_at_one_row_and_never_below_the_most_pairs() {
        assert_eq!(protocol::expert_pair_bound(1), 8);
        assert_eq!(protocol::expert_pair_bound(2), 16);
        assert_eq!(protocol::expert_pair_bound(16), 128);
        assert_eq!(protocol::expert_pair_bound(512), 8 * 512 / 16 + 128);
        // The bound never passes the 8 pairs of one row for each row, and it grows by 1 pair for each 2 rows after
        // 128 pairs of slack.
        for rows in 1..=1024usize {
            let bound = protocol::expert_pair_bound(rows);
            assert!(
                bound <= 8 * rows && bound <= 8 * rows / 16 + 128,
                "rows {rows}: bound {bound}"
            );
        }
    }

    #[test]
    fn the_header_of_the_kernel_comes_from_the_protocol() {
        let header = include_str!(concat!(env!("OUT_DIR"), "/mk_protocol.h"));
        let all = protocol::OPCODES
            .iter()
            .chain(protocol::PROBE_READ_VARIANTS)
            .chain(protocol::PROBE_MMA_KINDS)
            .chain(protocol::FLAGS);
        for (name, value) in all {
            assert!(
                header.contains(&format!("#define {name} {value}u\n")),
                "{name} is not in the header"
            );
        }
        assert!(header.contains("#define OP_COPY 21u\n"));
        assert!(header.contains("#define OP_HALT 255u\n"));
        assert!(header.contains("#define MK_DESC_BYTES 128\n"));
        assert!(header.contains("#define MK_DESC_OFF_a 32\n"));
        assert!(header.contains("#define MK_DESC_OFF_layer 108\n"));
        assert!(header.contains("#define MK_DESC_OFF_p3 124\n"));
        assert!(header.contains("#define MK_STATS_OFF_exec_ns 16\n"));
        assert!(header.contains("#define MK_PROFILE_WORDS 3u\n"));
        assert!(header.contains("#define MK_PROBE_ITEM_BYTES 32768\n"));
        assert!(header.contains("#define OP_PROBE_READ 230u\n"));
        assert!(header.contains("#define PROBE_READ_SMEM_BULK 4u\n"));
        assert!(header.contains("#define PROBE_MMA_MXFP8 2u\n"));
    }

    #[test]
    fn the_probe_smem_size_holds_the_stages_and_their_barriers() {
        assert_eq!(protocol::probe_smem_bytes(2), 2 * 16384 + 256);
        assert_eq!(protocol::probe_smem_bytes(6), 6 * 16384 + 256);
        let bytes = protocol::probe_smem_bytes(3);
        assert!(bytes >= 3 * protocol::PROBE_STAGE_BYTES + 3 * 8 && bytes.is_multiple_of(256));
    }

    #[test]
    fn opcodes_and_flags_are_distinct() {
        for table in [
            protocol::OPCODES,
            protocol::PROBE_READ_VARIANTS,
            protocol::PROBE_MMA_KINDS,
            protocol::FLAGS,
        ] {
            for (i, (name, value)) in table.iter().enumerate() {
                assert!(
                    table[..i].iter().all(|(other, v)| v != value && other != name),
                    "{name} = {value} repeats an earlier entry"
                );
            }
        }
    }

    #[test]
    fn each_opcode_sits_in_the_range_of_its_lane() {
        let lanes: [(&str, std::ops::RangeInclusive<u16>, &[&str]); 5] = [
            ("dense, lane D", 40..=49, &["OP_MATMUL_BF16", "OP_PROJ", "OP_OUT_MIX"]),
            (
                "MoE, lane M",
                50..=69,
                &[
                    "OP_QUANT_NVFP4",
                    "OP_QUANT_MXFP8",
                    "OP_EXPERT_NVFP4",
                    "OP_EXPERT_MXFP8",
                    "OP_ROUTE",
                ],
            ),
            (
                "rows, lane R",
                70..=79,
                &["OP_EMBED", "OP_MIX", "OP_GATE_DOWN", "OP_GATE_UP", "OP_MIX_OUT"],
            ),
            ("attention, lane A", 80..=89, &["OP_ATTN"]),
            ("shared", 0..=39, &["OP_NOP", "OP_COPY"]),
        ];
        let mut named = 0;
        for (lane, range, names) in &lanes {
            for name in *names {
                let value = protocol::OPCODES
                    .iter()
                    .find(|(known, _)| known == name)
                    .unwrap_or_else(|| panic!("{name} is not in the table of the opcodes"))
                    .1;
                assert!(
                    range.contains(&value),
                    "{name} = {value} is outside the range of {lane}"
                );
                named += 1;
            }
        }
        // Every opcode of the table is in a lane, or it is the halt, a probe or a test opcode (230 and up).
        for (name, value) in protocol::OPCODES {
            let in_lane = lanes
                .iter()
                .any(|(_, range, names)| range.contains(value) && names.contains(name));
            assert!(in_lane || *value >= 230, "{name} = {value} is in no lane");
        }
        assert_eq!(named, 16);
    }

    #[test]
    fn each_flag_is_one_bit_and_no_two_flags_share_a_bit() {
        let mut seen = 0u16;
        for (name, value) in protocol::FLAGS {
            assert_eq!(value.count_ones(), 1, "{name} = {value:#x} is not one bit");
            assert_eq!(seen & value, 0, "{name} = {value:#x} shares a bit with an earlier flag");
            seen |= value;
        }
        for name in [
            "FLAG_NORM",
            "FLAG_NORM_UNWEIGHTED",
            "FLAG_PAIRS",
            "FLAG_UP",
            "FLAG_DOWN",
        ] {
            assert!(
                protocol::FLAGS.iter().any(|(known, _)| *known == name),
                "{name} is not in the table"
            );
        }
        assert_eq!(protocol::FLAG_NORM, 1 << 3);
        assert_eq!(protocol::FLAG_NORM_UNWEIGHTED, 1 << 4);
        assert_eq!(protocol::FLAG_PAIRS, 1 << 5);
    }

    #[test]
    fn match_addresses_maps_case_insensitively_and_refuses_duplicates_and_missing() {
        let cards = vec!["0000:0a:00.0".to_owned(), "0000:08:00.0".to_owned()];
        assert_eq!(match_addresses(&cards, &["0000:0A:00.0".to_owned()]).unwrap(), vec![0]);
        let dup = match_addresses(&cards, &["0000:0a:00.0".to_owned(), "0000:0A:00.0".to_owned()]).unwrap_err();
        assert!(dup.contains("twice"), "{dup}");
        let missing = match_addresses(&cards, &["0000:99:00.0".to_owned()]).unwrap_err();
        assert!(missing.contains("0000:99:00.0"), "{missing}");
    }

    #[test]
    fn test_cards_parser_checks_format_duplicates_and_count() {
        assert!(test_cards::parse(None, 1).is_err());
        assert!(test_cards::parse(Some("not-a-pci"), 1).is_err());
        assert!(test_cards::parse(Some("0000:0a:00.0,0000:0a:00.0"), 2).is_err());
        assert!(test_cards::parse(Some("0000:0a:00.0"), 2).is_err());
        assert_eq!(
            test_cards::parse(Some("0000:0A:00.0"), 1).unwrap(),
            vec!["0000:0a:00.0".to_owned()]
        );
    }

    #[test]
    fn safetensors_header_parses_valid_and_refuses_truncated_or_oob() {
        let dir = std::env::temp_dir().join(format!("nagaya-mk-cuda-st-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();

        // Valid safetensors file with 1 tensor of 8 bytes.
        let valid_path = dir.join("valid.safetensors");
        let header = br#"{"w":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#;
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header);
        bytes.extend_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8]);
        std::fs::write(&valid_path, &bytes).unwrap();
        let parsed = SafeTensorsHeader::read(&valid_path).unwrap();
        assert_eq!(parsed.tensors["w"].nbytes, 8);
        assert_eq!(parsed.tensors["w"].shape, vec![2]);

        // Truncated file (< 8 bytes).
        let trunc_path = dir.join("trunc.safetensors");
        std::fs::write(&trunc_path, b"123").unwrap();
        assert!(SafeTensorsHeader::read(&trunc_path).is_err());

        // Out-of-bounds offset.
        let oob_path = dir.join("oob.safetensors");
        let oob_header = br#"{"w":{"dtype":"F32","shape":[4],"data_offsets":[0,16]}}"#;
        let mut oob = (oob_header.len() as u64).to_le_bytes().to_vec();
        oob.extend_from_slice(oob_header);
        oob.extend_from_slice(&[0u8; 8]);
        std::fs::write(&oob_path, &oob).unwrap();
        assert!(SafeTensorsHeader::read(&oob_path).is_err());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn compiled_cubins_are_distinct_elf_modules() {
        assert!(!NVCC_VERSION.is_empty());
        assert!(
            INKLING2_CUBIN.len() > 4096,
            "INKLING2_CUBIN holds {} bytes",
            INKLING2_CUBIN.len()
        );
        assert_eq!(&INKLING2_CUBIN[..4], b"\x7fELF");
        assert!(
            INKLING2_PROFILE_CUBIN.len() > 4096,
            "INKLING2_PROFILE_CUBIN holds {} bytes",
            INKLING2_PROFILE_CUBIN.len()
        );
        assert_eq!(&INKLING2_PROFILE_CUBIN[..4], b"\x7fELF");
        assert_ne!(INKLING2_CUBIN, INKLING2_PROFILE_CUBIN);
    }
}
