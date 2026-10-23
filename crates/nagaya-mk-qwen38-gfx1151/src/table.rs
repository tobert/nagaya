//! The weight table that the Rust host gives to mk's engine (`hip/src/engine/weight_table.h`).
//!
//! The host plans and fills the arena (`arena.rs`). The engine takes each tensor of the arena by name, in its final
//! layout, and reads no GGUF. The two calls of the pass build order the stages of a pass and give the footprint of a
//! stage (`schedule.rs`). Each call stops the program on an error, as mk's C++ code did.

use std::ffi::{CString, c_char};
use std::marker::PhantomData;

use crate::arena::{Arena, Planned};
use crate::schedule::call::{MkAccess, footprint_call, schedule_call};
use crate::stage::Stage;

/// `struct mk_weight` of `weight_table.h`.
#[repr(C)]
#[derive(Debug)]
pub(crate) struct MkWeight {
    pub name: *const c_char,
    pub kind: u32,
    pub pad: u32,
    pub ne: [u64; 3],
    pub row_bytes: u64,
    pub d: *const u8,
}

/// `struct mk_weight_table` of `weight_table.h`.
#[repr(C)]
pub(crate) struct MkWeightTable {
    pub n_weights: u64,
    pub weights: *const MkWeight,
    pub schedule: unsafe extern "C" fn(*const Stage, u32, *const u32, u32, u32, u32, *mut u32, *mut u32) -> u32,
    pub footprint: unsafe extern "C" fn(*const Stage, *mut MkAccess, u32, *mut u32) -> u32,
}

/// A weight table and the buffers that its pointers name. It borrows the arena, which holds the tensor data.
pub(crate) struct WeightTable<'a> {
    raw: MkWeightTable,
    _names: Vec<CString>,
    _weights: Vec<MkWeight>,
    _arena: PhantomData<&'a Arena>,
}

impl<'a> WeightTable<'a> {
    /// The table of every planned tensor at its place in `arena`.
    pub(crate) fn new(planned: &[Planned], arena: &'a Arena) -> Result<WeightTable<'a>, String> {
        let names: Vec<CString> = planned
            .iter()
            .map(|p| CString::new(p.name.as_bytes()).map_err(|_| format!("the tensor name {:?} holds a NUL", p.name)))
            .collect::<Result<_, _>>()?;
        let weights: Vec<MkWeight> = planned
            .iter()
            .zip(&names)
            .map(|(p, name)| {
                assert!(
                    p.offset + p.bytes <= arena.bytes,
                    "{} ends past the arena of {} bytes",
                    p.name,
                    arena.bytes
                );
                MkWeight {
                    name: name.as_ptr(),
                    kind: p.kind,
                    pad: 0,
                    ne: p.ne,
                    row_bytes: p.row_bytes,
                    // SAFETY: the offset is inside the arena (the assert above).
                    d: unsafe { arena.ptr.add(p.offset as usize) },
                }
            })
            .collect();
        let raw = MkWeightTable {
            n_weights: weights.len() as u64,
            weights: weights.as_ptr(),
            schedule: schedule_call,
            footprint: footprint_call,
        };
        // The vectors move into the struct below. Their heap buffers do not move, so the pointers stay valid.
        Ok(WeightTable {
            raw,
            _names: names,
            _weights: weights,
            _arena: PhantomData,
        })
    }

    /// The table for the engine's C ABI. It is valid while `self` lives.
    pub(crate) fn as_ptr(&self) -> *const MkWeightTable {
        &self.raw
    }
}

#[cfg(test)]
mod tests {
    use std::ffi::CStr;

    use super::WeightTable;
    use crate::arena::{Arena, Planned, Tile};

    fn planned(name: &str, kind: u32, offset: u64, bytes: u64) -> Planned {
        Planned {
            gguf: 0,
            name: name.to_owned(),
            kind,
            ne: [256, 16, 1],
            row_bytes: 210,
            tile: Tile::None,
            rows: 16,
            k: 256,
            src_bytes: bytes,
            bytes,
            offset,
        }
    }

    #[test]
    fn the_table_names_each_tensor_at_its_place_in_the_arena() {
        let mut memory = vec![0u8; 8192];
        let arena = Arena::for_test(memory.as_mut_ptr(), memory.len() as u64);
        let plan = [
            planned("output.weight", 114, 4096, 3360),
            planned("blk.0.ssm_a", 0, 7680, 192),
        ];
        let table = WeightTable::new(&plan, &arena).unwrap();
        let raw = unsafe { &*table.as_ptr() };
        assert_eq!(raw.n_weights, 2);
        let weights = unsafe { std::slice::from_raw_parts(raw.weights, 2) };
        let name = |index: usize| unsafe { CStr::from_ptr(weights[index].name) }.to_str().unwrap();
        assert_eq!((name(0), name(1)), ("output.weight", "blk.0.ssm_a"));
        assert_eq!(
            (weights[0].kind, weights[0].ne, weights[0].row_bytes),
            (114, [256, 16, 1], 210)
        );
        assert_eq!(weights[0].d, memory[4096..].as_ptr());
        assert_eq!(weights[1].d, memory[7680..].as_ptr());
    }

    #[test]
    #[should_panic(expected = "ends past the arena")]
    fn a_tensor_past_the_arena_stops_the_program() {
        let mut memory = vec![0u8; 4096];
        let arena = Arena::for_test(memory.as_mut_ptr(), memory.len() as u64);
        let _ = WeightTable::new(&[planned("x", 0, 4000, 128)], &arena);
    }
}
