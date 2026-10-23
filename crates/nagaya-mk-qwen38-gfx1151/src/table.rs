//! The tensor table that the Rust reader gives to mk's engine (`hip/src/engine/tensor_table.h`).
//!
//! mk's engine parses no GGUF. It plans its arena from this table, copies the tensor data out of the reader's maps,
//! and gathers PLE rows from the map of the PLE table for its life. Two of its calls back into Rust drop the page cache
//! of a copied range and dequantize PLE rows. The two calls of the pass build order the stages of a pass and give the
//! footprint of a stage (`schedule.rs`). Each call stops the program on an error, as mk's C++ code did.

use std::ffi::{CString, c_char, c_void};
use std::marker::PhantomData;
use std::os::unix::ffi::OsStrExt;

use crate::dequant::{GgmlType, dequant_f32};
use crate::gguf::Model;
use crate::schedule::call::{MkAccess, footprint_call, schedule_call};
use crate::stage::Stage;

/// `struct mk_tensor` of `tensor_table.h`.
#[repr(C)]
#[derive(Debug)]
pub(crate) struct MkTensor {
    pub name: *const c_char,
    pub kind: u32,
    pub n_dims: u32,
    pub ne: [u64; 4],
    pub n_elements: u64,
    pub n_bytes: u64,
    pub data: *const u8,
}

/// `struct mk_tensor_table` of `tensor_table.h`.
#[repr(C)]
pub(crate) struct MkTensorTable {
    pub n_tensors: u64,
    pub tensors: *const MkTensor,
    pub n_shards: u64,
    pub shards: *const *const c_char,
    pub ple_eos_token_id: i64,
    pub n_ple_layer_multipliers: u64,
    pub ple_layer_multipliers: *const i64,
    pub n_ple_head_offsets: u64,
    pub ple_head_offsets: *const i64,
    pub n_ple_head_vocab_sizes: u64,
    pub ple_head_vocab_sizes: *const i64,
    pub reader: *const c_void,
    pub drop_cached: unsafe extern "C" fn(*const c_void, *const u8, u64),
    pub dequant_f32: unsafe extern "C" fn(u32, *const c_void, i64, *mut f32),
    pub schedule: unsafe extern "C" fn(*const Stage, u32, *const u32, u32, u32, u32, *mut u32, *mut u32) -> u32,
    pub footprint: unsafe extern "C" fn(*const Stage, *mut MkAccess, u32, *mut u32) -> u32,
}

/// The keys of the PLE hash that the engine reads (`engine.hip` `wire`, `ple_rows`).
const PLE_EOS: &str = "qwen4exp.ple.eos_token_id";
const PLE_MULTIPLIERS: &str = "qwen4exp.ple.layer_multipliers";
const PLE_OFFSETS: &str = "qwen4exp.ple.head_offsets";
const PLE_SIZES: &str = "qwen4exp.ple.head_vocab_sizes";

/// A tensor table and the buffers that its pointers name. It borrows the model, whose maps hold the tensor data.
pub(crate) struct TensorTable<'a> {
    raw: MkTensorTable,
    _names: Vec<CString>,
    _tensors: Vec<MkTensor>,
    _paths: Vec<CString>,
    _path_pointers: Vec<*const c_char>,
    _ple: [Vec<i64>; 3],
    _model: PhantomData<&'a Model>,
}

impl<'a> TensorTable<'a> {
    /// The table of every tensor of `model`, and the PLE keys of its metadata.
    pub(crate) fn new(model: &'a Model) -> Result<TensorTable<'a>, String> {
        let names: Vec<CString> = model
            .tensors()
            .iter()
            .map(|tensor| {
                CString::new(tensor.name.as_bytes())
                    .map_err(|_| format!("the tensor name {:?} holds a NUL", tensor.name))
            })
            .collect::<Result<_, _>>()?;
        let tensors: Vec<MkTensor> = model
            .tensors()
            .iter()
            .zip(&names)
            .enumerate()
            .map(|(index, (tensor, name))| MkTensor {
                name: name.as_ptr(),
                kind: tensor.kind.id(),
                n_dims: tensor.n_dims,
                ne: tensor.ne,
                n_elements: tensor.n_elements,
                n_bytes: tensor.n_bytes,
                data: model.tensor_bytes(index).as_ptr(),
            })
            .collect();
        let paths: Vec<CString> = model
            .shard_paths()
            .map(|path| {
                CString::new(path.as_os_str().as_bytes())
                    .map_err(|_| format!("the shard path {} holds a NUL", path.display()))
            })
            .collect::<Result<_, _>>()?;
        let path_pointers: Vec<*const c_char> = paths.iter().map(|path| path.as_ptr()).collect();
        let ple = [
            model.int_array(PLE_MULTIPLIERS)?,
            model.int_array(PLE_OFFSETS)?,
            model.int_array(PLE_SIZES)?,
        ];
        let raw = MkTensorTable {
            n_tensors: tensors.len() as u64,
            tensors: tensors.as_ptr(),
            n_shards: path_pointers.len() as u64,
            shards: path_pointers.as_ptr(),
            ple_eos_token_id: model.int(PLE_EOS)?,
            n_ple_layer_multipliers: ple[0].len() as u64,
            ple_layer_multipliers: ple[0].as_ptr(),
            n_ple_head_offsets: ple[1].len() as u64,
            ple_head_offsets: ple[1].as_ptr(),
            n_ple_head_vocab_sizes: ple[2].len() as u64,
            ple_head_vocab_sizes: ple[2].as_ptr(),
            reader: model as *const Model as *const c_void,
            drop_cached: drop_cached_call,
            dequant_f32: dequant_f32_call,
            schedule: schedule_call,
            footprint: footprint_call,
        };
        // The vectors move into the struct below. Their heap buffers do not move, so the pointers stay valid.
        Ok(TensorTable {
            raw,
            _names: names,
            _tensors: tensors,
            _paths: paths,
            _path_pointers: path_pointers,
            _ple: ple,
            _model: PhantomData,
        })
    }

    /// The table for the engine's C ABI. It is valid while `self` lives.
    pub(crate) fn as_ptr(&self) -> *const MkTensorTable {
        &self.raw
    }
}

/// The engine's `drop_cached`: `Model::drop_cached` on the reader. An error stops the program.
unsafe extern "C" fn drop_cached_call(reader: *const c_void, p: *const u8, n: u64) {
    // SAFETY: the table names this model as its reader, and the model outlives the engine (`TensorTable` borrows it).
    let model = unsafe { &*(reader as *const Model) };
    if let Err(error) = model.drop_cached(p, n) {
        // A panic cannot unwind out of an `extern "C"` call, so it stops the program after it prints the message.
        panic!("mk's engine asked the reader to drop a range: {error}");
    }
}

/// The engine's `dequant_f32`: `dequant::dequant_f32` on `n` values at `src`. An unknown type or a part of a block
/// stops the program.
unsafe extern "C" fn dequant_f32_call(kind: u32, src: *const c_void, n: i64, dst: *mut f32) {
    let kind = GgmlType::from_id(kind).unwrap_or_else(|| panic!("mk's engine asked to dequantize the type id {kind}"));
    let values = u64::try_from(n).unwrap_or_else(|_| panic!("mk's engine asked to dequantize {n} values"));
    // A part of a block gives a short `src`, and `dequant_f32` refuses it.
    let bytes = values / kind.block_values() * kind.block_bytes();
    // SAFETY: the engine gives `bytes` readable bytes at `src` (a PLE row in the reader's map) and `n` writable floats
    // at `dst`.
    let (src, dst) = unsafe {
        (
            std::slice::from_raw_parts(src as *const u8, bytes as usize),
            std::slice::from_raw_parts_mut(dst, values as usize),
        )
    };
    dequant_f32(kind, src, dst);
}

#[cfg(test)]
mod tests {
    use std::ffi::CStr;

    use super::{TensorTable, dequant_f32_call};
    use crate::dequant::GgmlType;
    use crate::gguf::Model;

    /// A shard with the PLE keys, one Q8_0 tensor of two rows and one F32 tensor.
    fn shard(directory: &std::path::Path) -> std::path::PathBuf {
        fn string(text: &str) -> Vec<u8> {
            let mut out = (text.len() as u64).to_le_bytes().to_vec();
            out.extend_from_slice(text.as_bytes());
            out
        }
        fn array(values: &[i64]) -> Vec<u8> {
            let mut out = 9u32.to_le_bytes().to_vec();
            out.extend_from_slice(&11u32.to_le_bytes());
            out.extend_from_slice(&(values.len() as u64).to_le_bytes());
            values
                .iter()
                .for_each(|value| out.extend_from_slice(&value.to_le_bytes()));
            out
        }
        let mut entries: Vec<(&str, Vec<u8>)> = vec![
            (
                "split.no",
                [2u32.to_le_bytes().as_slice(), &0u16.to_le_bytes()].concat(),
            ),
            (
                "split.count",
                [2u32.to_le_bytes().as_slice(), &1u16.to_le_bytes()].concat(),
            ),
            (
                "qwen4exp.ple.eos_token_id",
                [4u32.to_le_bytes(), 248_044u32.to_le_bytes()].concat(),
            ),
        ];
        entries.push(("qwen4exp.ple.layer_multipliers", array(&[3, 5, 7])));
        entries.push(("qwen4exp.ple.head_offsets", array(&[0, 10])));
        entries.push(("qwen4exp.ple.head_vocab_sizes", array(&[10, 11])));
        let mut bytes = 0x4655_4747u32.to_le_bytes().to_vec();
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&2u64.to_le_bytes());
        bytes.extend_from_slice(&(entries.len() as u64).to_le_bytes());
        for (key, value) in &entries {
            bytes.extend(string(key));
            bytes.extend_from_slice(value);
        }
        for (name, ne, kind, offset) in [("q.weight", [32u64, 2], 8u32, 0u64), ("f.weight", [4, 1], 0, 96)] {
            bytes.extend(string(name));
            bytes.extend_from_slice(&2u32.to_le_bytes());
            ne.iter().for_each(|d| bytes.extend_from_slice(&d.to_le_bytes()));
            bytes.extend_from_slice(&kind.to_le_bytes());
            bytes.extend_from_slice(&offset.to_le_bytes());
        }
        bytes.resize(bytes.len().div_ceil(32) * 32, 0);
        let data = bytes.len();
        // Two Q8_0 rows with d = 2.0 (0x4000) and values j, then four f32 values at the aligned 96.
        for _ in 0..2 {
            bytes.extend_from_slice(&0x4000u16.to_le_bytes());
            bytes.extend((0..32).map(|j| j as u8));
        }
        bytes.resize(data + 96, 0);
        [1.0f32, 2.0, 3.0, 4.0]
            .iter()
            .for_each(|v| bytes.extend_from_slice(&v.to_le_bytes()));
        let path = directory.join("one.gguf");
        std::fs::write(&path, bytes).unwrap();
        path
    }

    #[test]
    fn the_table_names_each_tensor_its_data_and_the_ple_keys() {
        let directory = tempfile::tempdir().unwrap();
        let path = shard(directory.path());
        let model = Model::open(std::slice::from_ref(&path)).unwrap();
        let table = TensorTable::new(&model).unwrap();
        // SAFETY: the table lives to the end of the test.
        let raw = unsafe { &*table.as_ptr() };
        assert_eq!(raw.n_tensors, 2);
        let tensors = unsafe { std::slice::from_raw_parts(raw.tensors, 2) };
        let name = |index: usize| unsafe { CStr::from_ptr(tensors[index].name) }.to_str().unwrap();
        assert_eq!((name(0), name(1)), ("q.weight", "f.weight"));
        assert_eq!(tensors[0].kind, 8);
        assert_eq!(tensors[0].ne, [32, 2, 1, 1]);
        assert_eq!((tensors[0].n_elements, tensors[0].n_bytes), (64, 68));
        assert_eq!(tensors[1].data, model.tensor_bytes(1).as_ptr());
        assert_eq!(raw.n_shards, 1);
        assert_eq!(
            unsafe { CStr::from_ptr(*raw.shards) }.to_bytes(),
            path.as_os_str().as_encoded_bytes()
        );
        assert_eq!(raw.ple_eos_token_id, 248_044);
        let ple = |pointer: *const i64, n: u64| unsafe { std::slice::from_raw_parts(pointer, n as usize) }.to_vec();
        assert_eq!(ple(raw.ple_layer_multipliers, raw.n_ple_layer_multipliers), [3, 5, 7]);
        assert_eq!(ple(raw.ple_head_offsets, raw.n_ple_head_offsets), [0, 10]);
        assert_eq!(ple(raw.ple_head_vocab_sizes, raw.n_ple_head_vocab_sizes), [10, 11]);
        assert_eq!(raw.reader, &model as *const Model as *const std::ffi::c_void);
    }

    #[test]
    fn the_calls_of_the_table_dequantize_and_drop_through_the_reader() {
        let directory = tempfile::tempdir().unwrap();
        let model = Model::open(&[shard(directory.path())]).unwrap();
        let table = TensorTable::new(&model).unwrap();
        let raw = unsafe { &*table.as_ptr() };
        let tensors = unsafe { std::slice::from_raw_parts(raw.tensors, 2) };
        // The second row of q.weight, as the engine gathers a PLE row: 34 bytes after the first.
        let mut row = [0f32; 32];
        unsafe {
            (raw.dequant_f32)(
                GgmlType::Q8_0.id(),
                tensors[0].data.add(34).cast(),
                32,
                row.as_mut_ptr(),
            )
        };
        assert_eq!(row.to_vec(), (0..32).map(|j| j as f32 * 2.0).collect::<Vec<_>>());
        let mut four = [0f32; 4];
        unsafe { (raw.dequant_f32)(0, tensors[1].data.cast(), 4, four.as_mut_ptr()) };
        assert_eq!(four, [1.0, 2.0, 3.0, 4.0]);
        unsafe { (raw.drop_cached)(raw.reader, tensors[0].data, 68) };
        assert_eq!(model.tensor_bytes(0)[..2], 0x4000u16.to_le_bytes());
    }

    #[test]
    fn a_model_without_the_ple_keys_has_no_table() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("bare.gguf");
        let mut bytes = 0x4655_4747u32.to_le_bytes().to_vec();
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.extend_from_slice(&2u64.to_le_bytes());
        for (key, value) in [("split.no", 0u32), ("split.count", 1)] {
            bytes.extend_from_slice(&(key.len() as u64).to_le_bytes());
            bytes.extend_from_slice(key.as_bytes());
            bytes.extend_from_slice(&4u32.to_le_bytes());
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes.resize(bytes.len().div_ceil(32) * 32, 0);
        std::fs::write(&path, bytes).unwrap();
        let model = Model::open(&[path]).unwrap();
        let error = TensorTable::new(&model).err().expect("no PLE keys");
        assert!(error.contains("qwen4exp.ple.layer_multipliers is missing"), "{error}");
    }

    /// The call refuses a part of a block. A panic in an `extern "C"` call stops the process, so the test makes the
    /// call in a child process and reads the child's message.
    #[test]
    fn a_part_of_a_block_stops_the_program() {
        if std::env::var_os("NAGAYA_TABLE_CHILD").is_some() {
            let mut out = [0f32; 31];
            let src = [0u8; 34];
            unsafe { dequant_f32_call(GgmlType::Q8_0.id(), src.as_ptr().cast(), 31, out.as_mut_ptr()) };
            return;
        }
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "table::tests::a_part_of_a_block_stops_the_program",
                "--nocapture",
            ])
            .env("NAGAYA_TABLE_CHILD", "1")
            .output()
            .unwrap();
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(!output.status.success(), "the child must stop: {stderr}");
        assert!(stderr.contains("not whole blocks"), "{stderr}");
    }
}
