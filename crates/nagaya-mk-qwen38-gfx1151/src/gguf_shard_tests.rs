//! The Rust reader and dequantizer against the reference's C++ copy, on the real Qwen3.8 shards.
//!
//! `tests/fixtures/dequant/shards.txt` holds what the reference's C++ reader found in the three shards: each tensor's
//! record and the SHA-256 of its dequant. `tools/mk_dequant_digests.cpp` wrote it, with the reference's `gguf.cpp` and
//! `dequant.cpp` as they were copied from the reference. The tool and the C++ copy are in
//! commit `7d474c2`. A machine with no copy of the shards skips these tests
//! (`nagaya-text-qwen38/tests/common/model.rs`).

use std::path::PathBuf;

use sha2::{Digest, Sha256};

use crate::gguf::Model;

#[cfg(not(target_endian = "little"))]
compile_error!("the fixture hashes little-endian f32 bytes");

#[path = "../../nagaya-text-qwen38/tests/common/model.rs"]
mod model_file;

/// The fixture that the reference's C++ copy wrote.
const SHARDS: &str = include_str!("../tests/fixtures/dequant/shards.txt");

/// The part of the first shard's name that names its place in the split.
const FIRST: &str = "-00001-of-00003";

/// The three shard paths, from the path of the first shard.
fn shard_paths() -> Vec<PathBuf> {
    let first = model_file::gguf_path();
    let name = first.to_str().expect("a UTF-8 shard path");
    assert!(
        name.contains(FIRST),
        "the first shard {name} does not name its place {FIRST}; set the variable to the first of three shards"
    );
    (1..=3)
        .map(|place| PathBuf::from(name.replace(FIRST, &format!("-0000{place}-of-00003"))))
        .collect()
}

/// One tensor line of the fixture.
struct Row {
    index: usize,
    name: String,
    kind: String,
    n_dims: u32,
    ne: [u64; 4],
    n_bytes: u64,
    shard: usize,
    offset: u64,
    full: String,
    sample: String,
}

/// The lines of the fixture that start with `tag`, split into words after the tag.
fn lines(tag: &str) -> impl Iterator<Item = Vec<&'static str>> {
    SHARDS
        .lines()
        .filter(move |line| line.split(' ').next() == Some(tag))
        .map(|line| line.split(' ').skip(1).collect())
}

fn rows() -> Vec<Row> {
    let number = |word: &str| word.parse::<u64>().expect("a number");
    let rows: Vec<Row> = lines("tensor")
        .map(|w| Row {
            index: number(w[0]) as usize,
            name: w[1].to_owned(),
            kind: w[2].to_owned(),
            n_dims: number(w[3]) as u32,
            ne: [number(w[4]), number(w[5]), number(w[6]), number(w[7])],
            n_bytes: number(w[8]),
            shard: number(w[9]) as usize,
            offset: number(w[10]),
            full: w[11].to_owned(),
            sample: w[12].to_owned(),
        })
        .collect();
    let count: Vec<_> = lines("tensors").collect();
    assert_eq!(count, [vec!["1224"]], "the fixture names its tensor count");
    assert_eq!(rows.len(), 1224);
    rows
}

fn hex(hash: Sha256) -> String {
    hash.finalize().iter().map(|byte| format!("{byte:02x}")).collect()
}

/// The SHA-256 of the f32 dequant of `rows` of tensor `index`, in order.
fn rows_digest(model: &Model, index: usize, rows: impl IntoIterator<Item = (u64, u64)>) -> String {
    let ne0 = model.tensors()[index].ne[0];
    let mut hash = Sha256::new();
    let mut buffer = Vec::new();
    for (row0, row1) in rows {
        buffer.resize(((row1 - row0) * ne0) as usize, 0f32);
        model.dequant_rows(index, row0, row1, &mut buffer).expect("dequant");
        // SAFETY: f32 has no padding, and the bytes are little-endian on x86_64, as the tool hashed them.
        hash.update(unsafe { std::slice::from_raw_parts(buffer.as_ptr() as *const u8, buffer.len() * 4) });
    }
    hex(hash)
}

/// Rows 0, rows/2 and rows-1, each once, as the tool picks them.
fn sample_rows(rows: u64) -> Vec<(u64, u64)> {
    let mut pick = vec![0, rows / 2, rows - 1];
    pick.dedup();
    pick.into_iter().map(|row| (row, row + 1)).collect()
}

#[test]
fn the_tensor_table_is_the_table_of_the_cpp_reader() {
    if model_file::absent() {
        return;
    }
    let paths = shard_paths();
    let model = Model::open(&paths).expect("open the shards");
    let shards: Vec<Vec<&str>> = lines("shard").collect();
    assert_eq!(shards.len(), 3);
    for (place, words) in shards.iter().enumerate() {
        assert_eq!(words[0], place.to_string());
        assert_eq!(paths[place].file_name().unwrap().to_str(), Some(words[1]));
        assert_eq!(model.sizes()[place].to_string(), words[2], "the size of shard {place}");
        assert_eq!(
            model.data_offsets()[place].to_string(),
            words[3],
            "the data offset of shard {place}"
        );
    }
    for words in lines("key") {
        let key = words[0];
        let values: Vec<i64> = words[1..]
            .iter()
            .map(|word| word.parse().expect("an integer"))
            .collect();
        if values.len() == 1 && key.ends_with("eos_token_id") {
            assert_eq!(model.int(key).unwrap(), values[0], "{key}");
        } else {
            assert_eq!(model.int_array(key).unwrap(), values, "{key}");
        }
    }
    let rows = rows();
    assert_eq!(model.tensors().len(), rows.len());
    for (tensor, row) in model.tensors().iter().zip(&rows) {
        assert_eq!(
            (
                tensor.name.as_str(),
                tensor.kind.name(),
                tensor.n_dims,
                tensor.ne,
                tensor.n_bytes,
                tensor.shard,
                tensor.offset
            ),
            (
                row.name.as_str(),
                row.kind.as_str(),
                row.n_dims,
                row.ne,
                row.n_bytes,
                row.shard,
                row.offset
            ),
            "tensor {}",
            row.index
        );
    }
}

#[test]
fn each_tensor_dequantizes_to_the_bits_of_the_cpp_copy_on_three_rows() {
    if model_file::absent() {
        return;
    }
    let model = Model::open(&shard_paths()).expect("open the shards");
    let rows = rows();
    let wrong: Vec<&str> = rows
        .iter()
        .filter(|row| rows_digest(&model, row.index, sample_rows(model.tensors()[row.index].rows())) != row.sample)
        .map(|row| row.name.as_str())
        .collect();
    assert!(
        wrong.is_empty(),
        "{} tensors differ from the reference's C++ dequant: {wrong:?}",
        wrong.len()
    );
}

/// The rows of one chunk of the full digest: at most 16 Mi values, as the tool reads them. The digest does not
/// depend on the chunk size.
fn chunks(rows: u64, ne0: u64) -> Vec<(u64, u64)> {
    let per = ((16u64 << 20) / ne0).max(1);
    (0..rows)
        .step_by(per as usize)
        .map(|row| (row, (row + per).min(rows)))
        .collect()
}

#[test]
#[ignore = "reads all 94 GB of the shards and hashes 700 GB of f32; run it with --release --ignored"]
fn each_tensor_dequantizes_to_the_bits_of_the_cpp_copy_on_every_row() {
    if model_file::absent() {
        return;
    }
    let model = Model::open(&shard_paths()).expect("open the shards");
    let rows = rows();
    let next = std::sync::atomic::AtomicUsize::new(0);
    let threads = std::thread::available_parallelism().map_or(1, |count| count.get());
    let wrong: Vec<String> = std::thread::scope(|scope| {
        let workers: Vec<_> = (0..threads)
            .map(|_| {
                scope.spawn(|| {
                    let mut wrong = Vec::new();
                    loop {
                        let at = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        let Some(row) = rows.get(at) else {
                            return wrong;
                        };
                        let tensor = &model.tensors()[row.index];
                        if rows_digest(&model, row.index, chunks(tensor.rows(), tensor.ne[0])) != row.full {
                            wrong.push(row.name.clone());
                        }
                    }
                })
            })
            .collect();
        workers
            .into_iter()
            .flat_map(|worker| worker.join().expect("a worker"))
            .collect()
    });
    assert!(
        wrong.is_empty(),
        "{} tensors differ from the reference's C++ dequant: {wrong:?}",
        wrong.len()
    );
}
