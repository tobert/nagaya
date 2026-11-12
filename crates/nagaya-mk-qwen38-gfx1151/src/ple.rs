//! The PLE rows of a pass: the n-gram embedding that layer 1 reads (`Qwen4ExpTextNGramEmbedding.forward`).
//!
//! The reference's `engine.hip` `ple_rows` gathered them in C++ (reference commit 555a2a3). The host gathers them here,
//! bit for bit, from the PLE table in the GGUF map. The table stays in the map: its rows are scattered over about 29
//! GB, and a pass reads 16 rows for each token (milestone 15 step 6).

use crate::arena::PLE_TABLE;
use crate::dequant::{GgmlType, dequant_f32};
use crate::gguf::Model;

/// The compiled PLE reset token, `model.h` `kPleEos`. The GGUF key `qwen4exp.ple.eos_token_id` must agree.
pub(crate) const PLE_EOS: i64 = 248_044;
/// The values of one PLE row, `model.h` `kPleHeadDim`.
const HEAD_DIM: usize = 160;
/// The rows of one token, `model.h` `kPleHeads`: 8 for the 2-grams and 8 for the 3-grams.
const HEADS: usize = 16;
/// The values of one token's rows, `model.h` `kEmbd`.
pub(crate) const EMBD: usize = HEADS * HEAD_DIM;

/// The PLE table and the constants of its hash.
pub(crate) struct PleTable {
    /// The index of the table in the model.
    index: usize,
    kind: GgmlType,
    row_bytes: usize,
    n_rows: u64,
    multipliers: [i64; 3],
    offsets: [i64; HEADS],
    sizes: [i64; HEADS],
}

impl PleTable {
    /// The PLE table of `model`, and the keys of its hash. A file that does not agree with the compiled constants is
    /// an error.
    pub(crate) fn new(model: &Model) -> Result<PleTable, String> {
        let eos = model.int("qwen4exp.ple.eos_token_id")?;
        if eos != PLE_EOS {
            return Err(format!(
                "GGUF qwen4exp.ple.eos_token_id {eos} != compiled kPleEos {PLE_EOS}"
            ));
        }
        let shape = |key: &str, values: Vec<i64>, want: usize| {
            if values.len() == want {
                Ok(values)
            } else {
                Err(format!("the PLE key {key} holds {} values, want {want}", values.len()))
            }
        };
        let multipliers = shape(
            "qwen4exp.ple.layer_multipliers",
            model.int_array("qwen4exp.ple.layer_multipliers")?,
            3,
        )?;
        let offsets = shape(
            "qwen4exp.ple.head_offsets",
            model.int_array("qwen4exp.ple.head_offsets")?,
            HEADS,
        )?;
        let sizes = shape(
            "qwen4exp.ple.head_vocab_sizes",
            model.int_array("qwen4exp.ple.head_vocab_sizes")?,
            HEADS,
        )?;
        if sizes.iter().any(|size| *size <= 0) {
            return Err(format!("the PLE head vocab sizes {sizes:?} must be positive"));
        }
        let index = model
            .tensors()
            .iter()
            .position(|tensor| tensor.name == PLE_TABLE)
            .ok_or_else(|| format!("the model has no tensor {PLE_TABLE}"))?;
        let info = &model.tensors()[index];
        if info.ne[0] != HEAD_DIM as u64 {
            return Err(format!("PLE rows of {} values, want {HEAD_DIM}", info.ne[0]));
        }
        let n_rows = info.n_elements / info.ne[0];
        let row_bytes = (info.n_bytes / n_rows) as usize;
        let table = PleTable {
            index,
            kind: info.kind,
            row_bytes,
            n_rows,
            multipliers: multipliers.try_into().expect("three multipliers"),
            offsets: offsets.try_into().expect("sixteen offsets"),
            sizes: sizes.try_into().expect("sixteen sizes"),
        };
        // A fault should read its own page, not the readahead around it: `rows` asks for each row's pages itself.
        let bytes = model.tensor_bytes(index);
        advise(bytes.as_ptr(), bytes.len(), libc::MADV_RANDOM)
            .map_err(|error| format!("madvise(MADV_RANDOM) on the PLE table: {error}"))?;
        Ok(table)
    }

    /// The table rows of `tokens` after the history `hist`, 16 for each token, in order. A row past the table stops
    /// the program, because the hash or the file is wrong.
    fn row_ids(&self, hist: [i32; 2], tokens: &[i32]) -> Vec<u64> {
        let all: Vec<i64> = hist.iter().chain(tokens).map(|token| i64::from(*token)).collect();
        let mut ids = Vec::with_capacity(tokens.len() * HEADS);
        // The last eos strictly before p.
        let mut last_eos: i64 = -1;
        for (p, token) in all.iter().enumerate() {
            let p = p as i64;
            let pos_in_seg = p - (last_eos + 1);
            if p >= 2 {
                let tok: [i64; 3] = std::array::from_fn(|s| {
                    if pos_in_seg >= s as i64 {
                        all[(p - s as i64) as usize]
                    } else {
                        PLE_EOS
                    }
                });
                for n in 2..=3usize {
                    let mut mixed = (tok[0] as u64).wrapping_mul(self.multipliers[0] as u64);
                    for (token, multiplier) in tok.iter().zip(&self.multipliers).take(n).skip(1) {
                        mixed ^= (*token as u64).wrapping_mul(*multiplier as u64);
                    }
                    let m = mixed as i64;
                    for h in 0..8 {
                        let head = (n - 2) * 8 + h;
                        let v = self.sizes[head];
                        let id = ((m % v) + v) % v + self.offsets[head];
                        assert!(
                            id >= 0 && (id as u64) < self.n_rows,
                            "PLE row {id} out of {}",
                            self.n_rows
                        );
                        ids.push(id as u64);
                    }
                }
            }
            if *token == PLE_EOS {
                last_eos = p;
            }
        }
        ids
    }

    /// The PLE rows of `tokens` after the history `hist` into `out`, `EMBD` values for each token.
    pub(crate) fn rows(&self, model: &Model, hist: [i32; 2], tokens: &[i32], out: &mut [f32]) {
        assert_eq!(
            out.len(),
            tokens.len() * EMBD,
            "the PLE rows take EMBD values for each token"
        );
        let ids = self.row_ids(hist, tokens);
        let table = model.tensor_bytes(self.index);
        // First ask the kernel for every row's pages, so that the reads of a pass's rows overlap, then read the rows.
        for id in &ids {
            let start = *id as usize * self.row_bytes;
            // A hint: a fault still reads the page when the advice fails.
            let _ = advise(table[start..].as_ptr(), self.row_bytes, libc::MADV_WILLNEED);
        }
        for (index, id) in ids.iter().enumerate() {
            let start = *id as usize * self.row_bytes;
            let at = (index / HEADS) * EMBD + (index % HEADS) * HEAD_DIM;
            dequant_f32(
                self.kind,
                &table[start..start + self.row_bytes],
                &mut out[at..at + HEAD_DIM],
            );
        }
    }
}

/// `madvise` on the whole pages of `[p, p + len)`.
fn advise(p: *const u8, len: usize, advice: libc::c_int) -> Result<(), std::io::Error> {
    let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as usize;
    let start = p as usize & !(page - 1);
    let end = p as usize + len;
    if unsafe { libc::madvise(start as *mut libc::c_void, end - start, advice) } != 0 {
        return Err(std::io::Error::last_os_error());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn table() -> PleTable {
        PleTable {
            index: 0,
            kind: GgmlType::F32,
            row_bytes: HEAD_DIM * 4,
            n_rows: 1600,
            multipliers: [3, 5, 7],
            offsets: std::array::from_fn(|h| 100 * h as i64),
            sizes: std::array::from_fn(|h| 11 + h as i64),
        }
    }

    // The expected rows come from a Python port of the reference's `engine.hip` `ple_rows` (C division semantics).
    #[test]
    fn the_rows_of_a_token_after_an_empty_history_hash_with_eos() {
        let eos = PLE_EOS as i32;
        assert_eq!(
            table().row_ids([eos, eos], &[2]),
            [
                1, 106, 205, 300, 403, 510, 600, 700, 815, 902, 1005, 1100, 1212, 1314, 1402, 1516
            ]
        );
    }

    #[test]
    fn an_eos_starts_a_new_n_gram_segment() {
        let eos = PLE_EOS as i32;
        assert_eq!(
            table().row_ids([10, 20], &[30, eos, 40]),
            [
                7, 102, 210, 306, 402, 514, 611, 708, 806, 900, 1015, 1110, 1205, 1300, 1420, 1516, 0, 106, 202, 302,
                403, 502, 613, 706, 804, 918, 1002, 1108, 1216, 1314, 1408, 1512, 9, 108, 201, 304, 402, 504, 606, 702,
                816, 908, 1016, 1116, 1210, 1316, 1408, 1500
            ]
        );
    }
}
